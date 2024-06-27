# !pip install -U accelerate
# !pip install -U transformers
# !pip install datasets
# !pip install evaluate
# !pip install mlflow
# !pip install transformers[torch]
# !pip install jiwer
# !pip install nlptutti

from datasets import load_dataset
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from transformers import WhisperTokenizer, WhisperFeatureExtractor, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import mlflow
from mlflow.tracking.client import MlflowClient
import subprocess
from huggingface_hub import create_repo, Repository
import os
import shutil
import math # 임시 테스트용
model_dir = "./tmp" # 수정 X


#########################################################################################################################################
################################################### 사용자 설정 변수 #####################################################################
#########################################################################################################################################

model_description = """
- 파인튜닝 데이터셋 : maxseats/aihub-464-preprocessed-680GB-set-12-testing

# 설명
- AI hub의 주요 영역별 회의 음성 데이터셋을 학습 중이에요.
- 680GB 중 set_0~4 데이터(40GB)까지 파인튜닝한 모델을 불러와서, set_5 데이터(10GB)를 학습한 모델입니다.
- 링크 : https://huggingface.co/datasets/maxseats/aihub-464-preprocessed-680GB-set-5-testing
"""

model_name = "maxseats/SungBeom-whisper-small-ko-set4"          # 대안 : "SungBeom/whisper-small-ko"
dataset_name = "maxseats/aihub-464-preprocessed-680GB-set-5"    # 불러올 데이터셋(허깅페이스 기준)
repo_name = 'maxseats/SungBeom-whisper-small-ko-set5'                                          # 허깅페이스 레포지토리 이름 설정

CACHE_DIR = '/mnt/a/maxseats/.finetuning_cache'     # 캐시 디렉토리 지정
is_test = False                                     # True: 소량의 샘플 데이터로 테스트, False: 실제 파인튜닝

token = "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"     # 허깅페이스 토큰 입력

training_args = Seq2SeqTrainingArguments(
    output_dir=model_dir,               # 원하는 리포지토리 이름을 입력한다.
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,      # 배치 크기가 2배 감소할 때마다 2배씩 증가
    learning_rate=1e-5,
    warmup_steps=500,
    # max_steps=2,                      # epoch 대신 설정
    num_train_epochs=1,                 # epoch 수 설정 / max_steps와 이것 중 하나만 설정
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="cer",        # 한국어의 경우 'wer'보다는 'cer'이 더 적합할 것
    greater_is_better=False,
    push_to_hub=True,
    save_total_limit=5,                 # 최대 저장할 모델 수 지정
    hub_token=token,                    # 허깅페이스 토큰
)

#########################################################################################################################################
################################################### 사용자 설정 변수 #####################################################################
#########################################################################################################################################


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # 인풋 데이터와 라벨 데이터의 길이가 다르며, 따라서 서로 다른 패딩 방법이 적용되어야 한다. 그러므로 두 데이터를 분리해야 한다.
        # 먼저 오디오 인풋 데이터를 간단히 토치 텐서로 반환하는 작업을 수행한다.
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Tokenize된 레이블 시퀀스를 가져온다.
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # 레이블 시퀀스에 대해 최대 길이만큼 패딩 작업을 실시한다.
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # 패딩 토큰을 -100으로 치환하여 loss 계산 과정에서 무시되도록 한다.
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # 이전 토크나이즈 과정에서 bos 토큰이 추가되었다면 bos 토큰을 잘라낸다.
        # 해당 토큰은 이후 언제든 추가할 수 있다.
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # pad_token을 -100으로 치환
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # metrics 계산 시 special token들을 빼고 계산하도록 설정
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    cer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}


# GPU 사용 가능 여부 확인
def check_GPU():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("GPU is available. Using GPU.")
    else:
        device = torch.device('cpu')
        print("GPU is not available. Using CPU.")
    return device


# model_dir, ./repo 초기화
def init_dirs():
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
        os.makedirs(model_dir)

    if os.path.exists('./repo'):
        shutil.rmtree('./repo')
        os.makedirs('./repo')


# mlflow 관련 변수 설정
def set_mlflow_env(model_name):
    # MLflow UI 관리 폴더 지정
    mlflow.set_tracking_uri("sqlite:////mnt/a/maxseats/mlflow.db")

    experiment_name = model_name    # MLflow 실험 이름 -> 모델 이름으로 설정
    existing_experiment = mlflow.get_experiment_by_name(experiment_name)

    if existing_experiment is not None:
        experiment_id = existing_experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(experiment_name)

    return experiment_id


# 파인튜닝을 진행하고자 하는 모델의 processor, tokenizer, feature extractor, model, data_collator, metric 로드
def get_model_variable(model_name, device):

    processor = WhisperProcessor.from_pretrained(model_name, language="Korean", task="transcribe")
    tokenizer = WhisperTokenizer.from_pretrained(model_name, language="Korean", task="transcribe")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    metric = evaluate.load('cer')

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.to(device)
    
    return processor, tokenizer, feature_extractor, model, data_collator, metric

def login_huggingface(token):
    while True: # 로그인
        if token =="exit":
            break
        try:
            result = subprocess.run(["huggingface-cli", "login", "--token", token])
            if result.returncode != 0:
                raise Exception()
            break
        except Exception as e:
            token = input("Please enter your Hugging Face API token: ")

    os.environ["HUGGINGFACE_HUB_TOKEN"] = token

# model_dir 필요한 파일 복사
def copy_model(moder_dir, tokenizer):
    
    max_depth = 1  # 순회할 최대 깊이
    
    for root, dirs, files in os.walk(model_dir):
        depth = root.count(os.sep) - model_dir.count(os.sep)
        if depth < max_depth:
            for file in files:
                # 파일 경로 생성
                source_file = os.path.join(root, file)
                # 대상 폴더에 복사
                shutil.copy(source_file, './repo')
    # 토크나이저 다운로드 및 로컬 디렉토리에 저장
    tokenizer.save_pretrained('./repo')

# 허깅페이스 readme 작성
def write_readme(dataset_name, model_name, model_description):

    readme = f"""
---
language: ko
tags:
- whisper
- speech-recognition
datasets:
- {dataset_name}
metrics:
- cer
---
# Model Name : {model_name}
# Description
{model_description}
    """

    # 모델 카드 및 기타 메타데이터 파일 작성
    with open("./repo/README.md", "w") as f:
        f.write(readme)

# 허깅페이스로 모델 업로드
def upload_huggingface(token, repo_name, model_dir, tokenizer, dataset_name, model_name, model_description):
    login_huggingface(token)
    create_repo(repo_name, exist_ok=True)
    repo = Repository(local_dir='./repo', clone_from=f"{repo_name}")
    copy_model(model_dir, tokenizer)
    write_readme(dataset_name, model_name, model_description)
    repo.push_to_hub(commit_message="Initial commit")
    
# 라벨 추가 전처리
def additional_preprocess(text):

    text = re.sub(r'/\([^\)]+\)', '', text)  # /( *) 패턴 제거, /(...) 형식 제거
    text = text.replace('\n', '')
    # text = re.sub(r'\(([^/]+)\/([^)]+)\)', r'\1', text)
    
    NOISE = ['o', 'n', 'u', 'b', 'l']
    EXCEPT = ['?', '!', '/', '+', '*', '-', '@', '$', '^', '&', '[', ']', '=', ':', ';', '.', ',', '(', ')']
    
    for noise in NOISE:
        text = text.replace(noise+'/', '')
        
    for specialChar in EXCEPT:
        text = text.replace(specialChar, '')

    text = text.replace('  ', ' ')
    return text.strip()

def preprocess_batch(batch):
    
    # transcripts 컬럼 전처리
    batch["transcripts"] = additional_preprocess( batch["transcripts"] )

    # label ids로 변환
    batch["labels"] = tokenizer(batch["transcripts"]).input_ids

    return batch    


#############################################################   MAIN 시작   ############################################################################


for set_num in range(5, 13): # 학습할 데이터셋 번호 지정

    model_description = f"""
    - 파인튜닝 데이터셋 : maxseats/aihub-464-preprocessed-680GB-set-{set_num}

    # 설명
    - AI hub의 주요 영역별 회의 음성 데이터셋을 학습 중이에요.
    - 680GB 중 set_0~{set_num-1} 데이터({set_num-1}0GB)까지 파인튜닝한 모델을 불러와서, set_{set_num} 데이터(10GB)를 학습한 모델입니다.
    - 링크 : https://huggingface.co/datasets/maxseats/aihub-464-preprocessed-680GB-set-{set_num}-testing
    """

    model_name = f"maxseats/SungBeom-whisper-small-ko-set{set_num-1}"          # 대안 : "SungBeom/whisper-small-ko"
    dataset_name = f"maxseats/aihub-464-preprocessed-680GB-set-{set_num}"    # 불러올 데이터셋(허깅페이스 기준)
    repo_name = f"maxseats/SungBeom-whisper-small-ko-set{set_num}"                                          # 허깅페이스 레포지토리 이름 설정



    device = check_GPU()         # GPU 사용 설정
    init_dirs() # model_dir, ./repo 초기화
    processor, tokenizer, feature_extractor, model, data_collator, metric = get_model_variable(model_name, device)
    print("Model is on device:", next(model.parameters()).device)
    print("<현재 description>\n", model_description)

    preprocessed_dataset = load_dataset(dataset_name, cache_dir=CACHE_DIR)  # Huggingface 데이터셋 로드

    # 데이터셋의 모든 샘플에 전처리 적용
    preprocessed_dataset = preprocessed_dataset.map(preprocess_batch)


    # 전처리된 데이터셋의 처음, 끝 5개 항목 transcripts 확인
    print("처음 5개 항목의 transcripts 데이터 확인:")
    for i in range(5):
        print(f"{i+1}번째 항목: {preprocessed_dataset['train'][i]['transcripts']}")

    print("\n마지막 5개 항목의 transcripts 데이터 확인:")
    for i in range(1, 6):
        print(f"{len(preprocessed_dataset['train'])-i+1}번째 항목: {preprocessed_dataset['train'][-i]['transcripts']}")




    if is_test: # 30%까지의 valid 데이터셋 선택(코드 작동 테스트를 위함)
        preprocessed_dataset["valid"] = preprocessed_dataset["valid"].select(range(math.ceil(len(preprocessed_dataset) * 0.3)))

    # 구글 드라이브의 mlflow.db 파일 받아오기(업데이트)
    # gdown.download('https://drive.google.com/uc?id=14v7CGtEI4PPOX7rsS6a4LrWCV-AovuPQ', '/mnt/a/maxseats/mlflow.db', quiet=False)

    experiment_id = set_mlflow_env(model_name)

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=preprocessed_dataset["train"],
        eval_dataset=preprocessed_dataset["valid"],  # or "test"
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    # MLflow 로깅 + train
    with mlflow.start_run(experiment_id=experiment_id, description=model_description):

        # training_args 로깅
        for key, value in training_args.to_dict().items():
            mlflow.log_param(key, value)

        mlflow.set_tag("Dataset", dataset_name) # 데이터셋 로깅

        trainer.train()

        # Metric 로깅
        metrics = trainer.evaluate()
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # MLflow 모델 레지스터
        model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=mlflow.active_run().info.run_id, artifact_path=model_dir)
        model_details = mlflow.register_model(model_uri=model_uri, name=model_name.replace('/', '-'))   # 모델 이름에 '/'를 '-'로 대체

        # 모델 Description 업데이트
        MlflowClient().update_model_version(name=model_details.name, version=model_details.version, description=model_description)

    upload_huggingface(token, repo_name, model_dir, tokenizer, dataset_name, model_name, model_description)

    # 폴더와 하위 내용 삭제
    shutil.rmtree(model_dir)
    shutil.rmtree('./repo')
    shutil.rmtree(CACHE_DIR)
    print('완료. 넘나 축하.')