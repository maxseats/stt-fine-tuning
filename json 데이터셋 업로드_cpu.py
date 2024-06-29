# !pip install -U accelerate
# !pip install -U transformers
# !pip install datasets
# !pip install evaluate
# !pip install mlflow
# !pip install transformers[torch]
# !pip install jiwer
# !pip install nlptutti
# !huggingface-cli login --token token

import os
import json
from pydub import AudioSegment
from tqdm import tqdm
import re
from datasets import Audio, Dataset, DatasetDict, load_from_disk, concatenate_datasets
from transformers import WhisperFeatureExtractor, WhisperTokenizer
import pandas as pd
import shutil

# 사용자 지정 변수를 설정해요.

set_num = 12                                                                       # 데이터셋 번호
token = "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"                                   # 허깅페이스 토큰
CACHE_DIR = '/mnt/a/maxseats/.cache_' + str(set_num)                              # 허깅페이스 캐시 저장소 지정
dataset_name = "maxseats/aihub-464-preprocessed-680GB-set-" + str(set_num)        # 허깅페이스에 올라갈 데이터셋 이름
model_name = "SungBeom/whisper-small-ko"                                          # 대상 모델 / "openai/whisper-base"
batch_size = 500                                                                 # 배치사이즈 지정, 8000이면 에러 발생

json_path = '/mnt/a/maxseats/mp3_dataset.json'                                    # 생성한 json 데이터셋 위치

print('현재 데이터셋 : ', 'set_', set_num)

def prepare_dataset(batch):
    # 오디오 파일을 16kHz로 로드
    audio = batch["audio"]

    # input audio array로부터 log-Mel spectrogram 변환
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # 'input_features'와 'labels'만 포함한 새로운 딕셔너리 생성
    return {"input_features": batch["input_features"], "labels": batch["labels"]}


# JSON 파일에서 데이터 로드
def load_dataset(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        return json.load(file)

# 파일 경로 참조해서 오디오, set_num 데이터 정답 라벨 불러오기
def getLabels(json_path, set_num):
    
    # JSON 파일 로드
    json_dataset = load_dataset(json_path)
    
    set_identifier = 'set_' + str(set_num) + '/'
    
    # "audio" 경로에 set_identifier가 포함된 데이터만 필터링
    filtered_data = [item for item in json_dataset if set_identifier in item['audio']]

    return pd.DataFrame(filtered_data)


# Sampling rate 16,000khz 전처리 + 라벨 전처리를 통해 데이터셋 생성
def df_transform(batch_size, prepare_dataset):
    # 오디오 파일 경로를 dict의 "audio" 키의 value로 넣고 이를 데이터셋으로 변환
    batches = []
    for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
        batch_df = df.iloc[i:i+batch_size]
        ds = Dataset.from_dict(
            {"audio": [path for path in batch_df["audio"]],
             "labels": [transcript for transcript in batch_df["transcripts"]]}
        ).cast_column("audio", Audio(sampling_rate=16000))

        batch_datasets = DatasetDict({"batch": ds})
        batch_datasets = batch_datasets.map(prepare_dataset, num_proc=1)
        batch_datasets.save_to_disk(os.path.join(CACHE_DIR, f'batch_{i//batch_size}'))
        batches.append(os.path.join(CACHE_DIR, f'batch_{i//batch_size}'))
        print(f"Processed and saved batch {i//batch_size}")

    # 모든 배치 데이터셋 로드, 병합
    loaded_batches = [load_from_disk(path) for path in batches]
    full_dataset = concatenate_datasets([batch['batch'] for batch in loaded_batches])

    return full_dataset

# 데이터셋을 훈련 데이터와 테스트 데이터, 밸리데이션 데이터로 분할
def make_dataset(full_dataset):
    train_testvalid = full_dataset.train_test_split(test_size=0.2)
    test_valid = train_testvalid["test"].train_test_split(test_size=0.5)
    datasets = DatasetDict(
        {"train": train_testvalid["train"],
         "test": test_valid["test"],
         "valid": test_valid["train"]}
    )
    return datasets

# 허깅페이스 로그인 후, 최종 데이터셋을 업로드
def upload_huggingface(dataset_name, datasets, token):
    
    while True:
        
        if token =="exit":
            break
        
        try:
            datasets.push_to_hub(dataset_name, token=token)
            print(f"Dataset {dataset_name} pushed to hub successfully. 넘나 축하.")
            break
        except Exception as e:
            print(f"Failed to push dataset: {e}")
            token = input("Please enter your Hugging Face API token: ")



for set_num in range(21, 69):  # 지정된 데이터셋 처리 후 업로드

    CACHE_DIR = '/mnt/a/maxseats/.cache_' + str(set_num)                              # 허깅페이스 캐시 저장소 지정
    dataset_name = "maxseats/aihub-464-preprocessed-680GB-set-" + str(set_num)        # 허깅페이스에 올라갈 데이터셋 이름
    print('현재 데이터셋 : ', 'set_', set_num)
    
    # 캐시 디렉토리 설정
    os.environ['HF_HOME'] = CACHE_DIR
    os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name, cache_dir=CACHE_DIR)
    tokenizer = WhisperTokenizer.from_pretrained(model_name, language="Korean", task="transcribe", cache_dir=CACHE_DIR)
    
    
    df = getLabels(json_path, set_num)
    print("len(df) : ", len(df))
    
    full_dataset = df_transform(batch_size, prepare_dataset)
    datasets = make_dataset(full_dataset)
    
    
    
    upload_huggingface(dataset_name, datasets, token)
    
    # 캐시 디렉토리 삭제
    shutil.rmtree(CACHE_DIR)
    print("len(df) : ", len(df))
    print(f"Deleted cache directory: {CACHE_DIR}")