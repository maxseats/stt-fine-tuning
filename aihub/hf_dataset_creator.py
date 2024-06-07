# 원본 데이터 -> 학습가능한 허깅페이스 데이터셋 형태로 변환

import os
import subprocess
import re

from datasets import Audio, Dataset, DatasetDict, config
from transformers import WhisperFeatureExtractor, WhisperTokenizer
from tqdm import tqdm
import pandas as pd



############################################
# 오디오(.wav), label(.txt) 불러오기
# 경로 설정은 각자 다를테니 알아서 잘 설정 부탁드려요.

# aihubshell -mode l -> 데이터셋 별 datasetkey 확인

# 사전 작업 1
#   tmux new -s maxseats_aihub
#   cd /mnt/a/maxseats/aihub 
#   aihubshell -mode d -datasetkey 71481 -aihubid oooooo123456@naver.com -aihubpw neuroflow37!
# 명령어 입력을 통한 AI hub 데이터셋 다운로드 ( 71481 : 전문분야 심층인터뷰 데이터 )

# 사전 작업 2
#   ./unzip_all_recursive.sh
# AI hub 데이터셋 압축 해제 후 하나의 폴더에 모든 파일을 담아주는 명령어

DIR_PATH = os.path.dirname(__file__)
DATA_DIR = 'maxseats-git/maxseats-ignore/aihub/Test_sample' # os.path.join(DIR_PATH, "Test") # 압축 해제된 파일들 있는 폴더
CACHE_DIR = '/mnt/a'    # 허깅페이스 캐시 저장소 지정 / 테스트 :  os.path.join(DIR_PATH, "cache_test")



dataset_name = "maxseats/bracket-test-dataset-tmp" # 허깅페이스에 올라갈 데이터셋 이름
model_name = "SungBeom/whisper-small-ko" #"openai/whisper-base"


# 캐시 디렉토리 설정
os.environ['HF_HOME'] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name, cache_dir=CACHE_DIR)
tokenizer = WhisperTokenizer.from_pretrained(model_name, language="Korean", task="transcribe", cache_dir=CACHE_DIR)

def exclude_json_files(file_names: list) -> list:
    # .json으로 끝나는 원소 제거
    return [file_name for file_name in file_names if not file_name.endswith('.json')]


def get_label_list(directory):
    # 빈 리스트 생성
    label_files = []

    # 디렉토리 내 파일 목록 불러오기
    for filename in os.listdir(directory):
        # 파일 이름에 'label'이 포함되어 있고 '.txt'로 끝나는지 확인
        if 'label' in filename and filename.endswith('.txt'):
            label_files.append(os.path.join(DATA_DIR, filename))

    return label_files


def get_wav_list(directory):
    # 빈 리스트 생성
    wav_files = []

    # 디렉토리 내 파일 목록 불러오기
    for filename in os.listdir(directory):
        # 파일 이름에 'label'이 포함되어 있고 '.txt'로 끝나는지 확인
        if 'wav' in filename and filename.endswith('.wav'):
            wav_files.append(os.path.join(DATA_DIR, filename))

    return wav_files

def bracket_preprocess(text):
    # 괄호 전처리
    
    # 1단계: o/ n/ 글자/ 과 같이. 앞 뒤에 ) ( 가 오지않는 /슬래쉬 는 모두 제거합니다. o,n 이 붙은 경우 해당 글자도 함께 제거합니다.
    text = re.sub(r'\b[o|n]/', '', text)
    text = re.sub(r'[^()]/', '', text)
    
    # 2단계: (70)/(칠십) 과 같은 경우, /슬래쉬 의 앞쪽 괄호의 내용만 남기고 삭제합니다.
    text = re.sub(r'\(([^)]*)\)/\([^)]*\)', r'\1', text)
    
    return text

def prepare_dataset(batch):
    # 오디오 파일을 16kHz로 로드
    audio = batch["audio"]

    # input audio array로부터 log-Mel spectrogram 변환
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # 괄호 전처리
    batch["transcripts"] = bracket_preprocess(batch["transcripts"])

    # target text를 label ids로 변환
    batch["labels"] = tokenizer(batch["transcripts"]).input_ids
    
    # 'input_features'와 'labels'만 포함한 새로운 딕셔너리 생성
    return {"input_features": batch["input_features"], "labels": batch["labels"]}


label_data = get_label_list(DATA_DIR)
wav_data = get_wav_list(DATA_DIR)

transcript_list = []
for label in tqdm(label_data):
    with open(label, 'r', encoding='UTF8') as f:
        line = f.readline()
        transcript_list.append(line)

df = pd.DataFrame(data=transcript_list, columns = ["transcript"]) # 정답 label
df['wav_data'] = wav_data # 오디오 파일 경로

# 오디오 파일 경로를 dict의 "audio" 키의 value로 넣고 이를 데이터셋으로 변환
# 이때, Whisper가 요구하는 사양대로 Sampling rate는 16,000으로 설정한다.
ds = Dataset.from_dict(
    {"audio": [path for path in df["wav_data"]],
     "transcripts": [transcript for transcript in df["transcript"]]}
).cast_column("audio", Audio(sampling_rate=16000))

# 데이터셋을 훈련 데이터와 테스트 데이터, 밸리데이션 데이터로 분할
train_testvalid = ds.train_test_split(test_size=0.2)
test_valid = train_testvalid["test"].train_test_split(test_size=0.5)
datasets = DatasetDict(
    {"train": train_testvalid["train"],
     "test": test_valid["test"],
     "valid": test_valid["train"]}
)

datasets = datasets.map(prepare_dataset, num_proc=None, batch_size=10000)
datasets = datasets.remove_columns(['audio', 'transcripts']) # 불필요한 부분 제거
print('-'*48)
print(type(datasets))
print(datasets)
print('-'*48)

# 허깅페이스 토큰
token = "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"
subprocess.run(["huggingface-cli", "login", "--token", token])

# 전처리 완료된 데이터셋을 Hubdataset_name에 저장
datasets.push_to_hub(dataset_name)