import os

from datasets import Audio, Dataset, DatasetDict, load_dataset
from transformers import WhisperFeatureExtractor, WhisperTokenizer
from tqdm import tqdm
import pandas as pd

# DIR_PATH = os.path.dirname(__file__)
# low_call_voices_prepreocessed = load_dataset("maxseats/meeting_valid_preprocessed", cache_dir=DIR_PATH)
# print(type(low_call_voices_prepreocessed))
# print(low_call_voices_prepreocessed)
# print('-'*80)

############################################
# 오디오(.wav), label(.txt) 불러오기
# 경로 설정은 각자 다를테니 알아서 잘 설정 부탁드려요.
DIR_PATH = os.path.dirname(__file__)
DATA_DIR = os.path.join(DIR_PATH, "unzipped_files")

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base", cache_dir=DIR_PATH)
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base", language="Korean", task="transcribe", cache_dir=DIR_PATH)

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


def prepare_dataset(batch):
    # 오디오 파일을 16kHz로 로드
    audio = batch["audio"]

    # input audio array로부터 log-Mel spectrogram 변환
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

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

datasets = datasets.map(prepare_dataset, num_proc=None)
datasets = datasets.remove_columns(['audio', 'transcripts']) # 불필요한 부분 제거
print('-'*48)
print(type(datasets))
print(datasets)
print('-'*48)