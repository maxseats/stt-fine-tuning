import os
import json
from pydub import AudioSegment
from tqdm import tqdm
import re
from datasets import Audio, Dataset, DatasetDict, load_from_disk, concatenate_datasets
from transformers import WhisperFeatureExtractor, WhisperTokenizer
import pandas as pd
import shutil

# mp3로 변환해서 저장하기 까지만 진행하는 코드에요.


# 사용자 지정 변수를 설정해요.

# DATA_DIR = '/mnt/a/maxseats/(주의-원본-680GB)주요 영역별 회의 음성인식 데이터' # 데이터셋이 저장된 폴더
DATA_DIR = '/mnt/a/maxseats/(주의-원본)split_files/set_3'  # 첫 10GB 테스트

# 원천, 라벨링 데이터 폴더 지정
json_base_dir = DATA_DIR
audio_base_dir = DATA_DIR
output_dir = '/mnt/a/maxseats/(주의-원본)clips/set_3'                     # 가공된 데이터셋이 저장될 폴더

'''
데이터셋 경로를 지정해서
하나의 폴더에 mp3, txt 파일로 추출해요. (clips_set_i 폴더)
추출 과정에서 원본 파일은 자동으로 삭제돼요. (저장공간 절약을 위해)
'''

def bracket_preprocess(text):
    
    # 정규 표현식을 사용하여 패턴 제거
    text = re.sub(r'/\([^\)]+\)', '', text)  # /( *) 패턴 제거, /(...) 형식 제거
    text = re.sub(r'[()]', '', text)         # 개별적으로 등장하는 ( 및 ) 제거
    
    return text.strip()

def process_audio_and_subtitle(json_path, audio_base_dir, output_dir):
    # JSON 파일 읽기
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 메타데이터에서 오디오 파일 이름 추출
    title = data['metadata']['title']
    
    # 각 TS, VS 폴더에서 해당 오디오 파일을 찾기
    audio_file = None
    for root, _, files in os.walk(audio_base_dir):
        for file in files:
            if file == title + '.wav':
                audio_file = os.path.join(root, file)
                break
        if audio_file:
            break
    
    # 오디오 파일 로드
    if not audio_file or not os.path.exists(audio_file):
        print(f"Audio file {audio_file} does not exist.")
        return
    
    audio = AudioSegment.from_wav(audio_file)
    audio_length_ms = len(audio)
    
    # 발화 데이터 처리
    for utterance in data['utterance']:
        start_time = int(float(utterance['start']) * 1000.0)# 밀리초로 변환
        end_time = int(float(utterance['end']) * 1000.0)    # 밀리초로 변환
        text = bracket_preprocess(utterance['form'])   # 괄호 전처리
        
        if not text:    # 비어 있으면 수행 x
            continue
        
        # 비정상적인 start_time 및 end_time 감지
        if start_time < 0 or end_time > audio_length_ms or start_time >= end_time:
            continue
        
        
        # 오디오 클립 추출
        audio_clip = audio[start_time:end_time]
        
        # 파일 이름 설정
        clip_id = utterance['id']
        audio_output_path = os.path.join(output_dir, clip_id + '.mp3')
        text_output_path = os.path.join(output_dir, clip_id + '.txt')
        
        # 오디오 클립 저장
        audio_clip.export(audio_output_path, format='mp3')
        
        # 괄호 전처리 텍스트 파일 저장
        with open(text_output_path, 'w', encoding='utf-8') as f:
            f.write(text)

    # 오디오 파일 삭제
    os.remove(audio_file)
    os.remove(audio_file.replace('.wav', '.txt'))
    print(f"Deleted audio file: {audio_file}")

def process_all_files(json_base_dir, audio_base_dir, output_dir):
    json_files = []
    
    # JSON 파일 목록 생성
    for root, dirs, files in os.walk(json_base_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    # JSON 파일 처리
    for json_file in tqdm(json_files, desc="Processing JSON files"):
        process_audio_and_subtitle(json_file, audio_base_dir, output_dir)
        
        # 완료 후 JSON 파일 삭제
        os.remove(json_file)
        print(f"Deleted JSON file: {json_file}")

start_set = 3
end_set = 68

for i in range(start_set, end_set + 1):
    DATA_DIR = f'/mnt/a/maxseats/(주의-원본)split_files/set_{i}'
    output_dir = f'/mnt/a/maxseats/(주의-원본)mp3_clips/set_{i}'
    
    # 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 프로세스 실행
    process_all_files(DATA_DIR, DATA_DIR, output_dir)


'''
가공된 mp3, txt 데이터를 학습 가능한 허깅페이스 데이터셋 형태로 변환해요.
'''

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
        # 파일 이름이 '.txt'로 끝나는지 확인
        if filename.endswith('.txt'):
            label_files.append(os.path.join(directory, filename))

    return label_files


def get_audio_list(directory):
    # 빈 리스트 생성
    audio_files = []

    # 디렉토리 내 파일 목록 불러오기
    for filename in os.listdir(directory):
        # 파일 이름이 '.wav'나 '.mp3'로 끝나는지 확인
        if filename.endswith('.wav') or filename.endswith('mp3'):
            audio_files.append(os.path.join(directory, filename))

    return audio_files

def prepare_dataset(batch):
    # 오디오 파일을 16kHz로 로드
    audio = batch["audio"]

    # input audio array로부터 log-Mel spectrogram 변환
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]


    # target text를 label ids로 변환
    batch["labels"] = tokenizer(batch["transcripts"]).input_ids
    
    # 'audio'와 'transcripts' 컬럼 제거
    # del batch["audio"]
    # del batch["transcripts"]
    
    # 'input_features'와 'labels'만 포함한 새로운 딕셔너리 생성
    return {"input_features": batch["input_features"], "labels": batch["labels"]}


label_data = get_label_list(output_dir)
audio_data = get_audio_list(output_dir)

transcript_list = []
for label in tqdm(label_data):
    with open(label, 'r', encoding='UTF8') as f:
        line = f.readline()
        transcript_list.append(line)

df = pd.DataFrame(data=transcript_list, columns = ["transcript"]) # 정답 label
df['audio_data'] = audio_data # 오디오 파일 경로

# 오디오 파일 경로를 dict의 "audio" 키의 value로 넣고 이를 데이터셋으로 변환
# 이때, Whisper가 요구하는 사양대로 Sampling rate는 16,000으로 설정한다.
# 데이터셋 배치 처리
batches = []
print("len(df) : ", len(df))
for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
    batch_df = df.iloc[i:i+batch_size]
    ds = Dataset.from_dict(
        {"audio": [path for path in batch_df["audio_data"]],
         "transcripts": [transcript for transcript in batch_df["transcript"]]}
    ).cast_column("audio", Audio(sampling_rate=16000))
    
    batch_datasets = DatasetDict({"batch": ds})
    batch_datasets = batch_datasets.map(prepare_dataset, num_proc=1)
    batch_datasets.save_to_disk(os.path.join(CACHE_DIR, f'batch_{i//batch_size}'))
    batches.append(os.path.join(CACHE_DIR, f'batch_{i//batch_size}'))
    print(f"Processed and saved batch {i//batch_size}")

# 모든 배치 데이터셋 로드
loaded_batches = [load_from_disk(path) for path in batches]

# 배치 데이터셋을 하나로 병합
full_dataset = concatenate_datasets([batch['batch'] for batch in loaded_batches])

# 데이터셋을 훈련 데이터와 테스트 데이터, 밸리데이션 데이터로 분할
train_testvalid = full_dataset.train_test_split(test_size=0.2)
test_valid = train_testvalid["test"].train_test_split(test_size=0.5)
datasets = DatasetDict(
    {"train": train_testvalid["train"],
     "test": test_valid["test"],
     "valid": test_valid["train"]}
)

# 열 제거 전 데이터셋 크기 확인
print(f"Dataset sizes before column removal: Train: {len(datasets['train'])}, Test: {len(datasets['test'])}, Valid: {len(datasets['valid'])}")

datasets = datasets.remove_columns(['audio', 'transcripts'])  # 불필요한 부분 제거

# 열 제거 후 데이터셋 크기 확인
print(f"Dataset sizes after column removal: Train: {len(datasets['train'])}, Test: {len(datasets['test'])}, Valid: {len(datasets['valid'])}")

#datasets = datasets.remove_columns(['audio', 'transcripts']) # 불필요한 부분 제거


'''
허깅페이스 로그인 후, 최종 데이터셋을 업로드해요.
'''

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

# 캐시 디렉토리 삭제
shutil.rmtree(CACHE_DIR)
print(f"Deleted cache directory: {CACHE_DIR}")