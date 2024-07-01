import os
from pydub import AudioSegment
from tqdm import tqdm
from datasets import Audio, Dataset, DatasetDict, load_from_disk, concatenate_datasets
from transformers import WhisperFeatureExtractor, WhisperTokenizer
import pandas as pd
import shutil

# 사용자 지정 변수 설정
DATA_DIR = '/mnt/a/yeh-jeans/home/minseo/temp_downloader/dataset132/010.회의_음성_데이터/01.데이터/1.Training'  
output_dir = '/mnt/a/yeh-jeans/home/minseo/temp_downloader/dataset132/010.회의_음성_데이터/01.데이터/1.Training/dataset_132'
# @@@@ 허깅페이스데이터셋 이름 변경도 잊지말기                  
dataset_name = "choejiin/aihub-132-preprocessed-D22-0" 
token = "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"
CACHE_DIR = '/mnt/a/yeh-jeans/.cache'                
model_name = "SungBeom/whisper-small-ko"                        
CACHE_DIR = '/mnt/a/yeh-jeans/.cache'  
batch_size = 700   
error_log = 'error_files.log'
os.environ['HF_DATASETS_CACHE'] = CACHE_DIR

audio_base_dir = '/mnt/a/yeh-jeans/home/minseo/temp_downloader/dataset132/010.회의_음성_데이터/01.데이터/1.Training/원천데이터_0908_add/KconfSpeech_train_D22_mp3_0'
label_base_dir = '/mnt/a/yeh-jeans/home/minseo/temp_downloader/dataset132/010.회의_음성_데이터/01.데이터/1.Training/라벨링데이터_0908_add/KconfSpeech_train_D22_label_0'
# @@@@@@@@@@@@@@@ 위 2개 변수 중요 

# 캐시 디렉토리 생성
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Whisper 모델 준비
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name, cache_dir=CACHE_DIR)
tokenizer = WhisperTokenizer.from_pretrained(model_name, language="Korean", task="transcribe", cache_dir=CACHE_DIR)


def get_audio_text_pairs(audio_base_dir, label_base_dir):
    """
    주어진 오디오 및 라벨 디렉토리에서 MP3 파일과 일치하는 텍스트 파일 경로를 찾음.
    :param audio_base_dir: 원본 오디오 파일이 저장된 디렉토리
    :param label_base_dir: 라벨 파일이 저장된 디렉토리
    :return: 오디오 파일 경로와 일치하는 텍스트 파일 경로 리스트
    """
    audio_files = []
    transcript_files = []
    for root, _, files in os.walk(audio_base_dir):
        for file in files:
            if file.endswith(".mp3"):
                audio_path = os.path.join(root, file)
                
                relative_path = os.path.relpath(audio_path, audio_base_dir)
                relative_label_path = relative_path.replace('원천데이터_0908_add', '라벨링데이터_0908_add').replace('.mp3', '.txt')
                transcript_path = os.path.join(label_base_dir, relative_label_path)
                if os.path.exists(transcript_path):
                    audio_files.append(audio_path)
                    transcript_files.append(transcript_path)
    return audio_files, transcript_files


def convert_to_16khz(audio_path):
    """
    오디오 파일을 16kHz로 변환하는 함수

    Args:
    - audio_path (str): 오디오 파일의 경로

    Returns:
    - AudioSegment: 16kHz로 변환된 오디오
    """
    audio = AudioSegment.from_file(audio_path)
    return audio.set_frame_rate(16000)


def prepare_dataset(batch):
    """
    Whisper 모델을 위한 데이터셋 준비 함수. 오디오 데이터를 log-Mel spectrogram으로 변환하고
    텍스트 데이터를 라벨 ID로 변환함.
    :param batch: 입력 배치 데이터
    :return: 변환된 배치 데이터
    """
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["transcripts"]).input_ids
    return {"input_features": batch["input_features"], "labels": batch["labels"]}

def log_error(file_path, error):
    """
    오류 로그를 기록함.
    :param file_path: 오류가 발생한 파일 경로
    :param error: 오류 메시지
    """
    with open(error_log, 'a') as f:
        f.write(f"{file_path}: {error}\n")

def create_dataframe(output_dir, label_base_dir):
    """
    output_dir에 저장된 16kHz MP3 파일과 일치하는 TXT 파일을 찾아 데이터프레임으로 변환함.
    :param output_dir: 변환된 16kHz MP3 파일이 저장된 디렉토리
    :param label_base_dir: 라벨 TXT 파일이 저장된 디렉토리
    :return: 오디오 파일 경로와 일치하는 텍스트 파일을 포함한 데이터프레임
    """
    audio_files = []
    transcript_files = []
    
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.endswith(".mp3"):
                audio_path = os.path.join(root, file)
                relative_path = os.path.relpath(audio_path, output_dir)
                relative_label_path = relative_path.replace('.mp3', '.txt')
                transcript_path = os.path.join(label_base_dir, relative_label_path)
                if os.path.exists(transcript_path):
                    audio_files.append(audio_path)
                    transcript_files.append(transcript_path)
    
    data = []
    for audio_path, transcript_path in zip(audio_files, transcript_files):
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript = f.read().strip()
        data.append({"audio": audio_path, "transcripts": transcript})
    
    df = pd.DataFrame(data)
    return df

audio_files, transcript_files = get_audio_text_pairs(output_dir, label_base_dir)

print(transcript_files)
# 데이터를 데이터프레임으로 변환
#df = pd.DataFrame(data)
df = create_dataframe(output_dir, label_base_dir)


batches = []
for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
    batch_df = df.iloc[i:i+batch_size]
    if not batch_df.empty:
        ds = Dataset.from_dict(
            {"audio": [path for path in batch_df["audio"]],
             "transcripts": [transcript for transcript in batch_df["transcripts"]]}
        ).cast_column("audio", Audio(sampling_rate=16000))
    
        batch_datasets = DatasetDict({"batch": ds})
        batch_datasets = batch_datasets.map(prepare_dataset, num_proc=1)
        batch_datasets.remove_columns(['audio', 'transcripts'])
        batch_path = os.path.join(CACHE_DIR, f'batch_{i//batch_size}')
        batch_datasets.save_to_disk(batch_path)
        batches.append(batch_path)
        print(f"Processed and saved batch {i//batch_size}")
    else:
        print(f"Skipping empty batch {i//batch_size}")

if batches:
    # 저장된 배치 데이터셋 로드
    loaded_batches = [load_from_disk(path) for path in batches]
    full_dataset = concatenate_datasets([batch['batch'] for batch in loaded_batches])

    # 데이터셋을 train, test, validation으로 분할
    train_testvalid = full_dataset.train_test_split(test_size=0.2)
    test_valid = train_testvalid["test"].train_test_split(test_size=0.5)
    datasets = DatasetDict(
        {"train": train_testvalid["train"],
         "test": test_valid["test"],
         "valid": test_valid["train"]}
    )

    # 허깅페이스에 데이터셋 업로드
    while True:
        if token == "exit":
            break
        try:
            datasets.push_to_hub(dataset_name, token=token)
            print(f"Dataset {dataset_name} pushed to hub successfully.")
            break
        except Exception as e:
            print(f"Failed to push dataset: {e}")
            token = "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"

# 캐시 디렉토리 삭제
shutil.rmtree(CACHE_DIR)
print(f"Deleted cache directory: {CACHE_DIR}")
# git checkout -b dataset-make origin/yeh-jeans/dataset-make

# nohup python /mnt/a/yeh-jeans/home/minseo/temp_downloader/dataset132/010.회의_음성_데이터/01.데이터/1.Training/dataset_132/허깅페이스_데이터셋만들다멈췄을때.py > hgf_whenstop_2.log 2>&1 &
# 6/30 4:14 pid 837736 killed 4:24 ㅅㅂ
# ''   4:24 pid 851040 