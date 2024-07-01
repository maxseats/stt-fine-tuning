import os
from pydub import AudioSegment
from tqdm import tqdm
from datasets import Audio, Dataset, DatasetDict, load_from_disk, concatenate_datasets
from transformers import WhisperFeatureExtractor, WhisperTokenizer
import pandas as pd
import shutil
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# 사용자 지정 변수 설정 @@@@@2

output_dir = '/mnt/a/yeh-jeans/home/minseo/temp_downloader/dataset132/010.회의_음성_데이터/01.데이터/1.Training/dataset_132'
#token = "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"                     
token = "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"
CACHE_DIR = '/mnt/a/yeh-jeans/.cache'             
# @@@@ 허깅페이스데이터셋 이름 변경도 잊지말기                  
dataset_name = "choejiin/aihub-132-preprocessed-D21-1"              
model_name = "SungBeom/whisper-small-ko"                        

batch_size = 1000   
error_log = 'error_files.log'
os.environ['HF_DATASETS_CACHE'] = CACHE_DIR

audio_base_dir = '/mnt/a/yeh-jeans/home/minseo/temp_downloader/dataset132/010.회의_음성_데이터/01.데이터/1.Training/원천데이터_0908_add/KconfSpeech_train_D21_mp3_1'
label_base_dir = '/mnt/a/yeh-jeans/home/minseo/temp_downloader/dataset132/010.회의_음성_데이터/01.데이터/1.Training/라벨링데이터_0908_add/KconfSpeech_train_D21_label_1'
# @@@@@@@@@@@@@@@ 위 2개 변수 중요 

# 캐시 디렉토리 생성
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Slack Client 설정
client = WebClient(token=slack_token)

def send_slack_message(message):
    try:
        response = client.chat_postMessage(channel=slack_channel, text=message)
        assert response["message"]["text"] == message
    except SlackApiError as e:
        print(f"Error sending message: {e.response['error']}")

# Whisper 모델 준비
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name, cache_dir=CACHE_DIR)
tokenizer = WhisperTokenizer.from_pretrained(model_name, language="Korean", task="transcribe", cache_dir=CACHE_DIR)

def get_audio_text_pairs(audio_base_dir, label_base_dir):
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
    audio = AudioSegment.from_file(audio_path)
    return audio.set_frame_rate(16000)

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["transcripts"]).input_ids
    return {"input_features": batch["input_features"], "labels": batch["labels"]}

def log_error(file_path, error):
    with open(error_log, 'a') as f:
        f.write(f"{file_path}: {error}\n")

def create_dataframe(output_dir, label_base_dir):
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

audio_files, transcript_files = get_audio_text_pairs(audio_base_dir, label_base_dir)

# 오디오 파일을 16kHz로 변환하고, 변환된 파일을 output_dir에 저장
for audio_path, transcript_path in zip(audio_files, transcript_files):
    relative_audio_path = os.path.relpath(audio_path, audio_base_dir)
    output_audio_path = os.path.join(output_dir, relative_audio_path)
    output_audio_dir = os.path.dirname(output_audio_path)
    os.makedirs(output_audio_dir, exist_ok=True)
    if os.path.exists(output_audio_path):
        continue  # 이미 변환된 파일이 존재하면 건너뜀
    try:
        audio_16k = convert_to_16khz(audio_path)
        audio_16k.export(output_audio_path, format="mp3")
    except Exception as e:
        log_error(audio_path, str(e))
        continue

send_slack_message("Audio conversion to 16kHz completed.")

# 데이터를 데이터프레임으로 변환
df = create_dataframe(output_dir, label_base_dir)
send_slack_message("Dataframe creation completed.")

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
    loaded_batches = [load_from_disk(path) for path in batches]
    full_dataset = concatenate_datasets([batch['batch'] for batch in loaded_batches])
    train_testvalid = full_dataset.train_test_split(test_size=0.2)
    test_valid = train_testvalid["test"].train_test_split(test_size=0.5)
    datasets = DatasetDict(
        {"train": train_testvalid["train"],
         "test": test_valid["test"],
         "valid": test_valid["train"]}
    )
    while True:
        if token == "exit":
            break
        try:
            datasets.push_to_hub(dataset_name, token=token)
            send_slack_message(f"Dataset {dataset_name} pushed to hub successfully.")
            break
        except Exception as e:
            send_slack_message(f"Failed to push dataset: {e}")
            token = "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"

# 캐시 디렉토리 삭제
shutil.rmtree(CACHE_DIR)
print(f"Deleted cache directory: {CACHE_DIR}")
send_slack_message(f"Deleted cache directory: {CACHE_DIR}")
