''' Author: @maxseats
모든 모델은 모델명과 버전으로 구분해요.

ex) 
    model_name = "openai/whisper-base"
    model_version = 3
    
결과는 test_log_path에 저장되어요.

사용법 참고: https://pss2947.atlassian.net/issues/SCRUM-40?filter=10007
'''

''' Author: @kooqooo
@maxseats의 whisper_test.ipynb 파일을 참고하여 작성했습니다.

원본 파일과 설명은 아래의 링크에서 받을 수 있습니다.
https://koogmlcks.notion.site/32f5ab9b703047d3a6002770ddf65e15?pvs=4

# 모델 테스트 해보기
- 데이터셋 불러와서 평가지표(CER, WER)로 모델 점수 출력
- 파일로 저장
    - total.json : 모델 전체 평가
    - 각 모델 폴더 : 해당 모델 평가 과정 로그 기록
'''
import json
import os
import time
import gc

import gdown
import torch 
import nlptutti as metrics
from transformers import pipeline

### 데이터 셋 준비 ###
file_path = os.path.dirname(__file__)
data_path = os.path.join(file_path, 'discord_dataset')

is_dataset_exists = os.path.exists(data_path)
if not is_dataset_exists: # 파일이 없는 경우 다운로드와 압축해제
    try:
        ID = "gdrive_file_id" # 여기에 실제 파일 ID를 입력
        gdown.download(id=ID, output="dataset.zip", quiet=False)

        # unzip
        os.system("unzip dataset.zip")
        os.system("rm dataset.zip")
    except:
        print("구글 드라이브 file id가 잘못되었습니다.")


### 사용자 설정 변수 ###
model_names = ["SungBeom/whisper-small-ko"]
data_num = 75   # 데이터 개수
test_log_path = os.path.join(file_path, "test_log")    # 테스트 결과 및 로그 저장위치
data_directory = "discord_dataset"


### 테스트 & 결과 기록
# 모델 별 테스트 파이프라인 실행
for model_name in model_names:

    start_time = time.time()    # 시작 시간 기록

    # 평균 계산용
    CER_total = 0.0
    WER_total = 0.0

    # 모델 폴더 생성 및 로그파일 폴더 지정
    model_log_dir = os.path.join(test_log_path, model_name)
    os.makedirs(model_log_dir, exist_ok=True)
    log_file_path = os.path.join(model_log_dir, "log.txt")

    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        # GPU 사용을 위해 device=0 설정
        device = 0 if torch.cuda.is_available() else -1
        device = "mps" # macOS 사용시

        pipe = pipeline("automatic-speech-recognition", model=model_name, device=device)   # STT 파이프라인

        for i in range(1, data_num+1):
            print(i, "번째 데이터:")
            log_file.write(f"{i} 번째 데이터:\n")

            sample = data_directory + "/" + "{:03d}".format(i) + ".mp3"    # 음성파일 경로

            result = pipe(sample, return_timestamps=False)

            preds = result["text"]  # STT 예측 문자열
            target_path = data_directory + "/" + "{:03d}".format(i) + ".txt" # 텍스트파일 경로


            # 파일 열기
            with open(target_path, 'r', encoding='utf-8') as file:
                # 파일 내용 읽기
                target = file.read()

            print("예측 : ", result["text"])
            print("정답 : ", target)
            log_file.write(f"예측 : {preds}\n")
            log_file.write(f"정답 : {target}\n")

            # CER 출력
            cer_result = metrics.get_cer(target, preds)

            cer_substitutions = cer_result['substitutions']
            cer_deletions = cer_result['deletions']
            cer_insertions = cer_result['insertions']
            # prints: [cer, substitutions, deletions, insertions] -> [CER = 0 / 34, S = 0, D = 0, I = 0]
            CER_total += cer_result['cer']
            print("CER, S, D, I : ", cer_result['cer'], cer_substitutions, cer_deletions, cer_insertions)
            log_file.write(f"CER, S, D, I : {cer_result['cer']}, {cer_substitutions}, {cer_deletions}, {cer_insertions}\n")


            # WER 출력
            wer_result = metrics.get_wer(target, preds)

            wer_substitutions = wer_result['substitutions']
            wer_deletions = wer_result['deletions']
            wer_insertions = wer_result['insertions']
            # prints: [wer, substitutions, deletions, insertions] -> [WER =  2 / 4, S = 1, D = 1, I = 0]
            WER_total += wer_result['wer']
            print("WER, S, D, I : ", wer_result['wer'], wer_substitutions, wer_deletions, wer_insertions)
            print()
            log_file.write(f"WER, S, D, I : {wer_result['wer']}, {wer_substitutions}, {wer_deletions}, {wer_insertions}\n\n")

            # 로그 버퍼에서 파일로 flush(중간 저장)
            log_file.flush()
            os.fsync(log_file.fileno())

    end_time = time.time()  # 종료 시간 기록
    elapsed_time = end_time - start_time    # 실행 시간

    # 시간, 분, 초 단위로 변환
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)


    print("현재 모델 : ", model_name)
    print("CER 평균 : ", CER_total / data_num)
    print("WER 평균 : ", WER_total / data_num)
    print("실행시간 : ", "{:02d}시간 {:02d}분 {:02d}초".format(hours, minutes, seconds))

    # 데이터 딕셔너리 생성
    data = {
        "model_name": model_name,
        "CER_mean": CER_total / data_num,
        "WER_mean": WER_total / data_num,
        "running_time" : "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)
    }


    # 기존 데이터 읽기(없으면 빈 리스트)
    try:
        with open(test_log_path + "/total_result.json", "r", encoding="utf-8") as file:
            data_list = json.load(file)
    except FileNotFoundError:
        data_list = []

    # 새 데이터 추가
    data_list.append(data)

    # CER_mean, WER_mean을 기준으로 오름차순 정렬
    sorted_data = sorted(data_list, key=lambda x: (x['CER_mean'], x['WER_mean']))

    # 정렬된 데이터를 파일로 저장
    with open(test_log_path + "/total_result.json", "w", encoding="utf-8") as file:
        json.dump(sorted_data, file, ensure_ascii=False, indent=4)

    # 파이프라인 사용 후 메모리 해제
    del pipe
    gc.collect()