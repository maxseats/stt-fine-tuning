import nlptutti as metrics
from transformers import pipeline
import json
import os
import time
import gdown
import gc
import torch
from transformers import WhisperTokenizer

'''
모든 모델은 모델명과 버전으로 구분해요.

ex) 
    model_name = "openai/whisper-base"
    model_version = 3
    
결과는 test_log_path에 저장되어요.

사용법 참고: https://pss2947.atlassian.net/issues/SCRUM-40?filter=10007
'''

#########################################################################################################################################
################################################### 사용자 설정 변수 #####################################################################
#########################################################################################################################################

model_names = ["maxseats/SungBeom-whisper-small-ko-set10"]  # 사용할 모델 이름
test_log_path = "/mnt/a/maxseats/STT_test"    # 테스트 결과 및 로그 저장위치
data_directory = "/mnt/a/yeh-jeans/test_data_merged"          # 데이터셋 폴더 지정
token = "hf_"
#########################################################################################################################################
################################################### 사용자 설정 변수 #####################################################################
#########################################################################################################################################

# 폴더가 없을 경우에만 실행
if not os.path.exists(data_directory):
    raise FileNotFoundError(f"데이터셋 폴더 '{data_directory}'를 찾을 수 없습니다.")

# 모델 별 테스트 파이프라인 실행
for model_name in model_names:

    tokenizer = WhisperTokenizer.from_pretrained(model_name, language="Korean", task="transcribe")  # 토크나이저 불러오기
    
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
        pipe = pipeline("automatic-speech-recognition", model=model_name, tokenizer=tokenizer, device=device)   # STT 파이프라인

        # 데이터셋 폴더 내 모든 파일에 대해 반복 처리
        for filename in sorted(os.listdir(data_directory)):
            if filename.endswith(".mp3"):
                mp3_file_path = os.path.join(data_directory, filename)
                txt_file_path = os.path.join(data_directory, filename.replace(".mp3", ".txt"))

                if not os.path.exists(txt_file_path):
                    print(f"{txt_file_path}을 찾을 수 없습니다.")
                    continue

                # 파일 번호 추출
                file_number = int(filename.split("_")[1].split(".")[0])

                print(f"{file_number}번째 데이터:")
                log_file.write(f"{file_number}번째 데이터:\n")

                try:
                    result = pipe(mp3_file_path, return_timestamps=False)
                except ValueError as e:
                    print(f"오류 발생: {e}")
                    print(f"{mp3_file_path} 파일을 건너뜁니다.")
                    log_file.write(f"오류 발생: {e}\n")
                    log_file.write(f"{mp3_file_path} 파일을 건너뜁니다.\n")
                    continue

                preds = result["text"]  # STT 예측 문자열

                # 파일 열기
                with open(txt_file_path, 'r', encoding='utf-8') as file:
                    target = file.read()  # 정답 텍스트 읽기

                print("예측 : ", preds)
                print("정답 : ", target)
                log_file.write(f"예측 : {preds}\n")
                log_file.write(f"정답 : {target}\n")

                # CER 계산 및 출력
                cer_result = metrics.get_cer(target, preds)
                CER_total += cer_result['cer']
                print(f"CER : {cer_result['cer']}, S : {cer_result['substitutions']}, D : {cer_result['deletions']}, I : {cer_result['insertions']}")
                log_file.write(f"CER, S, D, I : {cer_result['cer']}, {cer_result['substitutions']}, {cer_result['deletions']}, {cer_result['insertions']}\n")

                # WER 계산 및 출력
                wer_result = metrics.get_wer(target, preds)
                WER_total += wer_result['wer']
                print(f"WER : {wer_result['wer']}, S : {wer_result['substitutions']}, D : {wer_result['deletions']}, I : {wer_result['insertions']}\n")
                log_file.write(f"WER, S, D, I : {wer_result['wer']}, {wer_result['substitutions']}, {wer_result['deletions']}, {wer_result['insertions']}\n\n")

                # 로그 버퍼에서 파일로 flush(중간 저장)
                log_file.flush()
                os.fsync(log_file.fileno())

        # 파이프라인 사용 후 메모리 해제
        del pipe
        gc.collect()

    end_time = time.time()  # 종료 시간 기록
    elapsed_time = end_time - start_time    # 실행 시간

    # 시간, 분, 초 단위로 변환
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    print("현재 모델 : ", model_name)
    print("CER 평균 : ", CER_total / len(os.listdir(data_directory)))
    print("WER 평균 : ", WER_total / len(os.listdir(data_directory)))
    print("실행시간 : ", "{:02d}시간 {:02d}분 {:02d}초".format(hours, minutes, seconds))

    # 결과 데이터 저장
    data = {
        "model_name": model_name,
        "CER_mean": CER_total / len(os.listdir(data_directory)),
        "WER_mean": WER_total / len(os.listdir(data_directory)),
        "running_time" : "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)
    }

    # 기존 데이터 읽기(없으면 빈 리스트)
    try:
        with open(os.path.join(test_log_path, "total_result.json"), "r", encoding="utf-8") as file:
            data_list = json.load(file)
    except FileNotFoundError:
        data_list = []

    # 새 데이터 추가
    data_list.append(data)

    # CER_mean, WER_mean을 기준으로 오름차순 정렬
    sorted_data = sorted(data_list, key=lambda x: (x['CER_mean'], x['WER_mean']))

    # 정렬된 데이터를 파일로 저장
    with open(os.path.join(test_log_path, "total_result.json"), "w", encoding="utf-8") as file:
        json.dump(sorted_data, file, ensure_ascii=False, indent=4)

print("테스트 완료 및 결과 저장 완료")