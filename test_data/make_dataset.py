import os
import string
from pydub import AudioSegment
import pysrt

def process_audio_and_subtitles(audio_file, subs_file, output_dir):
    # 자막 파일 열기
    subs = pysrt.open(subs_file)

    # 오디오 파일 로드
    audio = AudioSegment.from_mp3(audio_file)

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 줄바꿈 및 문장 부호 제거용 변환 테이블 생성
    translator = str.maketrans('', '', string.punctuation + '\n\r')

    # 각 자막에 따라 오디오 분할 및 자막 저장
    for sub in subs:
        start_ms = sub.start.ordinal  # 시작 시간 (밀리초)
        end_ms = sub.end.ordinal  # 종료 시간 (밀리초)
        
        # 오디오 분할
        segment = audio[start_ms:end_ms]
        segment_filename = f"segment_{str(sub.index).zfill(3)}"
        segment_path = os.path.join(output_dir, f"{segment_filename}.mp3")
        segment.export(segment_path, format="mp3")  # 분할된 오디오를 MP3로 저장
        
        # 자막 텍스트 파일 저장
        text_path = os.path.join(output_dir, f"{segment_filename}.txt")
        # 줄바꿈 및 문장 부호 제거
        new_content = sub.text.translate(translator)
        
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

    print("오디오 분할 및 자막 저장이 완료되었습니다.")

# 사용 예제
audio_file = "/Users/yejinchoe/Documents/futomo/output_it/[Dataplorer](자막有) 팟캐스트 EP.1 - 데이터로 돈을 벌었던 사례 (1).mp3"
subs_file = "/Users/yejinchoe/Documents/futomo/output_it/[Korean] [Dataplorer](자막有) 팟캐스트 EP.1 - 데이터로 돈을 벌었던 사례 [DownSub.com] (2).srt"
output_dir = "/Users/yejinchoe/Documents/futomo/output_it/potcast_EP1"

process_audio_and_subtitles(audio_file, subs_file, output_dir)
