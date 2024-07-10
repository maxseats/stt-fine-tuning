#!/bin/bash
# 디렉토리 안의 JSON 파일 개수를 세는 스크립트에요. 데이터 개수를 셀 때 사용해요.

# 기본 디렉토리 설정
base_dir="/mnt/a/maxseats/(주의-원본)주요 영역별 회의 음성인식 데이터/002.주요_영역별_회의_음성인식_데이터/01.데이터/unzipped_files"

# JSON 파일 개수 세기
json_count=$(find "$base_dir" -type f -name "*.json" | wc -l)

# 결과 출력
echo "Number of JSON files in $base_dir: $json_count"
