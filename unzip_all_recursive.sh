#!/bin/bash
# 압축 파일들을 해제하여 모든 파일을 unzipped_files 디렉토리에 모으는 스크립트에요.

# 현재 디렉토리 저장
base_dir="/mnt/a/maxseats/(주의-원본)주요 영역별 회의 음성인식 데이터/002.주요_영역별_회의_음성인식_데이터/01.데이터"

# 압축 해제한 파일들을 저장할 디렉토리
output_dir="$base_dir/unzipped_files"
mkdir -p "$output_dir"

# 모든 하위 디렉토리를 재귀적으로 탐색하여 .zip 파일들을 해제
find "$base_dir" -type f -name "*.zip" | while read zip_file; do
    unzip -o "$zip_file" -d "$output_dir"
done

echo "All files have been unzipped to $output_dir"
