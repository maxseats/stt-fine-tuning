#!/bin/sh

# 상위 폴더 지정
parent_folder=$(pwd)
# 출력 폴더 지정
output_folder=$parent_folder/unzipped_files

# 출력 폴더가 존재하지 않으면 생성
mkdir -p "$output_folder"

# 상위 폴더 및 하위 폴더 내의 모든 .zip 파일을 재귀적으로 처리
find "$parent_folder" -type f -name "*.zip" | while read zip_file; do
    # .zip 파일 이름에서 확장자 제거
    base_name=$(basename "$zip_file" .zip)
    
    # 임시 폴더 생성
    temp_folder=$(mktemp -d)

    # .zip 파일을 임시 폴더에 해제
    unzip -jn "$zip_file" -d "$temp_folder"

    # 해제된 파일들을 출력 폴더로 복사하면서 이름 변경
    for file in "$temp_folder"/*; do
        if [ -e "$file" ]; then
            new_file_name="${base_name}_$(basename "$file")"
            cp "$file" "$output_folder/$new_file_name"
        fi
    done

    # 임시 폴더 삭제
    rm -rf "$temp_folder"
done

echo $output_folder