import os
import shutil

from tqdm import tqdm

base_dir = os.getcwd()  # 기본 디렉터리 경로
target_dir = './renamed_data'

# 타겟 디렉터리 생성
os.makedirs(target_dir, exist_ok=True)

label_prefix = '[라벨]'
wav_prefix = '[원천]'

# 디렉터리 구분 및 리스트 생성 함수
def get_source_dirs(base_dir, label_prefix, wav_prefix):
    label_dirs = []
    wav_dirs = []
    
    for root, dirs, _ in os.walk(base_dir):
        for dir_name in dirs:
            full_path = os.path.join(root, dir_name)
            if label_prefix in dir_name:
                label_dirs.append(full_path)
            elif wav_prefix in dir_name:
                wav_dirs.append(full_path)
    
    return label_dirs, wav_dirs


# 파일 이동 및 이름 변경 함수
def move_and_rename_files(source_dirs, target_dir, file_extension):
    for source_dir in source_dirs:
        for root, _, files in os.walk(source_dir):
            for file in tqdm(files):
                if file.endswith(file_extension):
                    # 새로운 파일 이름 생성
                    relative_path = os.path.relpath(root, source_dir)
                    new_name = f"{relative_path.replace(os.sep, '_')}_{file}"
                    
                    # 파일 이동
                    source_path = os.path.join(root, file)
                    target_path = os.path.join(target_dir, new_name)
                    shutil.move(source_path, target_path)

# 디렉터리 구분 및 리스트 생성
label_dirs, wav_dirs = get_source_dirs(base_dir, label_prefix, wav_prefix)


# txt 파일 이동 및 이름 변경
move_and_rename_files(label_dirs, target_dir, '.txt')

# wav 파일 이동 및 이름 변경
move_and_rename_files(wav_dirs, target_dir, '.wav')

print("모든 파일이 성공적으로 이동되었습니다.")
print(f"이동된 파일 수: {len(os.listdir(target_dir))}")