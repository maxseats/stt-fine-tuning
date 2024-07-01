# import os
# import shutil

# # 원본 데이터셋이 있는 폴더 경로
# source_folders = [
#     "/mnt/a/yeh-jeans/test_data/potcast_AI_13",
#     "/mnt/a/yeh-jeans/test_data/potcast_AI_19",
#     "/mnt/a/yeh-jeans/test_data/potcast_AI_20",
#     "/mnt/a/yeh-jeans/test_data/potcast_EP1"
# ]
# # 복사 대상 폴더
# target_folder = "/mnt/a/yeh-jeans/test_data_merged"


# # 만약 타겟 폴더가 없으면 생성
# if not os.path.exists(target_folder):
#     os.makedirs(target_folder)

# # 모든 폴더에서 파일을 가져와 타겟 폴더로 복사 및 이름 변경
# file_counter = 1

# for source_folder in source_folders:
#     mp3_files = [file for file in os.listdir(source_folder) if file.endswith(".mp3")]
    
#     for mp3_file in mp3_files:
#         # mp3 파일과 같은 이름의 txt 파일 찾기
#         txt_file = mp3_file[:-4] + ".txt"
        
#         if txt_file in os.listdir(source_folder):
#             mp3_file_path = os.path.join(source_folder, mp3_file)
#             txt_file_path = os.path.join(source_folder, txt_file)
            
#             # 새 파일 이름 설정
#             new_mp3_filename = f"testdatas_{file_counter:05d}.mp3"  # 파일 이름 형식 지정 (00001.mp3, 00002.mp3, ...)
#             new_txt_filename = f"testdatas_{file_counter:05d}.txt"  # 파일 이름 형식 지정 (00001.txt, 00002.txt, ...)
            
#             new_mp3_file_path = os.path.join(target_folder, new_mp3_filename)
#             new_txt_file_path = os.path.join(target_folder, new_txt_filename)
            
#             # 파일 복사 및 이름 변경
#             shutil.copy(mp3_file_path, new_mp3_file_path)
#             shutil.copy(txt_file_path, new_txt_file_path)
            
#             file_counter += 1


import os
import shutil

def merge_datasets(source_folder, target_folder):
    """
    Merge files from source_folder to target_folder with a new naming convention.
    
    Args:
    - source_folder (str): Path to the source folder containing files to be merged.
    - target_folder (str): Path to the target folder where files will be copied with a new naming convention.
    """
    # Ensure target_folder exists; create if it doesn't
    os.makedirs(target_folder, exist_ok=True)
    
    # List all files in the source_folder
    files = os.listdir(source_folder)
    
    # Find the number of existing testdatas in the target_folder
    existing_testdatas = [file for file in os.listdir(target_folder) if file.startswith("testdatas_")]
    if existing_testdatas:
        last_index = max([int(file.split("_")[1].split(".")[0]) for file in existing_testdatas])
    else:
        last_index = 0
    
    # Copy and rename each file from source_folder to target_folder
    for filename in files:
        if filename.endswith(".mp3"):
            new_index = last_index + 1
            new_filename_mp3 = f"testdatas_{new_index:05d}.mp3"
            new_filename_txt = f"testdatas_{new_index:05d}.txt"
            
            source_file_mp3 = os.path.join(source_folder, filename)
            source_file_txt = os.path.join(source_folder, filename.replace(".mp3", ".txt"))
            
            target_file_mp3 = os.path.join(target_folder, new_filename_mp3)
            target_file_txt = os.path.join(target_folder, new_filename_txt)
            
            # Copy mp3 file
            shutil.copyfile(source_file_mp3, target_file_mp3)
            print(f"Copied {filename} to {new_filename_mp3}")
            
            # Copy txt file
            shutil.copyfile(source_file_txt, target_file_txt)
            print(f"Copied {filename.replace('.mp3', '.txt')} to {new_filename_txt}")
            
            last_index += 1

# 예시 사용법:
source_folder = "/mnt/a/discord_dataset"
target_folder = "/mnt/a/yeh-jeans/test_data_merged"

merge_datasets(source_folder, target_folder)

print("테스트 데이터셋 합치기 및 이름 변경 완료")
# 
