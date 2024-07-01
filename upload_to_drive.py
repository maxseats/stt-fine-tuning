from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os


gauth = GoogleAuth()
gauth.CommandLineAuth()
drive = GoogleDrive(gauth)

# 업로드할 로컬 폴더 경로
local_folder_path = '/home/minseo/temp_downloader/010.회의_음성_데이터.zip'

# Google Drive에 폴더 생성
folder_name = 'aihub_dataset_132'
folder_metadata = {
    'title': folder_name,
    'mimeType': 'application/vnd.google-apps.folder'
}
folder = drive.CreateFile(folder_metadata)
folder.Upload()

folder_id = folder['id']
print(f"Folder '{folder_name}' created with ID: {folder_id}")

# 로컬 폴더 내의 파일들을 Google Drive에 업로드
for filename in os.listdir(local_folder_path):
    file_path = os.path.join(local_folder_path, filename)
    if os.path.isfile(file_path):  # 파일인지 확인
        file_metadata = {
            'title': filename,
            'parents': [{'id': folder_id}]
        }
        file_to_upload = drive.CreateFile(file_metadata)
        file_to_upload.SetContentFile(file_path)
        file_to_upload.Upload()
        print(f"File '{filename}' uploaded to folder '{folder_name}' with ID: {file_to_upload['id']}")


# 업로드할 로컬 폴더 경로
local_folder_path = './dataset132/aihub_dataset132_mp3'

# Google Drive에 폴더 생성
folder_name = 'aihub_dataset132'
folder_metadata = {
    'title': folder_name,
    'mimeType': 'application/vnd.google-apps.folder'
}
folder = drive.CreateFile(folder_metadata)
folder.Upload()

folder_id = folder['id']
print(f"Folder '{folder_name}' created with ID: {folder_id}")

# 로컬 폴더 내의 파일들을 Google Drive에 업로드
for filename in os.listdir(local_folder_path):
    file_path = os.path.join(local_folder_path, filename)
    if os.path.isfile(file_path):  # 파일인지 확인
        file_metadata = {
            'title': filename,
            'parents': [{'id': folder_id}]
        }
        file_to_upload = drive.CreateFile(file_metadata)
        file_to_upload.SetContentFile(file_path)
        file_to_upload.Upload()
        print(f"File '{filename}' uploaded to folder '{folder_name}' with ID: {file_to_upload['id']}")
