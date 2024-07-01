import logging
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
import time

# 로그 설정
logging.basicConfig(filename='/mnt/a/yeh-jeans/upload_to_drive.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# GoogleAuth 객체 생성 및 인증
gauth = GoogleAuth()
gauth.LoadClientConfigFile('/mnt/a/yeh-jeans/client_secret_54898579230-7ljlu7nf7jm018tvrbvg7klhku44ripv.apps.googleusercontent.com.json')
gauth.LoadCredentialsFile("mycreds.txt")

if gauth.credentials is None:
    gauth.CommandLineAuth()
    gauth.SaveCredentialsFile("mycreds.txt")
elif gauth.access_token_expired:
    gauth.Refresh()
else:
    gauth.Authorize()

drive = GoogleDrive(gauth)

def upload_file_to_drive(file_path, folder_name):
    # Google Drive에 폴더 생성
    folder_metadata = {
        'title': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    folder = drive.CreateFile(folder_metadata)
    folder.Upload()
    folder_id = folder['id']
    logging.debug(f"Folder '{folder_name}' created with ID: {folder_id}")

    if not os.path.exists(file_path):
        logging.error(f"File '{file_path}' does not exist.")
        return

    # 파일을 Google Drive에 업로드
    logging.debug(f"Uploading file: {file_path}")
    file_metadata = {
        'title': os.path.basename(file_path),
        'parents': [{'id': folder_id}]
    }
    file_to_upload = drive.CreateFile(file_metadata)
    
    # 파일 업로드 시 재시도 로직 추가
    for attempt in range(3):  # 최대 3번 재시도
        try:
            file_to_upload.SetContentFile(file_path)
            file_to_upload.Upload()
            logging.debug(f"File '{os.path.basename(file_path)}' uploaded to folder '{folder_name}' with ID: {file_to_upload['id']}")
            break
        except Exception as e:
            logging.error(f"Failed to upload file on attempt {attempt + 1}: {e}")
            if attempt == 2:
                raise
            time.sleep(5)  # 5초 대기 후 재시도

# 업로드할 로컬 파일 경로와 Google Drive 폴더 이름 지정
file_to_upload = '/mnt/a/yeh-jeans/home/minseo/temp_downloader/dataset132/010.회의_음성_데이터/01.데이터/1.Training.zip'
drive_folder_name = f'aihub_dataset_132_{int(time.time())}'

# 파일을 Google Drive에 업로드
upload_file_to_drive(file_to_upload, drive_folder_name)


