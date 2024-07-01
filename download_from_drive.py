from google.oauth2 import service_account
from googleapiclient.discovery import build
import io
from googleapiclient.http import MediaIoBaseDownload
# python download_from_drive.py

# 인증 및 Drive API 서비스 생성
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
SERVICE_ACCOUNT_FILE = '/mnt/a/yeh-jeans/intrepid-nova-370814-40b5bc7c6667.json'  # 업로드한 credentials.json 파일 경로

credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)

service = build('drive', 'v3', credentials=credentials)

# 파일 ID로 파일 다운로드
file_id = '1SLA2v-1Od1D7Uaa5xM4ikBkMqV9LK7s1'
request = service.files().get_media(fileId=file_id)
file_name = '/mnt/a/yeh-jeans/010_data.zip'
fh = io.FileIO(file_name, 'wb')
downloader = MediaIoBaseDownload(fh, request)

done = False
while not done:
    status, done = downloader.next_chunk()
    print(f"Download {int(status.progress() * 100)}%.")

print(f"File downloaded to {file_name}")
