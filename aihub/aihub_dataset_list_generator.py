import os
import subprocess

modes = {
    "list" : "l",
    "download" : "d",
}

datasetkeys = [
    "71417",    # 극한 소음 음성인식 데이터
    "568",      # 소음 환경 음성인식 데이터
    "571",      # 저음질 전화망 음성인식 데이터
    "132",      # 회의 음성
    "71405",    # 명령어 인식을 위한 소음 환경 데이터
    "71627",    # 한국어 대학 강의 데이터
    "542",      # 다화자 음성합성 데이터
    "109",      # 자유대화 음성(일반남여)
    "71296",    # 생활환경소음 AI학습용 데이터 및 민원 관리 서비스 구축 사업
    "71557",    # 뉴스 대본 및 앵커 음성 데이터
    "537",      # 화자 인식용 음성 데이터
    "464",      # 주요 영역별 회의 음성인식 데이터
]


def run_command(mode: str, datasetkey: str) -> subprocess.CompletedProcess:
    commands = ["aihubshell", "-mode", mode, "-datasetkey", datasetkey]    
    return subprocess.run(commands, capture_output=True, text=True)


def save_list_to_file(strings: list[str], filename: str) -> None:
    directory = os.path.join(os.path.dirname(__file__), 'dataset_list')
    os.makedirs(directory, exist_ok=True)

    filename += '.txt'
    file_path = os.path.join(directory, filename)


    # 리스트의 원소를 개행으로 구분하여 하나의 문자열로 결합
    content = '\n'.join(strings)
    
    # 파일에 문자열을 저장
    with open(file_path, 'w') as file:
        file.write(content)

    
for datasetkey in datasetkeys:
    result = run_command(mode='l', datasetkey=datasetkey)
    title = f"{str(datasetkey)}_{str(result).split(r'\n')[10].split('.')[-1]}"
    dataset_list = str(result).split(r'\n')[10:-1]
    save_list_to_file(strings=dataset_list, filename=title)
