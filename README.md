# STT Fine-tuning

## How to Run
### 0. 사전준비
`config/config.yaml`에서 사전 설정값을 원하는 것으로 바꿔주세요.

### 1. Train only
```
python train.py  # 일반적인 상황
python train.py --test # '--test' 옵션으로 전체의 10% 데이터로 evalution이 가능
```
> '--help'로 사용 가능한 Args 확인이 가능합니다. <br> `config.yaml`보다 우선적으로 적용됩니다.
### 2. Train with MLflow logging
```
python main.py
```
> 여기에도 `--test` 옵션을 사용할 수 있습니다.
### 2-1. MLflow UI
```
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:////mnt/a/mlflow.db # <- yaml 파일에 쓰여져있는 경로와 일치
```