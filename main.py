import math
import os
import mlflow
from datasets.config import HF_DATASETS_CACHE # 현재 미사용

from train import create_trainer
from config.config_manager import get_config, get_components
from utils import get_git_user_name

def main():
    config = get_config()  # yaml 파일과 argparse를 통해 받은 args를 합친 config 불러오기
    components = get_components(config)  # model, dataset, trainig_arguments, ... 등을 불러오기
    
    # (test용)valid dataset의 10%만 사용
    if config["test"]:
        valid_dataset = components["preprocessed_dataset"]["valid"]
        valid_dataset = valid_dataset.select(range(math.ceil(len(components["preprocessed_dataset"]) * 0.1)))
        components["preprocessed_dataset"]["valid"] = valid_dataset

    # (데이터셋의 경로 찾기) 데이터 불러오는 곳에 따라 다릅니다.
    # dataset_path = os.path.join(HF_DATASETS_CACHE, config["dataset_name"].replace('/', '___')) # 데이터셋을 Huggingface에서 불러오는 경우
    # dataset_path = config["dataset_name"] # 데이터셋을 로컬에서 불러오는 경우
    
    # MLflow 실험 이름을 모델 이름으로 설정
    experiment_name = config["model_name"]
    existing_experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if existing_experiment is not None:
        experiment_id = existing_experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(experiment_name)
    
    kwargs = {
        "experiment_id": experiment_id,
        "description": config["model_description"],
        "run_name": get_git_user_name()
    }
    
    # # 기존에 사용하던 코드 #
    # mlflow.set_experiment(experiment_name)

    # MLflow 로깅 시작
    with mlflow.start_run(**kwargs) as run:
        new_run_name = f"{run.info.run_name}-{run.info.run_id[:4]}" 
        mlflow.set_tag("mlflow.runName", new_run_name) # run name 재설정
        
        # mlflow.log_artifact(dataset_path, artifact_path="datasets")   # 데이터셋 경로 정해지면 사용
        mlflow.set_tag("Dataset", config["dataset_name"])  # 데이터셋 로깅
        
        for key, value in config["training_args"].items():
            mlflow.log_param(key, value)

        trainer = create_trainer(components)
        trainer.train()
        metrics = trainer.evaluate()

        # 결과를 MLflow에 로깅
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        
        model_uri = f"runs:/{run.info.run_id}/{config["training_args"]["output_dir"]}"
        model_name = config["model_name"].replace("/", "-")
        registered_model = mlflow.register_model(model_uri, model_name)
        
        print(f"Logged dataset metadata and model with CER: {metrics["eval_cer"]:0.4f}")
        print(f"Registered model: {registered_model.name}, version: {registered_model.version}")

    return metrics

if __name__ == "__main__":
    from pyprnt import prnt
    
    result = main()
    prnt(result)