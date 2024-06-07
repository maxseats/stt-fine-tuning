import math  # 임시 테스트용
import os
import shutil
import subprocess
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Union

import evaluate
import gdown
import mlflow
import torch
from datasets import load_dataset
from mlflow.tracking.client import MlflowClient
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          WhisperFeatureExtractor,
                          WhisperForConditionalGeneration, WhisperProcessor,            
                          WhisperTokenizer)

from config.config_manager import load_config, parse_args, override_config
from src.data_collator import DataCollatorSpeechSeq2SeqWithPadding
from utils import find_git_repo
from metrics import compute_metrics


repo_path = find_git_repo()
output_dir = os.path.join(repo_path, "tmp")  # 수정 X

args = parse_args()
config = load_config(args.config)
config = override_config(config, args)

model_description = """
직접 작성해주세요. 

파인튜닝한 데이터셋에 대해 최대한 자세히 설명해주세요.

(데이터셋 종류, 각 용량, 관련 링크 등)
"""
model_name = config["model_name"]
dataset_name = config["dataset_name"]  # 불러올 데이터셋(허깅페이스 기준)

is_test = config["test"]  # True: 소량의 샘플 데이터로 테스트, False: 실제 파인튜닝
training_args = config["training_args"]
training_args["output_dir"] = output_dir

training_args = Seq2SeqTrainingArguments(**training_args)

# 파인튜닝을 진행하고자 하는 모델의 processor, tokenizer, feature extractor, model 로드
processor = WhisperProcessor.from_pretrained(model_name, language="Korean", task="transcribe")
tokenizer = WhisperTokenizer.from_pretrained(model_name, language="Korean", task="transcribe")
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
metric = evaluate.load("cer")
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []


# Hub로부터 "16khz 전처리가 완료된" 데이터셋을 로드(이게 진짜 오래걸려요.)
preprocessed_dataset = load_dataset(dataset_name)

# 30%까지의 valid 데이터셋 선택(코드 작동 테스트를 위함)
if is_test:
    preprocessed_dataset["valid"] = preprocessed_dataset["valid"].select(range(math.ceil(len(preprocessed_dataset) * 0.3)))

compute_metrics = partial(compute_metrics, tokenizer=tokenizer, metric=metric)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=preprocessed_dataset["train"],
    eval_dataset=preprocessed_dataset["valid"],  # or "test"
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

trainer.train()

# Metric 로깅
metrics = trainer.evaluate()
print({"cer": metrics["eval_cer"]})