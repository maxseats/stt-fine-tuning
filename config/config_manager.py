import argparse
import os
import yaml

import evaluate
from datasets import load_dataset
from transformers import (Seq2SeqTrainingArguments,
                          WhisperTokenizer,
                          WhisperProcessor,
                          WhisperFeatureExtractor,
                          WhisperForConditionalGeneration,
                          )

from src.data_collator import DataCollatorSpeechSeq2SeqWithPadding
from utils import find_git_repo


def load_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",type=str,default="config/config.yaml",help="Path to the config file",)
    parser.add_argument("--lr", "--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--batch_size", type=int, help="Batch size per device for training")
    parser.add_argument("--max_steps", type=int, help="Max steps for training")
    parser.add_argument("--test", action="store_true", help="Test mode flag")

    args = parser.parse_args()
    return args


def override_config(config, args):
    if args.lr is not None:
        config["training_args"]["learning_rate"] = args.lr
    if args.batch_size is not None:
        config["training_args"]["batch_size"] = args.batch_size
    if args.max_steps is not None:
        config["training_args"]["max_steps"] = args.max_steps
    if args.test:
        config["test"] = True
    if not args.test:
        config["test"] = False
    return config


def get_config():
    """
    yaml 파일과 argparse를 통해 받은 args를 합친 config 불러와서 반환하는 함수
    """
    repo_path = find_git_repo()
    output_dir = os.path.join(repo_path, ".tmp")

    args = parse_args()
    config = load_config(args.config)
    config = override_config(config, args)
    config["training_args"]["output_dir"] = output_dir

    config["model_description"] = """
직접 작성해주세요. 

파인튜닝한 데이터셋에 대해 최대한 자세히 설명해주세요.

(데이터셋 종류, 각 용량, 관련 링크 등)
"""

    return config


def get_components(config):
    """
    위에서 불러온 config를 통해
    model, dataset, trainig_arguments, ... 등 trainer 구성에 필요한 요소들을 반환하는 함수
    """
    model_name = config["model_name"]
    dataset_name = config["dataset_name"]

    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    preprocessed_dataset = load_dataset(dataset_name)
    processor = WhisperProcessor.from_pretrained(model_name, language="Korean", task="transcribe")
    tokenizer = WhisperTokenizer.from_pretrained(model_name, language="Korean", task="transcribe")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    metric = evaluate.load("cer")
    
    training_args = Seq2SeqTrainingArguments(**config["training_args"])

    return {
        "model": model,
        "preprocessed_dataset": preprocessed_dataset,
        "processor": processor,
        "tokenizer": tokenizer,
        "feature_extractor": feature_extractor,
        "data_collator": data_collator,
        "metric": metric,
        "training_args": training_args,
    }


if __name__ == "__main__":
    from pprint import pprint

    args = parse_args()
    config = load_config(args.config)
    config = override_config(config, args)

    print(type(config))
    pprint(config)
