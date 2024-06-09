from functools import partial
from typing import Dict

from transformers import Seq2SeqTrainer
from metrics import compute_metrics


def create_trainer(components: Dict) -> Seq2SeqTrainer:
    tokenizer = components["tokenizer"]
    metric = components["metric"]
    compute_metrics_fn = partial(compute_metrics, tokenizer=tokenizer, metric=metric)

    trainer = Seq2SeqTrainer(
        args=components["training_args"],
        model=components["model"],
        train_dataset=components["preprocessed_dataset"]["train"],
        eval_dataset=components["preprocessed_dataset"]["valid"],
        data_collator=components["data_collator"],
        compute_metrics=compute_metrics_fn,
        tokenizer=components["processor"].feature_extractor,
    )

    return trainer


if __name__ == "__main__":
    import math
    from transformers import Seq2SeqTrainer
    from config.config_manager import get_config, get_components

    config = get_config()  # yaml 파일과 argparse를 통해 받은 args를 합친 config 불러오기
    components = get_components(config)  # model, dataset, trainig_arguments, ... 등을 불러오기

    # (test용)valid dataset의 10%만 사용
    if config["test"]:
        valid_dataset = components["preprocessed_dataset"]["valid"]
        valid_dataset = valid_dataset.select(
            range(math.ceil(len(components["preprocessed_dataset"]) * 0.1))
        )
        components["preprocessed_dataset"]["valid"] = valid_dataset

    trainer = create_trainer(components)
    trainer.train()
    metrics = trainer.evaluate()

    print(metrics)
