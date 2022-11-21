import train
import inference
import json
from typing import Dict


def main(config: Dict):
    """
    config에 따라서 모델을 학습합니다.
    학습된 모델을 주어진 데이터 셋에 예측합니다.

    Parameters
    ----------
    config : Dict
        config Dictionary
        "wandb":
            wandb logging check: true/false,
        "wandb_key":
            "<YOUR wandb API KEY>",
        "inference":
            학습완료 후 inference 진행 check: true/false,
        "train_data_path": "<train data path>",
        "val_data_path": "<val data path>",
        "test_data_path": "<test data path>",
        "model_name": "<pre-trained model name>",
        "epoch": number of epoch,
        "batch_size": number of batch_size,
        "lr": learning rate,
        "k-fold":
            k-fold 적용 여부 check: true/false,
        "k-fold_config":{
            "num_splits": number of folds,
            "split_seed": random seed value,
            "shuffle":
                k-fold split suffle 여부 check: true/false,
        }
    """
    if config["k-fold"]:
        save_mode_path = train.trainKFold(config)
    else:
        save_mode_path = train.train(config)

    if config["inference"]:
        inference.main_inference(config, save_mode_path)


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)

    main(config)
