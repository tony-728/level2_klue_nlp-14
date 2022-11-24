import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from transformers import ElectraModel, ElectraTokenizer
from sklearn.model_selection import KFold

from tqdm import tqdm
import wandb

from Model import Model
from Metric import compute_loss, compute_metrics
from load_data import RE_Dataset
import utils
from visualization import visualization_base

from typing import Dict, Optional
import warnings

warnings.filterwarnings(action="ignore")


def set_wandb(config: Dict, model, project: str, fold: int = 0):
    """
    wandb 관련 세팅

    Parameters
    ----------
    config : Dict
        모델학습과 관련된 config
    model : _type_
        학습에 사용될 모델
    project : str
        wandb project name
    fold: int
        fold 번호
    """
    wandb.login(key=config["wandb_key"])
    entity = "nlp02"

    if config["k-fold"]:
        wandb.init(
            reinit=True,
            entity=entity,
            project=project,
            name=f"(fold: {fold}, batch:{config['batch_size']},epoch:{config['epoch']},lr:{config['lr']})",
        )
    else:
        wandb.init(
            entity=entity,
            project=project,
            name=f"(batch:{config['batch_size']},epoch:{config['epoch']},lr:{config['lr']})",
        )
    wandb.watch(model, log_freq=100)
    return


def set_train(config: Dict):
    """
    모델 학습에 필요한 것들을 생성한다.
    K-fold로 검증할 때와 아닐 때를 구분한다.

    K-fold가 아닐 때
    - 모델
    - train dataloader
    - val dataloader
    - optimizer

    K-fold 일 때
    - K-fold 객체
    - total dataset

    Parameters
    ----------
    config : Dict
        모델학습과 관련된 config

    Returns
    -------
    Optional
        _description_
    """
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    train_dataset = RE_Dataset(config["train_data_path"], tokenizer)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )

    if config["k-fold"]:
        kfold_config = config["k-fold_config"]
        if kfold_config["shuffle"]:
            kf = KFold(
                n_splits=kfold_config["num_splits"],
                shuffle=True,
                random_state=kfold_config["split_seed"],
            )
        else:
            kf = KFold(n_splits=kfold_config["num_splits"], shuffle=False)

        return kf, train_dataset

    val_dataset = RE_Dataset(config["val_data_path"], tokenizer)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False
    )

    model = Model(config["model_name"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    return model, train_dataloader, val_dataloader, optimizer


def training(
    config: Dict,
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    fold: int = 0,
):
    """
    실제 학습을 진행한다.

    Parameters
    ----------
    config : Dict
        모델학습과 관련된 config
    model : _type_
        학습에 사용할 모델
    train_dataloader : _type_
        학습에 사용할 dataloader
    val_dataloader : _type_
        검증에 사용할 dataloader
    optimizer : _type_
        최적화 함수
    fold : int
        fold 번호

    Returns
    -------
    K-fold로 진행하지 않을 때
        f1이 가장 낮은 모델의 저장 경로
    K-fold로 진행할 때
        최종 validation loss
    """
    # wandb setting
    project = config["model_name"].replace("/", "-")

    if config["wandb"]:
        set_wandb(config, model, project, fold)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    highest_valid_f1 = 0.0

    model.to(device)

    accumulation_step = config["accumulation_step"]

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda epoch: 0.95**epoch,
        last_epoch=-1,
        verbose=False,
    )

    for epoch_num in range(config["epoch"]):
        # train
        model.train()
        epoch_loss = []
        running_loss = 0.0
        with tqdm(train_dataloader, unit="batch") as tepoch:
            for i, (item, labels, markers) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch_num}")

                batch = {k: v.to(device) for k, v in item.items()}
                markers = {k: v.to(device) for k, v in markers.items()}
                pred = model(batch, markers)
                loss = compute_loss(pred, labels.to(device))

                loss = loss / accumulation_step
                running_loss += loss.item()

                loss.backward()

                if accumulation_step > 1:
                    if (i + 1) % accumulation_step:
                        continue

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                epoch_loss.append(running_loss)
                tepoch.set_postfix(loss=running_loss)

                if config["wandb"]:
                    wandb.log({"train_learning_rate": scheduler.get_lr()[0]})
                    wandb.log({"train_loss": running_loss})

                running_loss = 0.0

            print(
                f"epoch: {epoch_num} train loss: {float(sum(epoch_loss) / len(epoch_loss)):.3f}"
            )
        # evaluation
        val_loss = []
        val_pred = []
        val_labels = []
        model.eval()
        with torch.no_grad():
            for i, (item, labels, markers) in enumerate(
                tqdm(val_dataloader, desc="Eval")
            ):
                batch = {k: v.to(device) for k, v in item.items()}
                pred = model(batch, markers)
                val_pred.append(pred)
                val_labels.append(labels)

                loss = compute_loss(pred, labels.to(device))
                val_loss.append(loss)

        val_pred = torch.cat(val_pred, dim=0).detach().cpu().numpy()
        val_labels = torch.cat(val_labels, dim=0).detach().cpu().numpy()

        metrics = compute_metrics(val_pred, val_labels)
        print(metrics)

        val_loss = float(sum(val_loss) / len(val_loss))
        print(f"epoch: {epoch_num} val loss: {val_loss:.3f}")

        # 시각화
        if not config["k-fold"]:
            visualization_base(
                config["val_data_path"],
                val_pred,
                val_labels,
                epoch_num,
                metrics,
                val_loss,
            )

        # wandb logging
        if config["wandb"]:
            wandb.log({"epoch": epoch_num})
            wandb.log({"eval_loss": val_loss})
            wandb.log({"eval_f1": metrics["micro f1 score"]})
            wandb.log({"eval_auprc": metrics["auprc"]})
            wandb.log({"eval_accuracy": metrics["accuracy"]})

        # model save
        if not config["k-fold"]:
            if metrics["micro f1 score"] > highest_valid_f1:
                save_model_dir = f"./best_model/{project}"
                if utils.create_directory(save_model_dir):
                    save_model_path = f"{save_model_dir}/{project}_b{config['batch_size']}_e{config['epoch']}_lr{config['lr']}.bin"
                    print(
                        "micro f1 score for model which have higher micro f1 score: ",
                        metrics["micro f1 score"],
                    )
                    torch.save(
                        model.state_dict(),
                        save_model_path,
                    )
                    highest_valid_f1 = metrics["micro f1 score"]

    if config["k-fold"]:
        # 마지막 validation loss, metrics 리턴
        return val_loss, metrics

    print("---- Train Finish! ----")
    return save_model_path


def train(config: Dict) -> Optional[str]:
    """
    입력된 config에 따라서 모델 학습을 진행한다.
    학습 동안 가장 낮은 loss를 기록한 모델을 저장한다.

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

    Returns
    -------
    Optional[str]
        저장된 모델 경로
    """
    if config["k-fold"]:
        kf, total_dataset = set_train(config)

        total_loss = []  # 각 폴드별로 나온 로스 넣어서 최종적으로 평균 떄려서 출력
        total_f1 = []  # 각 폴드별로 나온 f1 넣어서 최종적으로 평균 떄려서 출력
        total_auprc = []  # 각 폴드별로 나온 auprc 넣어서 최종적으로 평균 떄려서 출력
        total_accuracy = []  # 각 폴드별로 나온 accuracy 넣어서 최종적으로 평균 떄려서 출력

        for fold, (train_idx, val_idx) in enumerate(kf.split(total_dataset)):
            print("------------fold no.{}----------------------".format(fold))
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

            train_dataloader = torch.utils.data.DataLoader(
                total_dataset,
                batch_size=config["batch_size"],
                shuffle=False,
                sampler=train_subsampler,
            )

            val_dataloader = torch.utils.data.DataLoader(
                total_dataset,
                batch_size=config["batch_size"],
                shuffle=False,
                sampler=val_subsampler,
            )

            model = Model(config["model_name"])

            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

            val_loss, total_metrics = training(
                config, model, train_dataloader, val_dataloader, optimizer, fold
            )

            # k-fold로 검증하는 것은 각 fold의 loss, metrics를 구하고
            # 평균을 내서 최종적으로 loss와 metrics을 구한다.
            total_loss.append(val_loss)
            total_f1.append(total_metrics["micro f1 score"])
            total_auprc.append(total_metrics["auprc"])
            total_accuracy.append(total_metrics["accuracy"])

            ######################################

        print(f"K-fold mean loss: {sum(total_loss)/len(total_loss)}")
        print(f"K-fold mean f1 score: {sum(total_f1)/len(total_f1)}")
        print(f"K-fold mean auprc: {sum(total_auprc)/len(total_auprc)}")
        print(f"K-fold mean accuracy: {sum(total_accuracy)/len(total_accuracy)}")
        return None
    else:
        model, train_dataloader, val_dataloader, optimizer = set_train(config)

        ##training
        save_model_path = training(
            config, model, train_dataloader, val_dataloader, optimizer
        )

        return save_model_path


if __name__ == "__main__":
    import json

    config_file = "/opt/ml/level2_klue_nlp-14/config/klue-roberta-large-config.json"

    with open(config_file, "r") as f:
        config = json.load(f)

    train(config)
