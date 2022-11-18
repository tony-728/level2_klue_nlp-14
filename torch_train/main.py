import torch

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    RobertaConfig,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    BertTokenizer,
)

from Metric import *
from Dataset import *
from Model import Model

import warnings

warnings.filterwarnings(action="ignore")


config = {
    "train_data_path": "/opt/ml/dataset/train/sample_train.csv",
    "val_data_path": "",
    "model_name": "klue/bert-base",
    "epoch": 10,
    "batch_size": 32,
    "lr": 1e-5,
}
tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

train_dataset = TrainDataset(config["train_data_path"], tokenizer)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=config["batch_size"], shuffle=False
)

model_config = AutoConfig.from_pretrained(config["model_name"])
model_config.num_labels = 30
model = AutoModelForSequenceClassification.from_pretrained(
    config["model_name"], config=model_config
)

optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

##training
model.to(device)

for epoch_num in range(config["epoch"]):
    model.train()
    epoch_loss = 0
    for i, (item, labels) in enumerate(train_dataloader):
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in item.items()}
        pred = model(**batch).logits
        loss = compute_loss(pred, labels.to(device))
        epoch_loss += loss
        loss.backward()
        optimizer.step()
        metrics = compute_metrics(pred.detach().cpu().numpy(), labels.cpu().numpy())
        print(metrics)
    print(f"epoch: {epoch_num} train loss: {float(epoch_loss)}, ")


    val_loss = 0
    val_pred = [] ## val data 
    val_labels = [] ##
    with torch.no_grad():
        for i, (item, labels) in enumerate(val_dataloader):
            batch = {k: v.to(device) for k, v in item.items()}
            pred = model(**batch).logits
            loss = compute_loss(pred, labels.to(device))
            val_loss += loss

            val_pred에 batch 단위 pred를 extend
            val_labels에
        
        