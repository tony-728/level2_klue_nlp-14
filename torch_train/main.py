import torch

from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer

from Metric import *
from Dataset import *
from Model import Model
config = {
    'train_data_path' : './../dataset/train/sample_train.csv',
    'model_name' : "klue/bert-base",
    'epoch' : 1,
    'batch_size' : 32,
    'lr' : 1e-5
}
tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

train_dataset = TrainDataset(config['train_data_path'], tokenizer)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = config['batch_size'], shuffle=False)

model_config =  AutoConfig.from_pretrained(config['model_name'])
model_config.num_labels = 30
model = AutoModelForSequenceClassification.from_pretrained(config['model_name'], config=model_config)

optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

##training
model.to(device)
model.train()

for epoch_num in range(config['epoch']):
    epoch_loss = 0
    for i, (item, labels) in enumerate(train_dataloader):
        optimizer.zero_grad()
        batch = {k:v.to(device) for k,v in item.items()}
        pred = model(**batch).logits
        loss = compute_loss(pred, labels.to(device))
        epoch_loss += loss
        loss.backward()
        optimizer.step()
        metrics = compute_metrics(pred.detach().cpu().numpy(), labels.cpu())
        print(metrics)
    print("loss: ",float(epoch_loss))
        
