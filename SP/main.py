import torch
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

from transformers import RobertaForQuestionAnswering

from tqdm import tqdm

from load_data import SP_Train_Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_path = "/opt/ml/dataset/train/sample_train.csv"
tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
train_dataset = SP_Train_Dataset(data_path, tokenizer)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
Model = RobertaForQuestionAnswering.from_pretrained("klue/roberta-large")

#training
Model.to(device)
Model.train()
#for epoch_num in range(config['epoch']):


for epoch_num in range(1):
    with tqdm(train_dataloader, unit="batch") as tepoch:
        for step, (question1_set, question2_set, labels) in enumerate(train_dataloader): 
            tepoch.set_description(f"Epoch {epoch_num}")

            question1_batch = {}


q_set1, q_set2, label = next(iter(train_dataloader))
output =  Model(input_ids = q_set1['input_ids'], token_type_ids = q_set1['token_type_ids'], attention_mask = q_set1['attention_mask'])