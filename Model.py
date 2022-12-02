import torch
from torch import nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig, AutoModel
import transformers
from typing import Dict


class Type_Entity_Mean_Model(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        hidden_size = self.model.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 30),
        )

    def forward(self, batch, markers):
        outputs = self.model(**batch)

        # subject, object embedding의 평균
        last_hidden = outputs.last_hidden_state
        batch_output_list = []
        for _ in range(last_hidden.size(0)):
            ss = markers["ss"][_]
            se = markers["se"][_]
            os = markers["os"][_]
            oe = markers["oe"][_]
            subj_output = torch.mean(last_hidden[_][ss : se + 1], dim=0)  # (768)
            obj_output = torch.mean(last_hidden[_][os : oe + 1], dim=0)  # (768)

            batch_output_list.append(
                torch.cat((subj_output, obj_output), dim=0)
            )  # (768*3)

        batch_output = torch.stack(batch_output_list)
        logits = self.classifier(batch_output)

        return logits


class Type_Entity_SSOS_Model(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        hidden_size = self.model.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 30),
        )

    def forward(self, batch, markers):
        outputs = self.model(**batch)

        pooled_output = (
            outputs.last_hidden_state
        )  # 0번 last_hidden_state, 1번 pooler_output

        # subject, object의 첫번째 embedding
        idx = torch.arange(batch["input_ids"].size(0)).to(batch["input_ids"].device)
        ss_emb = pooled_output[idx, markers["ss"]]
        os_emb = pooled_output[idx, markers["os"]]
        h = torch.cat((ss_emb, os_emb), dim=-1)
        logits = self.classifier(h)

        return logits


class Type_Entity_LSTM_Model(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        hidden_size = self.model.config.hidden_size

        self.bilstm = torch.nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.1,
            bidirectional=True,
        )  # input_size, hidden_size, num_layers

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 30),
        )

    def forward(self, batch):
        outputs = self.model(**batch)

        pooled_output = (
            outputs.last_hidden_state
        )  # 0번 last_hidden_state, 1번 pooler_output

        # BiLSTM -> FC
        hidden, (last_hidden, last_cell) = self.bilstm(
            pooled_output
        )  # bilstm이므로 forward, backward hidden state가 존재(hidden_size * 2)
        fb_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
        # hidden : (batch, max_len, hidden_dim * 2)
        # last_hidden : (2, batch, hidden_dim)
        # fb_hidden : (batch, hidden_dim * 2)

        logits = self.classifier(fb_hidden)

        return logits


class Type_Entity_LSTM_T5Model(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = transformers.T5EncoderModel.from_pretrained(model_name)
        hidden_size = self.model.config.hidden_size

        self.bilstm = torch.nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.1,
            bidirectional=True,
        )  # input_size, hidden_size, num_layers

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 30),
        )

    def forward(self, batch):
        outputs = self.model(**batch)

        pooled_output = (
            outputs.last_hidden_state
        )  # 0번 last_hidden_state, 1번 pooler_output

        # BiLSTM -> FC
        hidden, (last_hidden, last_cell) = self.bilstm(
            pooled_output
        )  # bilstm이므로 forward, backward hidden state가 존재(hidden_size * 2)
        fb_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
        # hidden : (batch, max_len, hidden_dim * 2)
        # last_hidden : (2, batch, hidden_dim)
        # fb_hidden : (batch, hidden_dim * 2)

        logits = self.classifier(fb_hidden)

        return logits


class SimplePooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.d_model, config.d_model)
        self.dense2 = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        # hidden states: [batch_size, seq, model_dim]
        # attention masks: [batch_size, seq, 1]
        first_token_tensor = hidden_states[:, 0]

        pooled_output = self.dense1(first_token_tensor)
        pooled_output = F.relu(pooled_output)
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.dense2(pooled_output)

        return pooled_output


class Entity_Mean_T5Model(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = transformers.T5EncoderModel.from_pretrained(model_name)
        self.num_labels = 30

        self.model_dim = self.model.config.d_model

        self.pooler = SimplePooler(self.model.config)
        self.dropout = nn.Dropout(0.1)
        self.fc_layer = nn.Sequential(nn.Linear(self.model_dim, self.model_dim))
        self.classifier = nn.Sequential(nn.Linear(self.model_dim * 3, self.num_labels))

    def forward(self, batch, markers):
        outputs = self.model(**batch)

        pooled_output = outputs.last_hidden_state

        # CLS -> pooler layer
        last_hidden_state = outputs[0]
        pooled_output = self.pooler(last_hidden_state)
        pooled_output = self.dropout(pooled_output)

        sub_output = []
        obj_output = []

        for _ in range(last_hidden_state.size(0)):  # 배치단위로 가져옴
            ss = markers["ss"][_]
            se = markers["se"][_]
            os = markers["os"][_]
            oe = markers["oe"][_]
            sub_output.append(
                torch.mean(last_hidden_state[_, ss:se, :], dim=0).unsqueeze(0)
            )
            obj_output.append(
                torch.mean(last_hidden_state[_, os:oe, :], dim=0).unsqueeze(0)
            )
        sub_hidden_mean_cat = self.fc_layer(torch.cat((sub_output)))
        sub_hidden_mean_cat = self.dropout(sub_hidden_mean_cat)
        obj_hidden_mean_cat = self.fc_layer(torch.cat((obj_output)))
        obj_hidden_mean_cat = self.dropout(obj_hidden_mean_cat)

        entities_concat = torch.cat(
            [pooled_output, sub_hidden_mean_cat, obj_hidden_mean_cat], dim=-1
        )

        logits = self.classifier(entities_concat)

        return logits
