import torch
from torch import nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel
from transformers import ElectraModel, ElectraTokenizer
from typing import Dict


class Model(nn.Module):
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

        # subject, object의 첫번째 embedding
        pooled_output = (
            outputs.last_hidden_state
        )  # 0번 last_hidden_state, 1번 pooler_output

        idx = torch.arange(batch["input_ids"].size(0)).to(batch["input_ids"].device)
        ss_emb = pooled_output[idx, markers["ss"]]
        os_emb = pooled_output[idx, markers["os"]]
        h = torch.cat((ss_emb, os_emb), dim=-1)
        logits = self.classifier(h)

        # subject, object embedding의 평균
        # last_hidden = outputs.last_hidden_state
        # batch_output_list = []
        # for _ in range(last_hidden.size(0)):
        #     ss = markers["ss"][_]
        #     se = markers["se"][_]
        #     os = markers["os"][_]
        #     oe = markers["oe"][_]
        #     subj_output = torch.mean(last_hidden[_][ss : se + 1], dim=0)  # (768)
        #     obj_output = torch.mean(last_hidden[_][os : oe + 1], dim=0)  # (768)

        #     batch_output_list.append(
        #         torch.cat((subj_output, obj_output), dim=0)
        #     )  # (768*3)

        # batch_output = torch.stack(batch_output_list)
        # batch_output = self.classifier(batch_output)

        return logits
