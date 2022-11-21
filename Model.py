import torch
from torch import nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel

from typing import Dict


class Model(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.linear = nn.Linear(768*3, 30)
        self.softmax = nn.Softmax()
    def forward(self, batch, markers):
        last_hidden = self.model(**batch).last_hidden_state
        batch_output_list = []
        for _ in range(last_hidden.size(0)):
            ss = markers['ss'][_]
            se = markers['se'][_]
            os = markers['os'][_]
            oe = markers['oe'][_]
            cls_output = last_hidden[_][0] #(768)
            subj_output = torch.mean(last_hidden[_][ss:se+1], dim = 0) #(768)
            obj_output = torch.mean(last_hidden[_][os:oe+1], dim = 0) #(768)

            batch_output_list.append(torch.cat((cls_output, subj_output, obj_output), dim = 0)) #(768*3)

        batch_output = torch.stack(batch_output_list)
        batch_output = self.linear(batch_output)
        probability_distribution = self.softmax(batch_output)

        return probability_distribution
        