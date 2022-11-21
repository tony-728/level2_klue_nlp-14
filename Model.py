import torch
from torch import nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel

from typing import Dict


class Model(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        hidden_state = self.model.config.hidden_size
        self.linear = nn.Linear(hidden_state*3, 30)

    def forward(self, batch, markers):
        output = self.model(**batch)
        cls_outputs = output.pooler_output
        last_hidden = output.last_hidden_state
        batch_output_list = []
        for _ in range(last_hidden.size(0)):
            ss = markers['ss'][_]
            se = markers['se'][_]
            os = markers['os'][_]
            oe = markers['oe'][_]
            cls_output = cls_outputs[_]
            subj_output = torch.mean(last_hidden[_][ss:se+1], dim = 0) #(768)
            obj_output = torch.mean(last_hidden[_][os:oe+1], dim = 0) #(768)

            batch_output_list.append(torch.cat((cls_output, subj_output, obj_output), dim = 0)) #(768*3)

        batch_output = torch.stack(batch_output_list)
        batch_output = self.linear(batch_output)

        return batch_output
        