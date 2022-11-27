import torch
from torch import nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel
from transformers import ElectraModel, ElectraTokenizer
from typing import Dict


class Model(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, batch, markers):
        outputs = self.model(**batch)

        return outputs
