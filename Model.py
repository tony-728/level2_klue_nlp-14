import torch
from torch import nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel
# from transformers import ElectraModel, ElectraTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from typing import Dict


class Model(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def forward(self, input, labels):
        outputs = self.model(input_ids=input['input_ids'], attention_mask=input['attention_mask'], labels=labels['input_ids'], decoder_attention_mask=labels['attention_mask'])

        return outputs

    def generate(self, **karg):
        return self.model.generate(**karg)