import torch
from torch import nn as nn

class Model(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        print(inputs)
        output = self.model(inputs)
        return output
        