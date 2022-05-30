import torch
import torch.nn as nn
from torch.nn import Identity
import numpy as np

from ego4d.vaclip.config import ModelConfig

class ResBlock(nn.Module):
    def __init__(self, in_f, proj_f):
        super().__init__()
        self.x1 = nn.Linear(in_f, proj_f)
        self.x2 = nn.Linear(proj_f, in_f)
        self.activation = nn.ReLU(True)
    
    def forward(self, x):
        x_prime = self.x1(x)
        x_prime = self.activation(x_prime)
        x_prime = self.x2(x_prime)
        return x_prime + x


class EgoLangaugeAssociation(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # TODO
        self.text_proj = nn.Linear(self.config.nlp_feature_size, 100)
        self.visual_proj = nn.Linear(self.config.visual_feature_size, 100)

    def forward(self, x):
        ve = self.visual_proj(x["video"])
        te = self.text_proj(x["text"])
        return ve, te, self.logit_scale
