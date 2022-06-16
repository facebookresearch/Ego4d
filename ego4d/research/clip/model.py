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

def _get_layers_proj_dims(initial_dim, proj_dims, final_proj_size):
    assert (np.array(proj_dims) == proj_dims[0]).all()
    result = [
        nn.Linear(initial_dim, proj_dims[0]),
    ]
    for prev, nxt in zip(proj_dims[0:-1], proj_dims[:-1]):
        result += [
            nn.ReLU(True),
            ResBlock(prev, nxt),
        ]
    result += [
        nn.ReLU(True),
        nn.Linear(proj_dims[-1], final_proj_size)
    ]
    return result


def _get_layers(initial_dim, config):
    if len(config.proj_dims) == 1:
        return [nn.ReLU(True), nn.Linear(initial_dim, config.final_proj_size)]
    return _get_layers_proj_dims(initial_dim, config.proj_dims, config.final_proj_size)


class EgoLangaugeAssociation(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config
        txt_layers = _get_layers(config.nlp_feature_size, config)
        viz_layers = _get_layers(config.visual_feature_size, config)
        self.text_proj = nn.Sequential(*tuple(txt_layers))
        self.visual_proj = nn.Sequential(*tuple(viz_layers))

        self.apply(self.init_weights)

        # don't want to init this with 0
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x):
        ve = self.visual_proj(x["video"])
        te = self.text_proj(x["text"])
        return ve, te, self.logit_scale
    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight.data, gain=torch.nn.init.calculate_gain('relu'))
            module.bias.data.zero_()
