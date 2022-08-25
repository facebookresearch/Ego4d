import numpy as np
import torch
import torch.nn as nn

from ego4d.research.clep.config import ModelConfig


def _get_layers(initial_dim, config):
    return [
        nn.Linear(initial_dim, config.final_proj_size),
        nn.ReLU(True),
        nn.Linear(config.final_proj_size, config.final_proj_size),
    ]


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
            torch.nn.init.xavier_uniform_(
                module.weight.data, gain=torch.nn.init.calculate_gain("relu")
            )
            module.bias.data.zero_()
