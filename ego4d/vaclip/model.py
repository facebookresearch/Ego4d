import torch
import torch.nn as nn
from torch.nn import Identity
import numpy as np

import omnivore

from ego4d.vaclip.config import ModelConfig


class EgoLangaugeAssociation(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        # TODO: audio encoder
        self.config = config
        self.visual_encoder = None
        self.text_encoder = None
        if config.pretrained_visual:
            self.visual_encoder = torch.hub.load("facebookresearch/omnivore", model="omnivore_swinT")
            self.visual_encoder.heads.image = Identity()
            self.visual_encoder.heads.video = Identity()
            self.visual_encoder.heads.rgbd = Identity()
        else:
            # TODO: don't get from hub
            self.visual_encoder = torch.hub.load("facebookresearch/omnivore", model="omnivore_swinT")
            self.visual_encoder.heads.image = Identity()
            self.visual_encoder.heads.video = Identity()
            self.visual_encoder.heads.rgbd = Identity()


        if config.pretrained_text:
            self.text_encoder = torch.hub.load('pytorch/fairseq', 'roberta.large')
        else:
            raise AssertionError("todo")

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # TODO: configure the 100
        self.text_proj = nn.Linear(self.config.nlp_feature_size, 100)
        self.visual_proj = nn.Linear(self.config.visual_feature_size, 100)

    def encode_visual(self, x, input_type: str):
        # iterate over each frame_window frames
        res = []

        clip_len = x[input_type].shape[1]
        for _ in range(self.config.sample_windows):
            idx = torch.randint(low=0, high=max(1, clip_len-self.config.window_size), size=(1,)).item()
            sub_sample = x[input_type][:, idx:idx+self.config.window_size, :, :, :]
            y = self.visual_encoder(sub_sample, input_type=input_type)
            if self.config.sample_windows <= 1:
                return self.visual_proj(y)
            res.append(y)
        return self.visual_proj(torch.mean(torch.stack(res), dim=0))

    def encode_text(self, x):
        inputs = []
        for text in x["text"]:
            tokens = self.text_encoder.encode(x["text"][0])
            last_layer_features = self.text_encoder.extract_features(tokens)
            inputs.append(last_layer_features.mean(dim=1))
            
        return self.text_proj(torch.stack(inputs).squeeze(1))

    def forward(self, x):
        ve = self.encode_visual(x, input_type="video")
        te = self.encode_text(x)
        return ve, te, self.logit_scale
