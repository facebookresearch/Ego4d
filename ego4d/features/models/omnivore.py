# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

from dataclasses import dataclass
from typing import Tuple

import torch
from ego4d.features.config import BaseModelConfig, InferenceConfig
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale
from torch.nn import Identity, Module
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo


@dataclass
class ModelConfig(BaseModelConfig):
    model_name: str = "omnivore_swinB"
    input_type: str = "video"
    side_size: int = 256
    crop_size: int = 224
    mean: Tuple[float] = (0.485, 0.456, 0.406)
    std: Tuple[float] = (0.229, 0.224, 0.225)


class WrapModel(Module):
    def __init__(self, model: Module, input_type: str):
        super().__init__()
        self.model = model
        self.input_type = input_type

    def forward(self, x) -> torch.Tensor:
        return self.model(x["video"], input_type=self.input_type)


def load_model(
    inference_config: InferenceConfig,
    config: ModelConfig,
    patch_final_layer: bool = True,
) -> Module:
    model = torch.hub.load("facebookresearch/omnivore", model=config.model_name)

    if patch_final_layer:
        model.heads.image = Identity()
        model.heads.video = Identity()
        model.heads.rgbd = Identity()

    # Set to GPU or CPU
    model = WrapModel(model, config.input_type)
    model = model.eval()
    model = model.to(inference_config.device)
    return model


def norm_pixels(x):
    return x / 255.0


def get_transform(inference_config: InferenceConfig, config: ModelConfig):
    if config.input_type == "video":
        transforms = [
            Lambda(norm_pixels),
            NormalizeVideo(config.mean, config.std),
            ShortSideScale(size=config.side_size),
            CenterCropVideo(config.crop_size),
        ]
    else:
        assert inference_config.frame_window == 1
        transforms = [
            Lambda(norm_pixels),
            NormalizeVideo(config.mean, config.std),
            ShortSideScale(size=config.side_size),
            CenterCropVideo(config.crop_size),
        ]

    return ApplyTransformToKey(
        key="video",
        transform=Compose(transforms),
    )
