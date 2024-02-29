# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

from dataclasses import dataclass
from typing import Tuple

import torch
import functools
from ego4d.features.config import BaseModelConfig, InferenceConfig
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale
from torch.nn import Module
from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    ToTensor,
    Normalize,
    Lambda,
)
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo
from maws.model_builder import build_model


@dataclass
class ModelConfig(BaseModelConfig):
    model_name: str = "vit_2b14_xlmr_l"
    base_model: str = "maws_clip"
    input_type: str = "video"
    side_size: int = 224
    crop_size: int = 224
    mean: Tuple[float] = (0.485, 0.456, 0.406)
    std: Tuple[float] = (0.229, 0.224, 0.225)


class WrapModel(Module):
    def __init__(self, model: Module):
        super().__init__()
        self.model = model

    def forward(self, x) -> torch.Tensor:
        imgs = x["video"].half()
        imgs = imgs.view(-1, 3, imgs.shape[-2], imgs.shape[-1])
        return self.model.encode_images(imgs)


def load_model(
    inference_config: InferenceConfig,
    config: ModelConfig,
    patch_final_layer: bool = True,
) -> Module:
    assert patch_final_layer, "maws does not provide a head"

    model = build_model(config.model_name, config.base_model)
    # model_name = f"{config.model_name}_{config.base_model}"
    # model = torch.hub.load("facebookresearch/maws", model=model_name)

    # Set to GPU or CPU
    model = WrapModel(model)
    model = model.to(inference_config.device)
    model = model.eval().half()
    return model


def norm_pixels(x):
    return x / 255.0

def video_to_image(x, size):
    # x = x.permute(1, 2, 0, 3, 4).view(-1, 3, size, size)
    x = x.permute(1, 0, 2, 3)
    return x


def get_transform(inference_config: InferenceConfig, config: ModelConfig):
    assert inference_config.frame_window == 1
    transforms = [
        Lambda(norm_pixels),
        NormalizeVideo(config.mean, config.std),
        ShortSideScale(size=config.side_size),
        CenterCropVideo(config.crop_size),
        Lambda(functools.partial(video_to_image, size=config.crop_size)),
    ]

    return ApplyTransformToKey(
        key="video",
        transform=Compose(transforms),
    )