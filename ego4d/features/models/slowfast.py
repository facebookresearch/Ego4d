from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from ego4d.features.config import InferenceConfig, BaseModelConfig
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torch.nn import Module, Identity
from torchvision.transforms import Compose, Lambda

# TODO
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)


@dataclass
class ModelConfig(BaseModelConfig):
    model_path: Optional[str] = None
    hub_path: Optional[str] = "slowfast_r101"
    slowfast_alpha: int = 4

    # transformation config
    side_size: int = 256
    crop_size: int = 256
    mean: Tuple[float] = (0.45, 0.45, 0.45)
    std: Tuple[float] = (0.225, 0.225, 0.225)


class GetFv(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        fv_s = x.shape[1]
        return x.view(bs, fv_s, -1).mean(2)


def load_model(inference_config: InferenceConfig, config: ModelConfig) -> Module:
    if config.model_path is not None:
        raise AssertionError("not supported yet")
        model = None
    else:
        assert config.hub_path is not None
        model = torch.hub.load(
            "facebookresearch/pytorchvideo", config.hub_path, pretrained=True
        )

    assert model is not None

    model.blocks[6] = GetFv()

    # Set to GPU or CPU
    model = model.eval()
    model = model.to(inference_config.device)
    return model


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """

    def __init__(self, slow_fast_alpha):
        super().__init__()
        self.slow_fast_alpha = slow_fast_alpha

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // self.slow_fast_alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


def get_transform(inference_config: InferenceConfig, config: ModelConfig):
    return ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(inference_config.frame_window),
                Lambda(lambda x: x / 255.0),  # scale
                NormalizeVideo(config.mean, config.std),  # normalize
                ShortSideScale(size=config.side_size),  # scale to 256x-1 or -1x256
                CenterCropVideo(config.crop_size),  # crop to 256x256
                PackPathway(config.slowfast_alpha),  # create pathway
            ]
        ),
    )
