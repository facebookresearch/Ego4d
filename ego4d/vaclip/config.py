from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class InputConfig:
    narration_json_path: str = "/datasets01/ego4d_track2/v1/annotations/narration.json"
    video_dir_path: str = "/datasets01/ego4d_track2/v1/full_scale/"


@dataclass
class TransformConfig:
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    side_size: int
    crop_size: int


@dataclass
class ModelConfig:
    # TODO: use features, etc.
    pretrained_text: bool
    pretrained_visual: bool
    sample_windows: int
    window_size: int
    nlp_feature_size: int
    visual_feature_size: int


@dataclass
class TrainConfig:
    input_config: InputConfig
    model_config: ModelConfig

    batch_size: int
    num_workers: int
    prefetch_factor: int

    num_epochs: int
    accelerator: str
    devices: int
