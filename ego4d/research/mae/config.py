from dataclasses import dataclass
from typing import List, Optional

from ego4d.research.common import SlurmConfig

@dataclass
class InputConfig:
    video_path: str


@dataclass
class ModelConfig:
    model_name: str


@dataclass
class PreprocessConfig:
    # TODO
    pass


@dataclass
class TrainConfig:
    input_config: InputConfig
    model_config: ModelConfig
    pre_config: PreprocessConfig

    checkpoint_dir: str
    checkpoint_metric: str
    batch_size: int
    num_workers: int
    prefetch_factor: int

    num_epochs: int
    accelerator: str
    devices: int

    run_locally: bool
    tb_log_dir: str
    tb_log_name: str

    lr: float
    beta1: float
    beta2: float
    wd: float
    eps: float

    eval_per_iter: int
    eval_init: bool
