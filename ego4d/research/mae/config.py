from dataclasses import dataclass
from typing import List, Optional

from ego4d.research.common import SlurmConfig

@dataclass
class Ego4DInput:
    video_root_dir: str

@dataclass
class EgoExoInput:
    root_dir: str
    metadata_path: str

@dataclass
class InputConfig:
    ego4d_input: Optional[Ego4DInput]
    egoexo_input: Optional[EgoExoInput]


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

    slurm_config: SlurmConfig

    checkpoint_dir: str
    checkpoint_metric: str
    batch_size: int
    num_workers: int
    prefetch_factor: int

    num_epochs: int
    accelerator: str
    devices: int

    tb_log_dir: str
    tb_log_name: str

    lr: float
    beta1: float
    beta2: float
    wd: float
    eps: float

    eval_per_iter: int
    eval_init: bool
