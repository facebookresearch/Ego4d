from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class InputConfig:
    feature_path: str
    features_per_second: float


@dataclass
class ModelConfig:
    # TODO: use features, etc.
    nlp_feature_size: int
    visual_feature_size: int
    proj_dims: List[int]


@dataclass
class PreprocessConfig:
    narration_json_path: str
    num_workers: int
    st_model_name: str
    accelerator: str
    pre_root_dir: str
    metadata_out_path: str
    narration_out_path: str


@dataclass
class TrainConfig:
    input_config: InputConfig
    model_config: ModelConfig
    pre_config: PreprocessConfig

    batch_size: int
    num_workers: int
    prefetch_factor: int

    num_epochs: int
    accelerator: str
    devices: int

    tb_log_dir: str
    tb_log_name: str
    slurm_log_folder: str
