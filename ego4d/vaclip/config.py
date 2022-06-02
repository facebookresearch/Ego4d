from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class InputConfig:
    feature_path: str
    features_per_second: float
    narration_width_sample_sec: float
    max_num_feature_vec_video_uids: int


@dataclass
class ModelConfig:
    # TODO: use features, etc.
    nlp_feature_size: int
    visual_feature_size: int
    proj_dims: List[int]


@dataclass
class K400PreprocessConfig:
    dataset_dir: str
    set_to_use: str
    pre_root_dir: str
    viz_feature_dir: str
    metadata_out_path: str
    feature_extract_config_path: str
    num_labels_per_machine: int
    slurm_log_folder: str
    timeout_min: int
    constraint: str
    slurm_partition: str
    slurm_array_parallelism: int
    gpus_per_node: int
    cpus_per_task: int


@dataclass
class EgoPreprocessConfig:
    narration_json_path: str
    num_workers: int
    st_model_name: str
    accelerator: str
    pre_root_dir: str
    metadata_out_path: str
    narration_out_path: str

    timeout_min: int
    constraint: str
    slurm_partition: str
    slurm_array_parallelism: int
    gpus_per_node: int
    cpus_per_task: int


@dataclass
class TrainConfig:
    input_config: InputConfig
    model_config: ModelConfig
    ego_pre_config: EgoPreprocessConfig
    k400_pre_config: K400PreprocessConfig

    batch_size: int
    num_workers: int
    prefetch_factor: int

    num_epochs: int
    accelerator: str
    devices: int

    run_locally: bool
    tb_log_dir: str
    tb_log_name: str
    slurm_log_folder: str

    lr: float
    beta1: float
    beta2: float
    wd: float
    eps: float

    preprocess_mode: str
