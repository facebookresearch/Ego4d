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
class EgoPreprocessFeatureConfig:
    hdf5_path: str


@dataclass
class EgoPreprocessNarrConfig:
    narration_json_path: str
    num_workers: int
    st_model_name: str
    accelerator: str
    pre_root_dir: str
    metadata_out_path: str
    narration_out_path: str

    slurm_log_folder: str
    timeout_min: int
    constraint: str
    slurm_partition: str
    slurm_array_parallelism: int
    gpus_per_node: int
    cpus_per_task: int


@dataclass
class EgoCharadePreprocessConfig:
    out_path: str
    out_label_path: str

    slurm_log_folder: str
    timeout_min: int
    constraint: str
    slurm_partition: str
    slurm_array_parallelism: int
    gpus_per_node: int
    cpus_per_task: int

    num_vids_per_machine: int


@dataclass
class CCPreprocessConfig:
    hdf5_viz_path: str
    hdf5_sent_path: str
    meta_path: str
    batch_size: int
    num_workers: int
    prefetch_factor: int
    imgs_per_gpu: int

    slurm_log_folder: str
    timeout_min: int
    constraint: str
    slurm_partition: str
    slurm_array_parallelism: int
    gpus_per_node: int
    cpus_per_task: int


@dataclass
class PreprocessConfig:
    ego4d_narr: EgoPreprocessNarrConfig
    ego4d_features: EgoPreprocessFeatureConfig
    k400: K400PreprocessConfig
    ego_charade: EgoCharadePreprocessConfig
    cc: CCPreprocessConfig


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

    run_locally: bool
    tb_log_dir: str
    tb_log_name: str
    slurm_log_folder: str

    lr: float
    beta1: float
    beta2: float
    wd: float
    eps: float

    eval_per_iter: int
    eval_init: bool

    preprocess_mode: str

    use_soft_loss: Optional[bool]
    soft_loss_threshold: float
    use_bce: bool
    use_logit_scale: bool
    norm_logits: bool
