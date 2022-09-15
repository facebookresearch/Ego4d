from dataclasses import dataclass
from typing import List, Optional

from ego4d.research.common import SlurmConfig


@dataclass
class InputConfig:
    feature_path: str
    metadata_path: str
    features_per_second: float
    narration_width_sample_sec: float
    max_num_feature_vec_video_uids: int
    dsets_to_use: List[str]


@dataclass
class ModelConfig:
    nlp_feature_size: int
    visual_feature_size: int
    final_proj_size: int


@dataclass
class K400PreprocessConfig:
    dataset_dir: str
    set_to_use: str
    root_dir: str
    viz_feature_path: str
    metadata_out_path: str
    feature_extract_config_path: str
    num_labels_per_machine: int


@dataclass
class EgoPreprocessFeatureConfig:
    hdf5_path: str


@dataclass
class EgoPreprocessNarrConfig:
    narration_json_path: str
    num_workers: int
    st_model_name: str
    accelerator: str
    root_dir: str
    metadata_out_path: str
    narration_out_dir: str
    limit: Optional[int]
    num_narrs_per_machine: int


@dataclass
class EgoCharadePreprocessConfig:
    set_path: str
    video_root_path: str
    class_desc_path: str
    out_path: str
    out_label_path: str
    num_vids_per_machine: int


@dataclass
class CCPreprocessConfig:
    in_path: str
    helper_workers: int

    hdf5_viz_path: str
    hdf5_sent_path: str
    meta_path: str
    batch_size: int
    num_workers: int
    prefetch_factor: int
    imgs_per_gpu: int


@dataclass
class PreprocessConfig:
    slurm_config: SlurmConfig

    mode: str
    root_dir: str
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

    use_soft_loss: Optional[bool]
    soft_loss_threshold: float
    use_bce: bool
    use_logit_scale: bool
    norm_logits: bool
