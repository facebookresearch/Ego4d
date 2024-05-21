from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Input:
    from_frame_number: int
    to_frame_number: int
    sample_interval: int
    subclip_json_dir: Optional[str]
    # min_subclip_length:
    # example: suppose min_subclip_length=120 (unit: frames) and subclip=[30, 80],
    # it will extend the subclip to [30, 150]. This could be useful for some actions
    # that take more than 2 seconds (60 frames) so that we sample sufficient frames
    # for the action.
    # see function `calculate_frame_selection` for how it's used
    min_subclip_length: int
    take_name: Optional[str]
    take_uid: Optional[str]
    capture_root_dir: Optional[str]
    metadata_json_path: Optional[str]
    aria_trajectory_dir: Optional[str]
    exo_trajectory_dir: Optional[str]
    aria_streams: List[str]
    exo_timesync_name_to_calib_name: Optional[Dict[str, str]]


@dataclass
class Output:
    # storage_level:
    # 0 is minimum storage, meaning most stuff shall be cleaned up after the job is done
    # at the cost of losing debugging information;
    # the higher the level is, the more disk storage we are allow to use
    # (e.g., for debugging purpose)
    storage_level: int


@dataclass
class PreprocessFrameConfig:
    dataset_name: str
    vrs_bin_path: str
    extract_all_aria_frames: bool
    extract_frames: bool
    download_video_files: bool
    force_download: bool


@dataclass
class BBoxConfig:
    detector_model_config: Optional[str]
    detector_model_checkpoint: Optional[str]
    use_aria_trajectory: bool
    human_height: float
    human_radius: float
    min_bbox_score: float
    min_area_ratio: float


@dataclass
class PoseEstimationConfig:
    pose_config: str
    pose_checkpoint: str
    dummy_pose_config: str
    dummy_pose_checkpoint: str


@dataclass
class Pose3DConfig:
    min_body_kpt2d_conf: float


@dataclass
class TriangulationConfig:
    pass


@dataclass
class Config:
    legacy: bool
    data_dir: str
    cache_root_dir: str
    repo_root_dir: str
    gpu_id: int  # use -1 for CPU
    mode: str
    inputs: Input
    outputs: Output
    mode_preprocess: PreprocessFrameConfig
    mode_bbox: BBoxConfig
    mode_pose2d: PoseEstimationConfig
    mode_pose3d: Pose3DConfig
    mode_triangulate: TriangulationConfig
