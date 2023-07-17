from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Input:
    from_frame_number: int
    to_frame_number: int
    take_name: Optional[str]
    take_uid: Optional[str]
    capture_root_dir: Optional[str]
    metadata_json_path: Optional[str]
    aria_trajectory_dir: Optional[str]
    exo_trajectory_dir: Optional[str]
    aria_streams: List[str]
    exo_timesync_name_to_calib_name: Optional[Dict[str, str]]


@dataclass
class PreprocessFrameConfig:
    dataset_name: str
    vrs_bin_path: str
    extract_all_aria_frames: bool
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
class TriangulationConfig:
    pass


@dataclass
class Config:
    legacy: bool
    data_dir: str
    cache_root_dir: str
    root_repo_dir: str
    gpu_id: int  # use -1 for CPU
    mode: str
    inputs: Input
    mode_preprocess: PreprocessFrameConfig
    mode_bbox: BBoxConfig
    mode_pose2d: PoseEstimationConfig
    mode_triangulate: TriangulationConfig
