from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Input:
    from_frame_number: int
    to_frame_number: int
    metadata_json_path: str
    aria_trajectory_dir: str
    exo_trajectory_dir: str
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


@dataclass
class PoseEstimationConfig:
    pose_model_config: str
    pose_model_checkpoint: str
    use_preprocessed_frames: bool


@dataclass
class TriangulationConfig:
    pass


@dataclass
class Config:
    root_dir: str
    gpu_id: int  # use -1 for CPU
    mode: str  # one of {preprocess_frames,
    inputs: Input
    mode_preprocess: PreprocessFrameConfig
    mode_pose_estimation: PoseEstimationConfig
    mode_triangulate: TriangulationConfig
