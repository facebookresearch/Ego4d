# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

import importlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import pandas as pd
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf


@dataclass
class Video:
    """
    Description of a video
    """

    uid: str
    path: str
    frame_count: int
    is_stereo: bool = False


@dataclass
class InputOutputConfig:
    """
    Assumptions:
    1. videos are in a single subdirectory with the pattern: <dir>/<uid>.mp4
    2. the manifest file exists in the same subdirectory

    If you need to adjust any of these assumptions, then please refer to:
    - `_path_for`, and
    - `_uid_to_frame_count`
    """

    # input
    filter_completed: bool = True
    video_dir_path: str = "/datasets01/ego4d_track2/v1/full_scale/"
    ego4d_download_dir: str = "/checkpoint/miguelmartin/ego4d/"
    uid_list: Optional[List[str]] = None
    video_limit: int = -1
    debug_mode: bool = False
    debug_path: str = "/checkpoint/miguelmartin/ego4d_track2/v1/debug_frames"

    # output
    out_path: str = (
        "/checkpoint/miguelmartin/ego4d_track2_features/full_scale/v1_1/action_features"
    )


@dataclass
class InferenceConfig:
    device: str = "cuda"

    # 0 == don't use dataloader
    # >0 use dataloader with bs=batch_size
    batch_size: int = 1

    # only used if batch_size != 0
    num_workers: int = 9

    prefetch_factor: int = 2

    fps: int = 30
    frame_window: int = 32
    stride: int = 16
    include_audio: bool = False
    include_video: bool = True


@dataclass
class ScheduleConfig:
    run_locally: bool = False

    log_folder: str = "slurm_log/%j"

    # Scheduler Configuration
    timeout_min: int = int(12 * 60)
    constraint: str = "volta"
    slurm_partition: str = "pixar"
    slurm_array_parallelism: int = 256
    gpus_per_node: int = 1
    cpus_per_task: int = 10

    # Batching Configuration
    overhead: float = 2  # off in the worst case -- estimate will be wrong
    time_per_forward_pass: float = 0.8

    schedule_time_per_node: float = 10


@dataclass
class BaseModelConfig:
    pass


@dataclass(order=True)
class FeatureExtractConfig:
    io: InputOutputConfig
    inference_config: InferenceConfig
    schedule_config: ScheduleConfig
    model_config: BaseModelConfig
    model_module_str: str = "ego4d.features.models.slowfast"
    force_yes: bool = False


def get_model_module(config: FeatureExtractConfig):
    return importlib.import_module(config.model_module_str)


def _uids_for_dir(path: str) -> List[str]:
    ret = [
        p
        for p in os.listdir(path)
        if Path(p).suffix not in [".json", ".csv", ".csv"]
        and not p.startswith(".")
        and not p.startswith("manifest")
    ]
    return [Path(p).stem for p in ret]


def _path_for(config: InputOutputConfig, uid: str) -> str:
    return f"{config.video_dir_path}/{uid}.mp4"


def _unfiltered_uids(config: InputOutputConfig) -> List[str]:
    uids = config.uid_list
    if uids is None:
        assert config.video_dir_path is not None, "Not given any uids"
        uids = _uids_for_dir(config.video_dir_path)
    return uids


def _uids(config: InputOutputConfig) -> List[str]:
    uids = _unfiltered_uids(config)

    if config.filter_completed:
        completed_uids = set(_uids_for_dir(config.out_path))
        uids = [uid for uid in uids if uid not in completed_uids]

    assert uids is not None, "`uids` is None"
    assert len(uids) > 0, "`len(uids)` is 0"
    return uids


def _video_paths(config: InputOutputConfig, uids: List[str]) -> List[str]:
    return [_path_for(config, uid) for uid in uids]


def _uid_to_num_frames(config: InputOutputConfig) -> Dict[str, int]:
    manifest_df = pd.read_csv(f"{config.video_dir_path}/manifest.csv")
    return {row.video_uid: row.canonical_num_frames for row in manifest_df.itertuples()}


def _uid_to_is_stereo(config: InputOutputConfig) -> Dict[str, bool]:
    data_json = json.load(open(f"{config.ego4d_download_dir}/ego4d.json"))
    return {v["video_uid"]: v["is_stereo"] for v in data_json["videos"]}


def _videos(config: InputOutputConfig, unfiltered: bool = False) -> List[Video]:
    uids = _uids(config) if not unfiltered else _unfiltered_uids(config)
    uid_to_num_frames = _uid_to_num_frames(config)
    uids_to_is_stereo = _uid_to_is_stereo(config)
    return [
        Video(
            uid=uid,
            path=_path_for(config, uid),
            frame_count=uid_to_num_frames[uid],
            is_stereo=uids_to_is_stereo[uid],
        )
        for uid in uids
        if uid in uid_to_num_frames
    ]


def get_videos(config: FeatureExtractConfig) -> Tuple[List[Video], List[Video]]:
    """
    Return (videos_to_process, all_videos)
    """
    possibly_filtered_videos = _videos(config.io, unfiltered=False)
    all_videos = _videos(config.io, unfiltered=True)
    if config.io.video_limit > 0:
        random.shuffle(possibly_filtered_videos)
        return possibly_filtered_videos[0 : config.io.video_limit], all_videos
    return possibly_filtered_videos, all_videos


def get_transform(config: FeatureExtractConfig) -> Any:
    return get_model_module(config).get_transform(
        config.inference_config,
        config.model_config,
    )


def load_model(config: FeatureExtractConfig, patch_final_layer: bool = True) -> Any:
    module = get_model_module(config)
    return module.load_model(
        config.inference_config,
        config.model_config,
        patch_final_layer=patch_final_layer,
    )


@hydra.main(config_path="configs", config_name=None)
def test_load_config(config: FeatureExtractConfig):
    print(
        f"""
    Config:

{OmegaConf.to_yaml(config)}
    """
    )


if __name__ == "__main__":
    from ego4d.features.models.slowfast import ModelConfig

    cs = ConfigStore.instance()
    cs.store(
        name="default",
        node=FeatureExtractConfig(
            io=InputOutputConfig(),
            inference_config=InferenceConfig(),
            schedule_config=ScheduleConfig(),
            model_config=ModelConfig(),
        ),
    )

    test_load_config()  # pyre-ignore
