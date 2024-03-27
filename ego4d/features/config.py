# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

import importlib
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import pandas as pd
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from torchaudio.transforms import Resample
from torchvision.transforms import Compose, Lambda


@dataclass
class Video:
    """
    Description of a video
    """

    uid: str
    path: str
    frame_count: int
    w: int
    h: int
    has_audio: bool
    is_stereo: bool = False

    @property
    def dim(self) -> int:
        return (self.w * self.h) / (2 if self.is_stereo else 1)


@dataclass
class NormalizationConfig:
    normalize_audio: bool = False
    resample_audio_rate: int = 16000
    resampling_method: str = "sinc_interpolation"


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

    # output
    out_path: str

    # input
    # TODO: add file list
    filter_completed: bool = True
    video_dir_path: str = "/datasets01/ego4d_track2/v1/full_scale/"
    ego4d_download_dir: str = "/checkpoint/miguelmartin/ego4d/"
    dataset_version: str = "ego4d"
    egoexo_data_dir: str = "/checkpoint/miguelmartin/egoexo_data/dev/"
    uid_list: Optional[List[str]] = None
    video_limit: int = -1
    debug_mode: bool = False
    debug_path: Optional[str] = None
    exclude_no_audio: bool = False
    eligible_cam_prefixes: Optional[List[str]] = None


@dataclass
class InferenceConfig:
    device: str = "cuda"

    video_reader_class: str = "PyAvReader"
    video_reader_kwargs_override: dict = field(default_factory=lambda _: {})

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
    norm_config: NormalizationConfig = field(default_factory=NormalizationConfig)


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
    check_fv_count: bool = True


def get_model_module(config: FeatureExtractConfig):
    return importlib.import_module(config.model_module_str)


def _uids_for_dir(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
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
    assert len(uids) >= 0, "`len(uids)` is 0"
    return uids


def _video_paths(config: InputOutputConfig, uids: List[str]) -> List[str]:
    return [_path_for(config, uid) for uid in uids]


def _uid_to_info(config: InputOutputConfig) -> Dict[str, int]:
    manifest_df = pd.read_csv(f"{config.video_dir_path}/manifest.csv")
    return {
        row.video_uid: {
            "num_frames": row.canonical_num_frames,
            "w": row.canonical_display_width,
            "h": row.canonical_display_height,
            "has_audio": not (
                pd.isnull(row.canonical_audio_start_sec)
                and pd.isnull(row.canonical_audio_duration_sec)
            ),
        }
        for row in manifest_df.itertuples()
    }


def _uid_to_is_stereo(config: InputOutputConfig) -> Dict[str, bool]:
    data_json = json.load(open(f"{config.ego4d_download_dir}/ego4d.json"))
    return {v["video_uid"]: v["is_stereo"] for v in data_json["videos"]}


def _videos(config: InputOutputConfig, unfiltered: bool = False) -> List[Video]:
    if config.dataset_version == "ego4d":
        uids = _uids(config) if not unfiltered else _unfiltered_uids(config)
        uid_to_info = _uid_to_info(config)
        uids_to_is_stereo = _uid_to_is_stereo(config)
        videos = [
            Video(
                uid=uid,
                path=_path_for(config, uid),
                frame_count=uid_to_info[uid]["num_frames"],
                w=uid_to_info[uid]["w"],
                h=uid_to_info[uid]["h"],
                has_audio=uid_to_info[uid]["has_audio"],
                is_stereo=uids_to_is_stereo[uid],
            )
            for uid in uids
            if uid in uid_to_info
        ]
        if config.exclude_no_audio:
            return [v for v in videos if v.has_audio]

        return videos
    else:
        takes = json.load(open(os.path.join(config.egoexo_data_dir, "takes.json")))
        all_uids = [t["take_uid"] for t in takes]
        uids = config.uid_list
        if uids is None:
            uids = all_uids
        if uids and takes:
            uid_takes = [t for t in takes if t["take_uid"] in uids]
            if len(uid_takes) < len(takes):
                print(f"Filtered {len(takes)} -> {len(uid_takes)} on uid config")
                takes = uid_takes
        completed_uids = set(_uids_for_dir(config.out_path))
        videos = []
        for take in takes:
            for cam_id, streams in take["frame_aligned_videos"].items():
                eligible_prefixes = config.eligible_cam_prefixes or [
                    "cam",
                    "aria",
                    "gp",
                ]
                if not any(x in cam_id.lower() for x in eligible_prefixes):
                    continue
                for stream_name, stream in streams.items():
                    # Config?
                    if "aria" in cam_id and stream_name != "rgb":
                        continue
                    is_aria = "aria" in stream["cam_id"]
                    # NOTE: known constants for not downsampled videos
                    w = 1408 if is_aria else 3840
                    h = 1408 if is_aria else 2160
                    uid = f"{take['take_uid']}_{cam_id}_{stream_name}"
                    if (
                        not unfiltered
                        and config.filter_completed
                        and uid in completed_uids
                    ):
                        continue

                    path = os.path.join(
                        config.egoexo_data_dir,
                        take["root_dir"],
                        stream["relative_path"],
                    )

                    if not os.path.exists(path):
                        continue

                    videos.append(
                        Video(
                            uid=uid,
                            path=path,
                            # NOTE: w/h used to estimate time to complete
                            w=w,
                            h=h,
                            frame_count=take["timesync_end_idx"]
                            - take["timesync_start_idx"]
                            - 1,
                            has_audio=False,
                            is_stereo=False,
                        )
                    )
        return videos


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
    ic = config.inference_config
    nc = ic.norm_config
    model_transform = get_model_module(config).get_transform(ic, config.model_config)
    transforms = []
    if hasattr(config, "norm_config") and config.norm_config.normalize_audio:
        print(f"Normalizing with: {config.norm_config}")

        def resample_audio(x):
            return Resample(
                orig_freq=x["audio_sample_rate"],
                new_freq=nc.resample_audio_rate,
                resampling_method=nc.resampling_method,
            )

        transforms += [Lambda(resample_audio)]

    transforms += [model_transform]
    return Compose(transforms)


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
