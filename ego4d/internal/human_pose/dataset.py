import json
import os
from typing import Any, Dict, List, Optional

import cv2
import pandas as pd

from ego4d.internal.human_pose.readers import TorchAudioStreamReader

from iopath.common.file_io import PathManager
from iopath.common.s3 import S3PathHandler

pathmgr = PathManager()
pathmgr.register_handler(S3PathHandler(profile="default"))


def _get_synced_timesync_df(timesync_df):
    # start idx
    cam01_idx = timesync_df.cam01_global_time.first_valid_index()
    cam02_idx = timesync_df.cam02_global_time.first_valid_index()
    cam03_idx = timesync_df.cam03_global_time.first_valid_index()
    cam04_idx = timesync_df.cam04_global_time.first_valid_index()
    aria_rgb_idx = timesync_df["aria01_214-1_global_time"].first_valid_index()
    aria_slam1_idx = timesync_df["aria01_1201-1_global_time"].first_valid_index()
    aria_slam2_idx = timesync_df["aria01_1201-2_global_time"].first_valid_index()
    print(
        cam01_idx,
        cam02_idx,
        cam03_idx,
        cam04_idx,
        aria_rgb_idx,
        aria_slam1_idx,
        aria_slam2_idx,
    )
    first_idx = max(
        cam01_idx,
        cam02_idx,
        cam03_idx,
        cam04_idx,
        aria_rgb_idx,
        aria_slam1_idx,
        aria_slam2_idx,
    )

    # end idx
    last_cam01_idx = timesync_df.cam01_global_time.last_valid_index()
    last_cam02_idx = timesync_df.cam02_global_time.last_valid_index()
    last_cam03_idx = timesync_df.cam03_global_time.last_valid_index()
    last_cam04_idx = timesync_df.cam04_global_time.last_valid_index()
    aria_rgb_last_idx = timesync_df["aria01_214-1_global_time"].last_valid_index()
    aria_slam1_last_idx = timesync_df["aria01_1201-1_global_time"].last_valid_index()
    aria_slam2_last_idx = timesync_df["aria01_1201-2_global_time"].last_valid_index()
    print(
        aria_rgb_last_idx,
        aria_slam1_last_idx,
        aria_slam2_last_idx,
        last_cam01_idx,
        last_cam02_idx,
        last_cam03_idx,
        last_cam04_idx,
    )
    last_idx = min(
        aria_rgb_last_idx,
        aria_slam1_last_idx,
        aria_slam2_last_idx,
        last_cam01_idx,
        last_cam02_idx,
        last_cam03_idx,
        last_cam04_idx,
    )

    return timesync_df.iloc[first_idx : last_idx + 1]


def _get_accurate_timestamps(aria_path, stream_id):
    # NOTE if used with pyark datatools, pyvrs will cause a segfault
    from pyvrs import SyncVRSReader

    vrs_r = SyncVRSReader(aria_path, auto_read_configuration_records=True)
    rgb_stream = vrs_r.filtered_by_fields(stream_ids={stream_id}, record_types="data")

    return {idx: f.timestamp * 1e6 for idx, f in enumerate(rgb_stream)}


def get_synced_timesync_df(metadata_json):
    timesync_df = pd.read_csv(pathmgr.open(metadata_json["timesync_csv_path"]))
    return _get_synced_timesync_df(timesync_df)


# TODO: changeme to support a dynamic dataset, similar to what is present in notebook
class SyncedEgoExoCaptureDset:
    def __init__(
        self,
        root_dir: str,
        dataset_json_path: str,
        read_frames: bool,
    ):
        self.dataset_json = json.load(open(dataset_json_path))
        self.read_frames = read_frames
        self.root_dir = root_dir
        self.cache_dir = self.dataset_json["dataset_dir"]
        self.frame_dir = os.path.join(self.root_dir, self.cache_dir, "frames")

    def __getitem__(self, idx):
        row = self.dataset_json["frames"][idx]
        for cam_id in row.keys():
            # transform path to be absolute
            frame_path = os.path.join(self.frame_dir, row[cam_id]["frame_path"])
            row[cam_id]["abs_frame_path"] = frame_path
            if self.read_frames:
                row[cam_id]["frame"] = cv2.imread(frame_path)

        return row

    def all_cam_ids(self):
        return self.dataset_json["frames"][0].keys()

    def __len__(self):
        return len(self.dataset_json["frames"])


class FrameDirDset:
    def __init__(self, directory: str, read_file: bool = True, ext: str = ".jpg"):
        frames = [f for f in os.listdir(directory) if f.endswith(ext)]
        frames = sorted(frames, key=lambda x: int(x.split(".")[0]))
        self.frame_paths = [os.path.join(directory, f) for f in frames]
        self.read_file = read_file
        self.directory = directory

    def __getitem__(self, idx):
        if self.read_file:
            return cv2.imread(self.frame_paths[idx])
        return self.frame_paths[idx]

    def __len__(self):
        return len(self.frame_paths)


# NOTE(miguelmartin):
# using this is probably not optimal for extraction
# we can speed things up by setting frame_window_size > 1
# e.g.
# 50ms to load a single frame in GPU memory vs 90ms to load 32 frames => better
# to read a batch of frames
class VideoFrameDset:
    def __init__(
        self, path, gpu_idx=0, data_reader_class=TorchAudioStreamReader, **kwargs
    ):
        self.reader = data_reader_class(
            path,
            frame_window_size=1,
            stride=1,
            gpu_idx=gpu_idx,
            size=None,
            mean=None,
            **kwargs,
        )

    def __getitem__(self, idx):
        return self.reader[idx]

    def __len__(self):
        return len(self.reader)
