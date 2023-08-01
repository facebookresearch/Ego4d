import json
import os
from typing import Any, Dict, List, Optional

import cv2
import pandas as pd

from iopath.common.file_io import PathManager
from iopath.common.s3 import S3PathHandler

pathmgr = PathManager()
pathmgr.register_handler(S3PathHandler(profile="default"))


def _get_synced_timesync_df(timesync_df):
    ks = [k for k in timesync_df.keys() if "_global_time" in k]
    start_indices = [timesync_df[k].first_valid_index() for k in ks]
    last_indices = [timesync_df[k].last_valid_index() for k in ks]
    first_idx = max(start_indices)
    last_idx = min(last_indices)
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
        self, data_dir: str, dataset_json_path: str, read_frames: bool, legacy=False
    ):
        self.dataset_json = json.load(open(dataset_json_path))
        self.read_frames = read_frames
        self.root_dir = data_dir
        if legacy:
            self.frame_dir = os.path.join(
                self.root_dir, self.dataset_json["dataset_dir"], "frames"
            )
        else:
            self.frame_dir = os.path.join(self.root_dir, self.dataset_json["frame_dir"])

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
