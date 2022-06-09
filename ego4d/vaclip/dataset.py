import time
import json
import os
import math
from typing import List, Tuple
from collections import OrderedDict
from pathlib import Path

import h5py
import torch
import numpy as np
import pandas as pd
from pytorchvideo.data.encoded_video import EncodedVideo
from torch.utils.data import DataLoader
from ego4d.features.dataset import CropIfStereo
from ego4d.features.models.omnivore import get_transform as omnivore_transform
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)

from collections import defaultdict
from ego4d.vaclip.config import TrainConfig, InputConfig


# TODO: move to utils
def get_start_end_idx(t1, t2, feature_per_sec, nf):
    assert t2 >= 0
    x1 = min(
        max(0, int(np.round(t1 * feature_per_sec))),
        nf - 1,
    )
    x2 = min(
        math.ceil(np.round(t2 * feature_per_sec)),
        nf - 1,
    )
    assert x2 >= x1
    return x1, x2 + 1


class KineticsDset(torch.utils.data.Dataset):
    def __init__(self, config: TrainConfig):
        super().__init__()
        self.config = config

        root = os.path.join(config.k400_pre_config.pre_root_dir, config.k400_pre_config.set_to_use)
        viz_dir = os.path.join(root, config.k400_pre_config.viz_feature_dir)

        sent_meta_path = os.path.join(root, config.k400_pre_config.metadata_out_path)
        self.sent_features = torch.load(sent_meta_path)

        self.videos = [
            os.path.join(viz_dir, data_path)
            for data_path in os.listdir(viz_dir)
            if data_path.endswith(".pt")
        ]
        self.label_name_to_idx = self.sent_features["label_name_to_idx"]
        self.sent_ordered = torch.stack([torch.tensor(fv) for fv in self.sent_features["label_fv"]])

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        vf = torch.load(self.videos[idx])
        label_name = vf["label"]
        feat = vf["feature"]
        return feat, self.label_name_to_idx[label_name]


class Ego4DVaClip(torch.utils.data.Dataset):
    def __init__(
        self,
        config: TrainConfig,
    ):
        super().__init__()

        self.narr_meta_path = os.path.join(
            config.ego_pre_config.pre_root_dir,
            config.ego_pre_config.metadata_out_path
        )
        self.narr_meta = torch.load(self.narr_meta_path)
        self.config = config
        self.narr_feature_dir = os.path.join(
            config.ego_pre_config.pre_root_dir,
            config.ego_pre_config.narration_out_path
        )
        self.features = h5py.File(config.ego_pre_feature_config.hdf5_path)
        uids = set(meta["uid"] for meta in self.narr_meta)
        assert len(uids - set(self.features.keys())) == 0, "not all features cached"
        self.narr_meta = [meta for meta in self.narr_meta if meta["uid"] in uids]

    def __len__(self):
        return len(self.narr_meta)

    def __getitem__(self, idx):
        meta = self.narr_meta[idx]
        uid = meta["uid"]
        ts = meta["ts"]
        narr_idx = meta["idx"]

        # get txt feature
        txt_feature_path = os.path.join(self.narr_feature_dir, uid, f"{narr_idx}.pt")
        txt_feat = torch.load(txt_feature_path)

        offset = self.config.input_config.narration_width_sample_sec
        t1 = ts - offset
        t2 = ts + offset

        features = self.features[uid]
        start_idx, end_idx = get_start_end_idx(
            t1,
            t2,
            self.config.input_config.features_per_second,
            len(features)
        )
        features = features[start_idx:end_idx]
        v_feat = features.mean(0)  # aggregate

        return {
            "video": v_feat,
            "text": txt_feat,
            "raw_text": meta["post_txt"],
        }


def create_data_loader(dset, config: TrainConfig):
    return DataLoader(
        dset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        shuffle=True,
    )
