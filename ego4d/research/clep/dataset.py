import random
import time
import json
import os
import math
from typing import List, Tuple, Optional
from collections import OrderedDict
from pathlib import Path

import h5py
import torch
import numpy as np
import pandas as pd
from multiprocessing import Pool
from tqdm.auto import tqdm
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
from ego4d.research.clep.config import TrainConfig, InputConfig


def get_start_end_idx(t1, t2, feature_per_sec, nf):
    assert t2 >= 0
    x1 = min(
        max(0, math.floor(t1 * feature_per_sec)),
        nf - 1,
    )
    x2 = min(
        math.floor(t2 * feature_per_sec),
        nf - 1,
    )
    assert x2 >= x1
    return x1, x2 + 1


# TODO: convert to hdf5
class EgoCharadesDset(torch.utils.data.Dataset):
    def __init__(self,
        config: TrainConfig,
        use_ego_sent: Optional[bool],
        ego_only: Optional[bool],
    ):
        super().__init__()

        self.ego_only = ego_only
        val_df = pd.read_csv(config.pre_config.ego_charade.set_path)

        if self.ego_only is not None:
            if self.ego_only:
                val_df = val_df[val_df.egocentric == "Yes"]  # pyre-ignore
            else:
                val_df = val_df[val_df.egocentric == "No"]  # pyre-ignore

        val_df = val_df[~pd.isnull(val_df.actions)]

        data_path = config.pre_config.ego_charade.out_path
        self.data = h5py.File(data_path)
        self.id_classes_pairs = []
        for row in val_df.itertuples():
            clazzes = []
            for x in row.actions.split(";"):
                clazzes.append(int(x.split(" ")[0].split("c")[1]))
            self.id_classes_pairs.append((row.id, clazzes))

        assert len(self.id_classes_pairs) == len(val_df)
        self.sent = torch.load(config.pre_config.ego_charade.out_label_path)
        if use_ego_sent is not None:
            key = "sent_ego_fv" if use_ego_sent else "sent_non_ego_fv"
        else:
            key = "labels"
        self.sent_ordered = torch.stack([torch.tensor(fv) for fv in self.sent[key]])
        self.num_clazzes = len(self.sent_ordered)

    def __len__(self):
        return len(self.id_classes_pairs)

    def __getitem__(self, idx):
        uid, classes = self.id_classes_pairs[idx]
        feat = torch.tensor(self.data[uid][0:])
        gt = torch.zeros(self.num_clazzes)
        gt[classes] = 1
        return feat, gt


class KineticsDset(torch.utils.data.Dataset):
    def __init__(self, config: TrainConfig):
        super().__init__()
        self.config = config

        k400_config = config.pre_config.k400
        root = os.path.join(k400_config.pre_root_dir, k400_config.set_to_use)

        sent_meta_path = os.path.join(root, k400_config.metadata_out_path)
        self.sent_features = torch.load(sent_meta_path)

        viz_path = os.path.join(k400_config.pre_root_dir, k400_config.viz_feature_path)
        self.videos = h5py.File(viz_path)
        self.label_name_to_idx = self.sent_features["label_name_to_idx"]
        self.sent_ordered = torch.stack([
            torch.tensor(fv)
            for fv in self.sent_features["label_fv"]
        ])
        self.labels = self.sent_features["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        path, label_name = self.labels[idx]
        feat = self.videos[path][0:].mean(0)
        return feat, self.label_name_to_idx[label_name]


class CCDset(torch.utils.data.Dataset):
    def __init__(
        self,
        config: TrainConfig,
    ):
        super().__init__()
        self.config = config
        self.viz_feature_path = config.pre_config.cc.hdf5_viz_path
        self.sent_feature_path = config.pre_config.cc.hdf5_sent_path
        self.viz_dset = h5py.File(self.viz_feature_path)
        self.sent_dset = h5py.File(self.sent_feature_path)

        self.keys = torch.load(config.pre_config.cc.meta_path)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        k = self.keys[idx]
        v = self.viz_dset[k][0:]
        s = self.sent_dset[k][0:]
        return {
            "video": v,
            "text": s,
            "text_no_tag": s,
        }


class Ego4DVaClip(torch.utils.data.Dataset):
    def __init__(
        self,
        config: TrainConfig,
    ):
        super().__init__()

        self.narr_meta_path = os.path.join(
            config.pre_config.ego4d_narr.pre_root_dir,
            config.pre_config.ego4d_narr.metadata_out_path
        )
        self.narr_meta = torch.load(self.narr_meta_path)
        self.config = config
        self.narr_feature_dir = os.path.join(
            config.pre_config.ego4d_narr.pre_root_dir,
            config.pre_config.ego4d_narr.narration_out_path
        )
        self.features = h5py.File(config.pre_config.ego4d_features.hdf5_path)
        uids = set(meta["uid"] for meta in self.narr_meta)
        assert len(uids - set(self.features.keys())) == 0, "not all features cached"
        self.narr_meta = [meta for meta in self.narr_meta if meta["uid"] in uids]

        t_by_uid = defaultdict(list)
        for x in self.narr_meta:
            t_by_uid[x["uid"]].append(x["ts"])

        self.betas = {
            uid: torch.mean(torch.tensor(v)[1:] - torch.tensor(v)[0:-1])
            for uid, v in t_by_uid.items()
        }
        self.alpha = torch.mean(torch.stack(list(self.betas.values())))

        old_len = len(self.narr_meta)
        self.narr_meta = [x for x in self.narr_meta if self.betas[x["uid"]] >= 1e-1]
        # random.shuffle(self.narr_meta)
        # self.narr_meta = self.narr_meta[0:150000]
        print(f"{old_len} -> {len(self.narr_meta)}")


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
        # offset = self.betas[uid] / (2*self.alpha)
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
            "text": txt_feat["fv"],
            "text_no_tag": txt_feat["fv_no_tag"],
        }


def create_data_loader(dset, config: TrainConfig, shuffle=True):
    return DataLoader(
        dset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        shuffle=shuffle,
    )