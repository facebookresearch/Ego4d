import time
import json
import os
import math
from typing import List, Tuple
from collections import OrderedDict
from pathlib import Path

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


class LRUCache:
    def __init__(self, max_size=5):
        self.max_size = max_size

        # OrderedDict is a FILO container
        self.cache = dict()
        self.cache_keys = OrderedDict()

    def isin(self, key):
        return key in self.cache_keys
    
    def get(self, key):
        self.refresh_item(key)
        return self.cache[key]

    def refresh_item(self, key):
        self.cache_keys.move_to_end(key)

    def put(self, key, bs):
        self.cache[key] = bs
        self.cache_keys[key] = key
        if len(self.cache) > self.max_size:
            key = self.cache_keys.popitem(last=False)[0]
            del self.cache[key]


class FeatureRetrieval:
    def __init__(
        self,
        feature_idx_paths: List[Tuple[int, str]],
        feature_per_sec: float,
        num_features: int,
        partition_size: int,
    ):
        feature_idx_paths.sort(key=lambda x: x[0])
        self.feature_paths = [path for _, path in feature_idx_paths]
        self.feature_per_sec = feature_per_sec
        self.partition_size = partition_size
        self.num_features = num_features

    def _get_features(self, idx1, idx2):
        # [idx1, idx2] inclusive
        # == features[x1:x2+1]
        nf = idx2 - idx1 + 1
        ret = []
        idx = idx1
        # print("nf=", nf)
        # breakpoint()
        while idx <= idx2:
            bucket = idx // self.partition_size
            start_idx = idx - bucket * self.partition_size
            end_idx = start_idx + min(self.partition_size - start_idx, idx2 - idx + 1)
            assert start_idx < self.partition_size and start_idx >= 0
            assert end_idx <= self.partition_size
            assert end_idx > start_idx

            f = torch.load(self.feature_paths[bucket]) # , map_location="cuda")  # TODO
            f = f[start_idx:end_idx]
            idx += f.shape[0]
            ret.append(f)

        # breakpoint()
        # t1 = time.time()
        ret = torch.cat(ret, dim=0)
        # t2 = time.time()
        # print("Took", t2 - t1, "time to cat", flush=True)
        assert ret.shape[0] == nf
        return ret

    def get_clip(self, t1, t2):
        assert t2 >= 0
        x1 = min(
            max(0, int(np.round(t1 * self.feature_per_sec))),
            self.num_features - 1,
        )
        x2 = min(
            math.ceil(np.round(t2 * self.feature_per_sec)),
            self.num_features - 1,
        )
        assert x2 >= x1
        return self._get_features(x1, x2)


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

        # self.vid_features = LRUCache(max_size=config.input_config.max_num_feature_vec_video_uids)
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

        uids = set(meta["uid"] for meta in self.narr_meta)

        features_meta_path = os.path.join(config.ego_pre_feature_config.pre_root_dir, config.ego_pre_feature_config.meta_path)
        features_meta = torch.load(features_meta_path)

        # TODO
        # assert len(uids - set(features_meta.keys())) == 0, "not all features partitioned"
        uids = uids & set(features_meta.keys())
        self.narr_meta = [meta for meta in self.narr_meta if meta["uid"] in uids]
        self.features = {
            uid: FeatureRetrieval(
                feature_idx_paths=features_meta[uid]["idx_path_pairs"],
                feature_per_sec=self.config.input_config.features_per_second,
                num_features=features_meta[uid]["num_features"],
                partition_size=self.config.ego_pre_feature_config.num_features_per_file,
                # device=self.config.accelerator,
            )
            for uid in uids
        }
        # breakpoint()
        # import IPython; IPython.embed()
        # f = self.features["db9463d4-b1db-4e4b-b24b-60c7e6116664"]
        # ff = f.get_clip(1000.2, 2000.2)
        # breakpoint()

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

        # offset = (torch.rand(1) * self.config.input_config.narration_width_sample_sec).item()
        offset = self.config.input_config.narration_width_sample_sec
        t1 = ts - offset
        t2 = ts + offset

        # path = os.path.join(self.config.input_config.feature_path, f"{uid}.pt")
        # if not self.vid_features.isin(uid):
        #     path = os.path.join(self.config.input_config.feature_path, f"{uid}.pt")
        #     feature_ret = FeatureRetrieval(path, self.config.input_config.features_per_second)
        #     self.vid_features.put(uid, feature_ret)
        # features = self.vid_features.get(uid).get_clip(t1, t2)

        features = self.features[uid].get_clip(t1, t2)
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
