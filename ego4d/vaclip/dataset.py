import json
import os
from typing import List
from collections import OrderedDict

import torch
import numpy as np
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
    def __init__(self, feature_path: str, feature_per_sec: float):
        self.feature_per_sec = feature_per_sec
        self.features = torch.load(feature_path)

    def get_clip(self, t1, t2):
        # TODO: checkme
        x1 = max(0, int(np.round(t1 * self.feature_per_sec)))
        x2 = int(np.round(t2 * self.feature_per_sec))
        # if both are in the last feature bucket
        if x2 >= self.features.shape[0] - 1 and x1 >= self.features.shape[0] - 1:
            x2 = self.features.shape[0]
            x1 = x2 - 1
        elif x2 >= self.features.shape[0] - 1:
            x2 = self.features.shape[0] - 1
        return self.features[x1:x2+1]


class KineticsDset(torch.utils.data.Dataset):
    def __init__(self, config: TrainConfig):
        super().__init__()
        self.config = config

        root = os.path.join(config.k400_pre_config.pre_root_dir, config.k400_pre_config.set_to_use)
        sent_meta_path = os.path.join(root, config.k400_pre_config.metadata_out_path)
        viz_dir = os.path.join(root, config.k400_pre_config.viz_feature_dir)
        self.viz_features = [
            (label_pt.split(".pt")[0], row["feature"], row["video_path"])
            for label_pt in os.listdir(viz_dir)
            for row in torch.load(os.path.join(viz_dir, label_pt))
        ]
        self.sent_features = torch.load(sent_meta_path)
        self.sent_ordered = torch.stack([torch.tensor(feat) for feat in self.sent_features["label_fv"]])
        self.label_name_to_idx = {
            label_name: idx
            for idx, label_name in enumerate(self.sent_features["label_dirs"])
        }

    def __len__(self):
        return len(self.viz_features)

    def __getitem__(self, idx):
        label_name, feat, _ = self.viz_features[idx]
        # one_hot = torch.zeros(len(self.sent_ordered))
        # one_hot[self.label_name_to_idx[label_name]] = 1.0
        return feat, self.label_name_to_idx[label_name]


class Ego4DVaClip(torch.utils.data.Dataset):
    def __init__(
        self,
        config: TrainConfig,
    ):
        super().__init__()

        self.vid_features = LRUCache(max_size=config.input_config.max_num_feature_vec_video_uids)
        self.narr_meta_path = os.path.join(config.ego_pre_config.pre_root_dir, config.ego_pre_config.metadata_out_path)
        self.narr_meta = torch.load(self.narr_meta_path)
        self.config = config
        self.narr_feature_dir = os.path.join(config.ego_pre_config.pre_root_dir, config.ego_pre_config.narration_out_path)

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

        offset = (torch.rand(1) * self.config.input_config.narration_width_sample_sec).item()
        t1 = ts - offset
        t2 = ts + offset
        if not self.vid_features.isin(uid):
            path = os.path.join(self.config.input_config.feature_path, f"{uid}.pt")
            feature_ret = FeatureRetrieval(path, self.config.input_config.features_per_second)
            self.vid_features.put(uid, feature_ret)

        features = self.vid_features.get(uid).get_clip(t1, t2)
        v_feat = features.mean(0)  # aggregate

        return {
            "video": v_feat,
            "text": txt_feat,
        }


def create_data_loader(dset, config: TrainConfig):
    return DataLoader(
        dset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        shuffle=True,
    )
