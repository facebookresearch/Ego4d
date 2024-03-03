import math
import os
from collections import defaultdict
from typing import List, Tuple, Union

import h5py
import pandas as pd
import torch
from ego4d.research.clep.config import TrainConfig
from ego4d.research.dataset import LabelledFeatureDset
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


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


def _one_hot_encoding(n: int, clazzes: Union[int, List[int]]) -> torch.Tensor:
    result = torch.zeros(n)
    result[clazzes] = 1
    return result


def create_ego_charades_dset(
    config: TrainConfig, use_ego_sent: bool, ego_only: bool
) -> Tuple[
    Dataset,
    Tensor,
]:
    """
    Loads the ego charades dataset and returns a dataset and the associated
    sentence embeddings for each class (as a single tensor, where class_idx ==
    idx in the tensor).
    """
    val_df = pd.read_csv(config.pre_config.ego_charade.set_path)

    if ego_only is not None:
        if ego_only:
            val_df = val_df[val_df.egocentric == "Yes"]  # pyre-ignore
        else:
            val_df = val_df[val_df.egocentric == "No"]  # pyre-ignore

    val_df = val_df[~pd.isnull(val_df.actions)]

    sent = torch.load(
        os.path.join(
            config.pre_config.root_dir,
            config.pre_config.ego_charade.out_label_path,
        )
    )
    if use_ego_sent is not None:
        key = "sent_ego_fv" if use_ego_sent else "sent_non_ego_fv"
    else:
        key = "labels"

    sent_ordered = torch.stack([torch.tensor(fv) for fv in sent[key]])
    num_clazzes = len(sent_ordered)

    feature_hdf5_path = os.path.join(
        config.pre_config.root_dir,
        config.pre_config.ego_charade.out_path,
    )
    id_classes_pairs = []
    for row in val_df.itertuples():
        clazzes = []
        for x in row.actions.split(";"):
            clazzes.append(int(x.split(" ")[0].split("c")[1]))
        id_classes_pairs.append((row.id, _one_hot_encoding(num_clazzes, clazzes)))
    assert len(id_classes_pairs) == len(val_df)

    return LabelledFeatureDset(feature_hdf5_path, id_classes_pairs), sent_ordered


def create_kinetics_dset(
    config: TrainConfig,
) -> Tuple[
    Dataset,
    Tensor,
]:
    k400_config = config.pre_config.k400
    root = os.path.join(
        config.pre_config.root_dir,
        k400_config.root_dir,
    )

    sent_meta_path = os.path.join(root, k400_config.metadata_out_path)
    sent_features = torch.load(sent_meta_path)

    feature_hdf5_path = os.path.join(root, k400_config.viz_feature_path)
    label_name_to_idx = sent_features["label_name_to_idx"]
    sent_ordered = torch.stack([torch.tensor(fv) for fv in sent_features["label_fv"]])
    id_label_pairs = [
        (id, label_name_to_idx[label_name])
        for id, label_name in sent_features["labels"]
    ]
    return (
        LabelledFeatureDset(
            feature_hdf5_path, id_label_pairs, lambda x, _: x[0:].mean(0).squeeze()
        ),
        sent_ordered,
    )


class CCDset(Dataset):
    def __init__(
        self,
        config: TrainConfig,
    ):
        super().__init__()
        self.config = config
        self.viz_feature_path = os.path.join(
            config.pre_config.root_dir,
            config.pre_config.cc.hdf5_viz_path,
        )
        self.sent_feature_path = os.path.join(
            config.pre_config.root_dir,
            config.pre_config.cc.hdf5_sent_path,
        )
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


class Ego4DCLEP(Dataset):
    def __init__(
        self,
        config: TrainConfig,
    ):
        super().__init__()

        self.narr_meta_path = os.path.join(
            config.pre_config.root_dir, config.pre_config.ego4d_narr.metadata_out_path
        )
        self.narr_meta = torch.load(self.narr_meta_path)
        self.config = config
        self.narr_feature_dir = os.path.join(
            config.pre_config.root_dir, config.pre_config.ego4d_narr.narration_out_dir
        )
        self.features = h5py.File(
            os.path.join(
                config.pre_config.root_dir,
                config.pre_config.ego4d_features.hdf5_path,
            )
        )
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
            len(features),
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
