import os
from typing import Any, Callable, List, Optional, Tuple

import h5py
import torch

from tqdm.auto import tqdm


class LabelledFeatureDset(torch.utils.data.Dataset):
    """
    A simple utility class to load features associated with labels. The input this
    method requires is as follows:
        1. `feature_hdf5_path`: the features transposed to a HDF5 file.
            See `save_ego4d_features_to_hdf5`
        2. `uid_label_pairs` a list of (uid, label). `label` can be anything
            `uid` is a unique id associated to the `feature_hdf5_path` file.
        3. `aggr_function` a function to aggregate based off given label
    """

    def __init__(
        self,
        feature_hdf5_path: str,
        uid_label_pairs: List[Tuple[str, Any]],
        aggr_function: Optional[Callable[[torch.Tensor, Any], torch.Tensor]] = None,
    ):
        self.uid_label_pairs = uid_label_pairs
        self.features = h5py.File(feature_hdf5_path)
        self.aggr_function = (
            aggr_function
            if aggr_function is not None
            else lambda x, _: torch.tensor(x[0:]).squeeze()
        )

    def __len__(self):
        return len(self.uid_label_pairs)

    def __getitem__(self, idx: int):
        uid, label = self.uid_label_pairs[idx]
        feat = self.aggr_function(self.features[uid], label)
        return feat, label


def save_ego4d_features_to_hdf5(video_uids: List[str], feature_dir: str, out_path: str):
    """
    Use this function to preprocess Ego4D features into a HDF5 file with h5py
    """
    with h5py.File(out_path, "w") as out_f:
        for uid in tqdm(video_uids, desc="video_uid", leave=True):
            feature_path = os.path.join(feature_dir, f"{uid}.pt")
            fv = torch.load(feature_path)
            out_f.create_dataset(uid, data=fv.numpy())
