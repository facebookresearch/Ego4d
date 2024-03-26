import bisect
import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import h5py
import torch
from ego4d.research.readers import PyAvReader, StridedReader, TorchAudioStreamReader

from tqdm import tqdm


class LabelledFeatureDset(torch.utils.data.Dataset):
    """
    A simple utility class to load features associated with labels. The input this
    method requires is as follows:
        1. `feature_hdf5_path`: the features transposed to a HDF5 file.
            See `save_features_to_hdf5`
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
        self.uid_label_pairs = uid_label_pairs
        f_keys = set(self.features.keys())
        l_keys = {uid for uid, _ in self.uid_label_pairs}
        if len(l_keys - f_keys) > 0:
            print(
                f"WARN: missing {len(l_keys - f_keys)} keys in feature hdf5 path: {feature_hdf5_path}"
            )
            self.uid_label_pairs = [
                (uid, label) for uid, label in self.uid_label_pairs if uid in f_keys
            ]

    def __len__(self):
        return len(self.uid_label_pairs)

    def __getitem__(self, idx: int):
        uid, label = self.uid_label_pairs[idx]
        feat = self.aggr_function(self.features[uid], label)
        return feat, label


def save_features_to_hdf5(uids: List[str], feature_dir: str, out_path: str):
    """
    Use this function to preprocess Ego4D features into a HDF5 file with h5py
    """
    with h5py.File(out_path, "w") as out_f:
        for uid in tqdm(uids, leave=True):
            feature_path = os.path.join(feature_dir, f"{uid}.pt")
            fv = torch.load(feature_path)
            out_f.create_dataset(uid, data=fv.numpy())


# Related:
# https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#ConcatDataset
class VideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        paths: List[str],
        video_class: type,
        video_class_kwargs: Dict[str, Any],
        max_num_frames_per_video: Optional[int] = None,
        paths_to_n_frames: Optional[Dict[str, int]] = None,
        with_pbar: bool = False,
        transform_fn=None,
        labels_fn=None,
    ):
        paths = sorted(paths)
        self.video_class = video_class
        self.video_class_kwargs = video_class_kwargs
        self.paths = paths
        self.labels_fn = labels_fn
        self.transform_fn = transform_fn
        if paths_to_n_frames is None:
            print("Creating containers")
            path_iter = paths
            if with_pbar:
                path_iter = tqdm(paths)
            self.conts = {
                idx: (p, video_class(p, **video_class_kwargs))
                for idx, p in enumerate(path_iter)
            }
            print("Created containers")

            cont_iter = list(self.conts.values())
            if with_pbar:
                cont_iter = tqdm(cont_iter)

            self.fs_cumsum = [
                (
                    min(len(ct), max_num_frames_per_video)
                    if max_num_frames_per_video
                    else len(ct)
                )
                for _, ct in cont_iter
            ]
            self.fs_cumsum = [0] + torch.cumsum(
                torch.tensor(self.fs_cumsum), dim=0
            ).tolist()
        else:
            self.fs_cumsum = [
                int(
                    math.ceil(
                        (
                            paths_to_n_frames[path]
                            - video_class_kwargs["frame_window_size"]
                        )
                        / video_class_kwargs.get(
                            "stride", video_class_kwargs["frame_window_size"]
                        )
                    )
                )
                for path in paths
            ]
            self.fs_cumsum = [0] + torch.cumsum(
                torch.tensor(self.fs_cumsum), dim=0
            ).tolist()
            self.conts = {idx: (p, None) for idx, p in enumerate(paths)}

    def create_underlying_cont(self, gpu_id):
        # TODO clear_cuda_context_cache()
        self.gpu_id = gpu_id
        for _, c in self.conts.values():
            if c is not None:
                c.create_underlying_cont(self.gpu_id)

    def __getitem__(self, i):
        cont_idx = bisect.bisect_left(self.fs_cumsum, i + 1) - 1
        p, c = self.conts[cont_idx]
        if c is None:
            self.conts[cont_idx] = (p, self.video_class(p, **self.video_class_kwargs))
            # c.create_underlying_cont(self.gpu_id)
        _, c = self.conts[cont_idx]
        assert c is not None
        idx = i - self.fs_cumsum[cont_idx]
        ret = c[idx]

        labels = None
        if self.labels_fn:
            labels = self.labels_fn(
                p,
                ret["frame_start_idx"],
                ret["frame_end_idx"],
            )
            ret.update(labels)

        if self.transform_fn is not None:
            ret = self.transform_fn(ret)
        return ret

    def __len__(self):
        return self.fs_cumsum[-1]
