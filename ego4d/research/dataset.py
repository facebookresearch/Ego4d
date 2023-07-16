import os
import math
from typing import Any, Callable, List, Optional, Tuple

from tqdm.auto import tqdm

import h5py
import torch
import av
from torchaudio.io import StreamReader
from torchvision.transforms import Resize



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

def get_video_meta(path):
    with av.open(path) as cont:
        n_frames = cont.streams[0].frames
        codec = cont.streams[0].codec.name
        tb = cont.streams[0].time_base

        all_pts = []
        for x in cont.demux(video=0):
            if x.pts is None:
                continue
            all_pts.append(x.pts)

            if len(all_pts) >= 2:
                assert all_pts[-1] > all_pts[-2]

        assert len(all_pts) == n_frames
        return {
            "all_pts": all_pts,
            "codec": codec,
            "tb": tb,
        }


def _yuv_to_rgb(img):
    img = img.to(torch.float)
    y = img[..., 0, :, :]
    u = img[..., 1, :, :]
    v = img[..., 2, :, :]

    y /= 255
    u = u / 255 - 0.5
    v = v / 255 - 0.5

    r = y + 1.14 * v
    g = y + -0.396 * u - 0.581 * v
    b = y + 2.029 * u

    rgb = torch.stack([r, g, b], -1)
    rgb = (rgb * 255).clamp(0, 255).to(torch.uint8)
    return rgb

class StridedReader:
    def __init__(self, path, stride, frame_window_size):
        self.path = path
        self.meta = get_video_meta(path)
        self.all_pts = self.meta["all_pts"]
        self.stride = stride
        self.frame_window_size = frame_window_size
        if self.stride == 0:
            self.stride = self.frame_window_size

    def __getitem__(self, idx: int) -> torch.Tensor:
        raise AssertionError("Not implemented")
    
    def __len__(self):
        return int(math.ceil((len(self.all_pts) - self.frame_window_size) / self.stride))



class TorchAudioStreamReader(StridedReader):
    def __init__(self,
        path: str,
        resize: Optional[int],
        mean: Optional[torch.Tensor],
        frame_window_size: int,
        stride: int,
        gpu_idx: int,
    ):
        super().__init__(path, stride, frame_window_size)

        self.mean = mean
        self.size = resize
        self.create_underlying_cont(gpu_idx)
    
    def create_underlying_cont(self, gpu_id):
        self.gpu_id = gpu_id

        decoder_basename = self.meta["codec"] 
        if self.gpu_id >= 0:
            # NOTE: may need to change this as algorithm to downscale matters
            decoder_opt = (
                {"resize": f"{self.size}x{self.size}"}
                if size is not None 
                else {}
            )
            self.conf = {
                "decoder": f"{decoder_basename}_cuvid",
                "hw_accel": f"cuda:{gpu_id}",
                "decoder_option": decoder_opt,
                "stream_index": 0,
            }
            self.resize = None
        else:
            self.conf = {
                "decoder": decoder_basename,
                "stream_index": 0,
            }
            self.resize = Resize((self.size, self.size)) if self.size is not None else None

        self.cont = StreamReader(self.path)
        self.cont.add_video_stream(self.frame_window_size, **self.conf)

    def get(self, idx: int) -> torch.Tensor:
        frame_i = self.stride * idx
        frame_j = frame_i + self.frame_window_size
        assert frame_i >= 0 and frame_j < len(self.all_pts)

        frame_i_pts = self.all_pts[frame_i]
        self.cont.seek(float(frame_i_pts * self.meta["tb"]))
        fs = None
        for fs in self.cont.stream():
            break

        assert fs is not None
        assert len(fs) == 1
        # ret = _yuv_to_rgb(fs[0])  # TODO: fixme
        ret = fs[0]
        assert ret.shape[0] == self.frame_window_size
        if self.resize is not None:
            ret = self.resize(ret)
        if self.mean is not None:
            ret -= self.mean
        return ret


def save_ego4d_features_to_hdf5(video_uids: List[str], feature_dir: str, out_path: str):
    """
    Use this function to preprocess Ego4D features into a HDF5 file with h5py
    """
    with h5py.File(out_path, "w") as out_f:
        for uid in tqdm(video_uids, desc="video_uid", leave=True):
            feature_path = os.path.join(feature_dir, f"{uid}.pt")
            fv = torch.load(feature_path)
            out_f.create_dataset(uid, data=fv.numpy())
