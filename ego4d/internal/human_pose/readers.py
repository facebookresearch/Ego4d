# TODO: moveme to ego4d/research/data
import math
from typing import Optional

import av
import torch
import torchaudio
from torchaudio.io import StreamReader
from torchvision.transforms import Resize


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

    def __getitem__(self, _: int) -> torch.Tensor:
        raise AssertionError("Not implemented")

    def __len__(self):
        return int(
            math.ceil((len(self.all_pts) - self.frame_window_size) / self.stride)
        )


class TorchAudioStreamReader(StridedReader):
    def __init__(
        self,
        path: str,
        size: Optional[int],
        mean: Optional[torch.Tensor],
        frame_window_size: int,
        stride: int,
        gpu_idx: int,
    ):
        super().__init__(path, stride, frame_window_size)

        self.mean = mean
        self.size = size
        self.gpu_id = gpu_idx

        decoder_basename = self.meta["codec"]
        if self.gpu_id >= 0:
            print("gpu_idx=", self.gpu_id)
            decoder_opt = {"resize": f"{size}x{size}"} if size is not None else {}
            self.conf = {
                "decoder": f"{decoder_basename}_cuvid",
                "hw_accel": f"cuda:{self.gpu_id}",
                "decoder_option": decoder_opt,
                "stream_index": 0,
            }
            self.resize = None
        else:
            self.conf = {
                "decoder": decoder_basename,
                "stream_index": 0,
            }
            self.resize = Resize((size, size)) if size is not None else None

        self.cont = StreamReader(self.path)
        self.cont.add_video_stream(self.frame_window_size, **self.conf)

    def __getitem__(self, idx: int) -> torch.Tensor:
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
        # print([x.shape for x in fs])
        ret = _yuv_to_rgb(fs[0])
        #         ret = fs[0]
        assert ret.shape[0] == self.frame_window_size
        if self.resize is not None:
            ret = self.resize(ret)
        if self.mean is not None:
            ret -= self.mean
        return ret


def read_frame_idx_set(path, frame_indices, stream_id):
    meta = get_video_meta(path)
    with av.open(path) as cont:
        initial_pts = meta["all_pts"][frame_indices[0]]
        last_pts = meta["all_pts"][frame_indices[-1]]
        pts_to_idx = {meta["all_pts"][idx]: idx for idx in frame_indices}
        cont.seek(initial_pts, stream=cont.streams.video[stream_id], any_frame=False)
        seen = 0
        for f in cont.decode(video=stream_id):
            if f.pts > last_pts:
                break
            if f.pts not in pts_to_idx:
                # print("Skipping", f.pts)
                continue

            idx = pts_to_idx[f.pts]
            seen += 1
            yield idx, f.to_ndarray(format="rgb24")

        assert seen == len(pts_to_idx)


class PyAvReader(StridedReader):
    def __init__(
        self,
        path: str,
        size: Optional[int],
        mean: Optional[torch.Tensor],
        frame_window_size: int,
        stride: int,
        gpu_idx: int,
        to_tensor: bool = True,
        video_stream_id: int = 0,
    ):
        super().__init__(path, stride, frame_window_size)

        if gpu_idx >= 0:
            raise AssertionError("GPU decoding not support for pyav")

        self.cont = av.open(path)
        self.mean = mean
        self.resize = Resize((size, size)) if size is not None else None
        self.to_tensor = to_tensor
        self.video_stream_id = video_stream_id
        # self.cont.seek(0, stream=self.cont.streams.video[self.video_stream_id], any_frame=False)

    def __getitem__(self, idx: int):
        frame_i = self.stride * idx
        frame_j = frame_i + self.frame_window_size
        assert frame_i >= 0 and frame_j < len(self.all_pts)

        frame_i_pts = self.all_pts[frame_i]
        frame_j_pts = self.all_pts[frame_j]
        frame_pts = set(self.all_pts[frame_i : frame_j + 1])
        self.cont.seek(
            frame_i_pts,
            stream=self.cont.streams.video[self.video_stream_id],
            any_frame=False,
        )
        # self.cont.seek(frame_i_pts, stream=self.cont.streams.video[self.video_stream_id], any_frame=True)
        fs = []
        for f in self.cont.decode(video=self.video_stream_id):
            if f.pts > frame_j_pts:
                break
            if f.pts not in frame_pts:
                # print("Skipping", f.pts)
                continue
            assert f.pts in frame_pts, f"{f.pts}"
            if self.to_tensor:
                fs.append(torch.tensor(f.to_ndarray(format="rgb24")))
            else:
                fs.append(f.to_ndarray(format="rgb24"))

        assert len(fs) == (frame_j - frame_i + 1)
        if self.to_tensor:
            ret = torch.stack(fs)
            if self.resize is not None:
                ret = self.resize(ret)
            if self.mean is not None:
                ret -= self.mean
        else:
            assert (
                self.resize is None and self.mean is None
            ), "not supported in non tensor mode"
            ret = fs
        return ret
