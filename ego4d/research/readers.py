import math
from typing import Optional

import av

import torch
from pytorchvideo.transforms import Normalize, ShortSideScale
from torchaudio.io import StreamReader
from torchvision.transforms._transforms_video import CenterCropVideo


def get_video_meta(path):
    with av.open(path) as cont:
        n_frames = cont.streams[0].frames
        codec = cont.streams[0].codec.name
        codec_long_name = cont.streams[0].codec.long_name
        tb = cont.streams[0].time_base

        all_pts = []
        for x in cont.demux(video=0):
            if x.pts is None:
                continue
            all_pts.append(x.pts)

        assert len(all_pts) == n_frames
        return {
            "all_pts": sorted(all_pts),
            "codec": codec,
            "codec_long_name": codec_long_name,
            "tb": tb,
        }


def _yuv_to_rgb(img: torch.Tensor) -> torch.Tensor:
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
    return rgb.permute(3, 0, 1, 2)


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
        resize: Optional[int],
        crop: Optional[int],
        mean: Optional[torch.Tensor],
        std: Optional[torch.Tensor],
        frame_window_size: int,
        stride: int,
        gpu_idx: int,
    ):
        super().__init__(path, stride, frame_window_size)

        self.mean = mean
        self.crop = crop
        self.std = std
        self.resize = resize
        self.resize_transform = (
            ShortSideScale(self.resize) if self.resize is not None else None
        )
        self.norm_transform = (
            Normalize(mean=self.mean, std=self.std) if self.mean is not None else None
        )
        self.crop_transform = (
            CenterCropVideo(self.crop) if self.crop is not None else None
        )
        self.create_underlying_cont(gpu_idx)

    def create_underlying_cont(self, gpu_id):
        self.gpu_id = gpu_id

        decoder_basename = self.meta["codec"]
        if self.gpu_id >= 0:
            # TODO(miguelmartin): this resize may not work as it is:
            # 1. not shortest side (can fix with -1xside knowing w/h of video)
            # 2. unsure about the interpolation
            decoder_opt = (
                # TODO(miguelmartin): this resize is not the shortest side scale
                {"resize": f"{self.resize}x{self.resize}", "gpu": f"{gpu_id}"}
                if self.resize is not None
                else {"gpu": f"{gpu_id}"}
            )
            self.conf = {
                "decoder": f"{decoder_basename}_cuvid",
                "hw_accel": f"cuda:{gpu_id}",
                "decoder_option": decoder_opt,
                "stream_index": 0,
            }
            self.resize_transform = None
        else:
            self.conf = {
                "decoder": decoder_basename,
                "stream_index": 0,
            }
            self.resize_transform = (
                ShortSideScale(self.resize) if self.resize is not None else None
            )

        self.cont = StreamReader(self.path)
        self.cont.add_video_stream(self.frame_window_size, **self.conf)
        self.cont.fill_buffer()

    def __getitem__(self, idx: int) -> dict:
        frame_i = self.stride * idx
        frame_j = frame_i + self.frame_window_size - 1
        assert frame_i >= 0 and frame_j < len(self.all_pts)

        frame_i_pts = self.all_pts[frame_i]
        self.cont.seek(float(frame_i_pts * self.meta["tb"]))
        fs = None
        for fs in self.cont.stream():
            break

        assert fs is not None
        assert len(fs) == 1
        ret = _yuv_to_rgb(fs[0])  # TODO: optimize me
        assert ret.shape[1] == self.frame_window_size
        if self.resize_transform is not None:
            ret = self.resize_transform(ret)
        if self.crop_transform is not None:
            ret = self.crop_transform(ret)
        if self.norm_transform is not None:
            ret = self.norm_transform(ret)
        return {
            "video": ret,
            "frame_start_idx": frame_i,
            "frame_end_idx": frame_j,
        }


class PyAvReader(StridedReader):
    def __init__(
        self,
        path: str,
        resize: Optional[int],
        crop: Optional[int],
        mean: Optional[torch.Tensor],
        std: Optional[torch.Tensor],
        frame_window_size: int,
        stride: int,
        gpu_idx: int,
    ):
        super().__init__(path, stride, frame_window_size)

        if gpu_idx >= 0:
            print("WARN: GPU decoding not supported for PyAV, using CPU")

        self.mean = mean
        self.crop = crop
        self.std = std
        self.resize = resize
        self.resize_transform = (
            ShortSideScale(self.resize) if self.resize is not None else None
        )
        self.norm_transform = (
            Normalize(mean=self.mean, std=self.std) if self.mean is not None else None
        )
        self.crop_transform = (
            CenterCropVideo(self.crop) if self.crop is not None else None
        )
        self.path = path
        self.create_underlying_cont(gpu_idx)

    def create_underlying_cont(self, _):
        self.cont = av.open(self.path)

    def __getitem__(self, idx: int) -> dict:
        frame_i = self.stride * idx
        frame_j = frame_i + self.frame_window_size - 1
        assert frame_i >= 0 and frame_j < len(self.all_pts)

        frame_i_pts = self.all_pts[frame_i]
        frame_j_pts = self.all_pts[frame_j]
        self.cont.seek(frame_i_pts, stream=self.cont.streams.video[0])
        fs = []
        for f in self.cont.decode(video=0):
            if f.pts < frame_i_pts:
                continue
            if f.pts >= frame_j_pts:
                break
            fs.append(
                (
                    f.pts,
                    torch.tensor(f.to_ndarray(format="rgb24"), dtype=torch.float32)
                    / 255,
                )
            )
        fs = sorted(fs, key=lambda x: x[0])
        ret = torch.stack([x[1] for x in fs]).permute(3, 0, 1, 2)
        if self.resize_transform is not None:
            ret = self.resize_transform(ret)
        if self.crop_transform is not None:
            ret = self.crop_transform(ret)
        if self.norm_transform is not None:
            ret = self.norm_transform(ret)
        return {
            "video": ret,
            "frame_start_idx": frame_i,
            "frame_end_idx": frame_j,
        }
