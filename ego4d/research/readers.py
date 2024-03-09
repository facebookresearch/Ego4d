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
        resize: Optional[int],
        crop: Optional[int],
        mean: Optional[torch.Tensor],
        std: Optional[torch.Tensor],
        frame_window_size: int,
        stride: int,
        gpu_idx: int,
        norm_pixel_scale: bool = True,
        channel_first: bool = False,
    ):
        super().__init__(path, stride, frame_window_size)

        self.mean = mean
        self.crop = crop
        self.std = std
        self.norm_pixel_scale = norm_pixel_scale
        self.pixel_scale = 1.0 if norm_pixel_scale else 255
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
        self.has_transform = (
            self.norm_transform is not None or 
            self.crop_transform is not None or 
            self.resize_transform is not None
        )
        self.channel_first = channel_first
        self.create_underlying_cont(gpu_idx)

    def create_underlying_cont(self, gpu_id):
        self.gpu_id = gpu_id

        decoder_basename = self.meta["codec"]
        if self.gpu_id >= 0:
            # TODO(miguelmartin): 
            # interpolation algorithm is known & seems to be implementation specific
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
        ret = _yuv_to_rgb(fs[0])  # pyre-ignore
        if self.has_transform:
            ret = ret.permute(3, 0, 1, 2)
            assert ret.shape[1] == self.frame_window_size
            # NOTE: all transforms expect C, T, H, W
            if self.resize_transform is not None:
                ret = self.resize_transform(ret)
            if self.crop_transform is not None:
                ret = self.crop_transform(ret)
            if not self.norm_pixel_scale:
                ret = (ret * 255)
            if self.norm_transform is not None:
                ret = self.norm_transform(ret)
            elif not self.norm_pixel_scale:
                ret = ret.to(dtype=torch.uint8)

            # C, T, H, W -> T, H, W, C
            if not self.channel_first:
                ret = ret.permute(1, 2, 3, 0)
        elif self.channel_first:
            # T, H, W, C -> C, T, H, W
            ret = ret.permute(3, 0, 1, 2)
            if self.norm_pixel_scale:
                ret = (ret * 255)
                ret = ret.to(dtype=torch.uint8)
        else:
            # T, H, W, C
            pass

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
        pixel_scale: float = 1.0,
        channel_first: bool = False,
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
        self.pixel_scale = pixel_scale
        self.has_transform = (
            self.norm_transform is not None or 
            self.crop_transform is not None or 
            self.resize_transform is not None
        )
        self.channel_first = channel_first
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
                    (torch.tensor(f.to_ndarray(format="rgb24"), dtype=torch.float32) / 255) * self.pixel_scale,
                )
            )
        fs = sorted(fs, key=lambda x: x[0])
        ret = torch.stack([x[1] for x in fs])
        if self.has_transform:
            ret = ret.permute(3, 0, 1, 2)
            # NOTE: all transforms expect C, T, H, W
            if self.resize_transform is not None:
                ret = self.resize_transform(ret)
            if self.crop_transform is not None:
                ret = self.crop_transform(ret)
            if self.norm_transform is not None:
                ret = self.norm_transform(ret)
            # C, T, H, W -> T, H, W, C
            if not self.channel_first:
                ret = ret.permute(1, 2, 3, 0)
        elif self.channel_first:
            # T, H, W, C => C, T, H, W
            ret = ret.permute(3, 0, 1, 2)
        else:
            # T, H, W, C
            pass

        return {
            "video": ret,
            "frame_start_idx": frame_i,
            "frame_end_idx": frame_j,
        }
