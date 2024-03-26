# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

import functools
from fractions import Fraction
from typing import Any, Dict, List, Optional

import av
import av.error
import numpy as np
import torch
from ego4d.features.config import FeatureExtractConfig, get_transform, Video
from ego4d.research.dataset import VideoDataset

from ego4d.research.readers import PyAvReader, TorchAudioStreamReader
from pytorchvideo.data import UniformClipSampler
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.data.utils import thwc_to_cthw
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm.auto import tqdm


def get_frames(container, t1, t2, buffer, max_buffer_size, frame_window_size):
    # [t1, t2]
    ret = []

    tb = container.streams.video[0].time_base

    def is_in_range(frame):
        t = frame.pts * tb
        return t >= t1 and t < t2

    def exceeds_range(frame):
        return frame.pts * tb >= t2

    for frame in buffer.values():
        if is_in_range(frame):
            ret.append(frame)

    ret.sort(key=lambda x: x.pts)

    if len(ret) == 0 or ret[-1].pts + 1 < t2 / tb:
        while True:
            try:
                for frame in container.decode(video=0):
                    if frame.pts is None:
                        raise AssertionError("frame is None")
                    if not isinstance(frame, av.VideoFrame):
                        raise AssertionError("other packets not supported")

                    buffer[frame.pts] = frame

                    if len(buffer) > max_buffer_size:
                        del buffer[min(buffer.keys())]

                    if is_in_range(frame):
                        ret.append(frame)
                    elif exceeds_range(frame):
                        break
                break
            except av.error.EOFError:
                seek_pts = buffer[max(buffer.keys())].pts
                container.seek(seek_pts, stream=container.streams.video[0])

        ret.sort(key=lambda x: x.pts)

    ret = list({frame.pts: frame for frame in ret}.values())
    pts_in_ret = [frame.pts for frame in ret]
    if not (np.diff(pts_in_ret) > 0).all():
        raise AssertionError("not increasing sequence of frames")
    assert len(ret) == frame_window_size, f"{len(ret)} != {frame_window_size}"
    return ret


class EncodedVideoCached:
    def __init__(self, path, frame_buffer_size=16):
        self.path = path
        self.vid = EncodedVideo.from_path(path, decoder="pyav")
        self.vid._container.seek(0)

        self.frame_buffer_size = frame_buffer_size
        self.frame_buffer = {}
        self.last_t = None

    def get_clip(self, t1, t2, frame_window_size):
        if self.last_t is not None and t1 < self.last_t:
            raise AssertionError("cannot seek backward")

        vstream = self.vid._container.streams.video[0]
        vs = vstream.start_time * vstream.time_base
        frames = get_frames(
            self.vid._container,
            t1 + vs,
            t2 + vs,
            self.frame_buffer,
            self.frame_buffer_size,
            frame_window_size,
        )
        self.last_t = t1
        return {
            "video": thwc_to_cthw(
                torch.stack(
                    [torch.from_numpy(frame.to_rgb().to_ndarray()) for frame in frames]
                )
            ).to(torch.float32),
            "audio": None,
        }

    @property
    def duration(self) -> float:
        vstream = self.vid._container.streams.video[0]
        return vstream.duration * vstream.time_base


class IndexableVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config: FeatureExtractConfig,
        videos: List[Video],
        sampler,
        transform,
        frame_window_size,
    ):
        assert (
            config.inference_config.include_audio
            ^ config.inference_config.include_video
        ), """
        cannot include audio and video at the same time
        """
        self.config = config
        self.clips = []
        self.sampler = sampler
        self.transform = transform
        self.frame_window_size = frame_window_size

        if self.config.inference_config.include_video:
            self.encoded_videos = {v.uid: EncodedVideoCached(v.path) for v in videos}
        else:
            assert self.config.inference_config.include_audio
            self.encoded_videos = {
                v.uid: EncodedVideo.from_path(
                    v.path,
                    decode_audio=True,
                    decode_video=False,
                    perform_seek=True,
                    decoder="pyav",
                )
                for v in videos
            }

        for v in videos:
            self.clips.extend(
                list(get_all_clips(v, self.encoded_videos[v.uid].duration, sampler))
            )

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        video, clip = self.clips[idx]

        (
            clip_start,
            clip_end,
            clip_index,
            aug_index,
            is_last_clip,
        ) = clip

        encoded_video = self.encoded_videos[video.uid]
        datum = encoded_video.get_clip(clip_start, clip_end, self.frame_window_size)
        v_frames = datum["video"]
        a_frames = datum["audio"]
        sample_dict = {
            "video_name": video.uid,
            "video_index": idx,
            "clip_index": clip_index,
            "aug_index": aug_index,
            "is_stereo": video.is_stereo,
            "clip_start_sec": float(clip_start),
            "clip_end_sec": float(clip_end),
        }
        if v_frames is not None:
            sample_dict["video"] = v_frames
        if a_frames is not None:
            sample_dict["audio"] = a_frames
            sample_dict["audio_sample_rate"] = encoded_video._container.streams.audio[
                0
            ].rate

        sample_dict = self.transform(sample_dict)
        return sample_dict


class CropIfStereo:
    def __init__(self):
        pass

    def __call__(self, x):
        if x["is_stereo"]:
            v = x["video"]
            assert len(v.shape) == 4
            x["video"] = v[:, :, :, 0 : v.shape[-1] // 2]

            # edge case where some videos are incorrectly
            # encoded from source and weren't corrected
            if v.shape[-1] < v.shape[-2]:
                x["video"] = torch.nn.functional.interpolate(
                    x["video"],
                    size=(x["video"].shape[-1], x["video"].shape[-2] // 2),
                    mode="bilinear",
                )
            # for debugging
            # torchvision.utils.save_image(x["video"].permute(1, 0, 2, 3)[0] / 255.0, fp="/tmp/test.jpg")  # noqa
        return x


def get_all_clips(video, video_length, sampler):
    last_clip_time = None
    annotation = {}
    n_clips = 0
    while True:
        clip = sampler(last_clip_time, video_length, annotation)
        last_clip_time = clip.clip_end_sec
        n_clips += 1

        yield (video, clip)

        if clip.is_last_clip:
            break


def labels_fn(
    path: str, start_idx: int, end_idx: int, path_to_video: Dict[str, Any], config
):
    video = path_to_video[path]
    return {
        "video_name": video.uid,
        "is_stereo": video.is_stereo,
        "clip_index": start_idx // config.inference_config.stride,
        "clip_start_sec": start_idx,
        "clip_end_sec": end_idx,
    }


def create_dset(
    videos: List[Video], config: FeatureExtractConfig
) -> IndexableVideoDataset | VideoDataset:
    assert isinstance(videos[0], Video)
    transforms_to_use = [
        CropIfStereo(),
        get_transform(config),
    ]
    if config.io.debug_mode:
        transforms_to_use = [
            CropIfStereo(),
            ApplyTransformToKey(key="video", transform=ShortSideScale(size=256)),
        ]
    transform = Compose(transforms_to_use)

    clip_sampler = UniformClipSampler(
        clip_duration=(
            Fraction(config.inference_config.frame_window, config.inference_config.fps)
            if isinstance(config.inference_config.frame_window, int)
            else config.inference_config.frame_window
        ),
        stride=(
            Fraction(config.inference_config.stride, config.inference_config.fps)
            if isinstance(config.inference_config.stride, int)
            else config.inference_config.stride
        ),
        backpad_last=True,
    )

    return IndexableVideoDataset(
        config,
        videos,
        clip_sampler,
        transform,
        config.inference_config.frame_window,
    )


def create_data_loader(dset, config: FeatureExtractConfig) -> DataLoader:
    if config.inference_config.batch_size == 0:
        raise AssertionError("not supported")

    if config.inference_config.num_workers == -1:  # for debugging
        return dset

    return DataLoader(
        dset,
        batch_size=config.inference_config.batch_size,
        num_workers=config.inference_config.num_workers,
        prefetch_factor=config.inference_config.prefetch_factor,
    )


def create_data_loader_or_dset(
    videos: List[Video], config: FeatureExtractConfig
) -> Any:
    dset = create_dset(videos, config)
    return create_data_loader(dset=dset, config=config)
