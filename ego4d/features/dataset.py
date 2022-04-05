# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

from fractions import Fraction
from typing import List, Any

import torch
from ego4d.features.config import Video, FeatureExtractConfig, get_transform
from pytorchvideo.data import UniformClipSampler
from pytorchvideo.data.encoded_video import EncodedVideo
from torch.utils.data import DataLoader


class IndexableVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config: FeatureExtractConfig,
        videos: List[Video],
        sampler,
        transform
    ):
        self.config = config
        self.clips = []
        self.sampler = sampler
        self.transform = transform

        self.encoded_videos = {
            v.uid: EncodedVideo.from_path(
                v.path,
                decode_video=config.inference_config.include_video,
                decode_audio=config.inference_config.include_audio,
                perform_seek=False,
            )
            for v in videos
        }

        for v in videos:
            self.clips.extend(
                list(
                    get_all_clips(
                        v,
                        self.encoded_videos[v.uid].duration,
                        sampler
                    )
                )
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
        datum = encoded_video.get_clip(clip_start, clip_end)
        v_frames = datum["video"]
        a_frames = datum["audio"]

        sample_dict = {
            "video_name": video.uid,
            "video_index": idx,
            "clip_index": clip_index,
            "aug_index": aug_index,
            "clip_start_sec": float(clip_start),
            "clip_end_sec": float(clip_end),
        }
        if v_frames is not None:
            sample_dict["video"] = v_frames
        if a_frames is not None:
            sample_dict["audio"] = a_frames
        if encoded_video._has_audio:
            sample_dict["audio_sample_rate"] = encoded_video._container.streams.audio[0].rate

        sample_dict = self.transform(sample_dict)
        return sample_dict


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


def create_dset(
    videos: List[Video], config: FeatureExtractConfig
) -> IndexableVideoDataset:
    assert isinstance(videos[0], Video)

    clip_sampler = UniformClipSampler(
        clip_duration=Fraction(
            config.inference_config.frame_window, config.inference_config.fps
        )
        if isinstance(config.inference_config.frame_window, int)
        else config.inference_config.frame_window,
        stride=Fraction(config.inference_config.stride, config.inference_config.fps)
        if isinstance(config.inference_config.stride, int)
        else config.inference_config.stride,
        backpad_last=True,
    )

    transform = get_transform(config)
    return IndexableVideoDataset(config, videos, clip_sampler, transform)


def create_data_loader(dset, config: FeatureExtractConfig) -> DataLoader:
    if config.inference_config.batch_size == 0:
        raise AssertionError("not supported")

    if config.inference_config.num_workers == 0:  # for debugging
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
