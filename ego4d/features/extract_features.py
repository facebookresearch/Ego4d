# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

import gc
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import torch
from ego4d.features.config import (
    FeatureExtractConfig,
    InferenceConfig,
    load_model,
    Video,
)
from ego4d.features.dataset import create_data_loader_or_dset
from torch.nn import Module
from tqdm.auto import tqdm


@dataclass
class TimeStats:
    to_load: List[float]
    transfer_device: List[float]
    forward_pass: List[float]

    overall: Optional[float] = None
    to_save: Optional[float] = None


@dataclass
class FeatureExtractionResult:
    result: Dict[str, torch.Tensor]
    time_stats: TimeStats


@dataclass
class ExtractedFeature:
    video_uid: str
    clip_index: int
    feature: torch.Tensor
    time_to_load: List[float]
    time_transfer_device: List[float]
    time_forward_pass: List[float]


# TODO: use this method in pytorchvideo test_uniform_clip_sampler
# NOTE this is copied from there - hence has been tested
def _num_fvs(
    num_frames: int,
    fps: float,
    stride_frames: int,
    window_size_frames: int,
    backpad_last: bool = True,
) -> int:
    N = num_frames - window_size_frames
    if N < 0:
        return 1

    result = N // stride_frames + 1

    # handle padded frame
    if backpad_last and N % stride_frames != 0:
        result += 1
    return result


def num_fvs(vid: Video, config: InferenceConfig) -> int:
    return _num_fvs(vid.frame_count, config.fps, config.stride, config.frame_window)


def _extract_features(
    model: Module,
    videos: List[Video],
    config: FeatureExtractConfig,
    max_examples: int = -1,
    silent: bool = False,
) -> Iterator[ExtractedFeature]:
    device = config.inference_config.device

    data = create_data_loader_or_dset(videos, config)

    if not silent:
        print(f"extracting from {videos}")

    t1 = time.time()
    for i, x in enumerate(data):
        t2 = time.time()

        inputs = x["video"]

        load_time = t2 - t1

        t1 = time.time()
        if isinstance(inputs, list):
            inputs = [i.to(device) for i in inputs]
        else:
            inputs = inputs.to(device)
        batch_size = inputs[0].shape[0]
        t2 = time.time()
        transfer_time = t2 - t1

        with torch.no_grad():
            t1 = time.time()
            fv = model(inputs).detach().cpu()
            t2 = time.time()
            forward_pass_time = t2 - t1

            yield ExtractedFeature(
                video_uid=x["video_name"],
                clip_index=x["clip_index"],
                feature=fv,
                time_to_load=load_time,
                time_transfer_device=transfer_time,
                time_forward_pass=forward_pass_time,
            )

        if i % 50 == 0:
            gc.collect()

        if max_examples > 0 and (i + 1) * batch_size >= max_examples:
            if not silent:
                print("Breaking...")
            break
        t1 = time.time()


def extract_features(
    videos: List[Video],
    config: FeatureExtractConfig,
    model: Optional[Module] = None,
    log_info: bool = True,
    max_examples: int = -1,
    silent: bool = False,
    assert_feature_size: bool = True,
) -> FeatureExtractionResult:
    if model is None:
        model = load_model(config)

    if log_info and not silent:
        print(f"config={config}")
        print(f"Number of videos = {len(videos)}")

    fvs = defaultdict(list)
    time_to_load = []
    time_transfer_device = []
    time_forward_pass = []

    uid_to_video_clips = {}
    for v in videos:
        assert v.uid not in uid_to_video_clips
        uid_to_video_clips[v.uid] = v

    # TODO: check this
    # NOTE: not quite right if one is < 0
    batch_size = config.inference_config.batch_size
    total_num_clips = sum(num_fvs(v, config.inference_config) for v in videos)
    total_num_clips /= max(batch_size, 1)
    total_num_clips = math.ceil(total_num_clips)

    all_vectors = []
    if not silent:
        print(
            f"extracting features - there are {total_num_clips} for {len(videos)} videos",
            flush=True,
        )

    for ef in tqdm(
        _extract_features(
            model, videos, config, max_examples=max_examples, silent=silent
        ),
        total=total_num_clips,
    ):
        if batch_size == 0:
            key = ef.video_uid
            fvs[key].append((ef.clip_index.item(), ef.feature.squeeze()))
            all_vectors.append(ef.feature.squeeze())
        else:
            assert len(ef.video_uid) == len(ef.feature)
            assert len(ef.video_uid) == len(ef.clip_index)

            for i in range(len(ef.video_uid)):
                key = ef.video_uid[i]
                fvs[key].append((ef.clip_index[i].item(), ef.feature[i].squeeze()))
                all_vectors.append(ef.feature[i].squeeze())

        time_to_load.append(ef.time_to_load)
        time_transfer_device.append(ef.time_transfer_device)
        time_forward_pass.append(ef.time_forward_pass)

    result = {}
    total_num = 0
    for k, efs in fvs.items():
        efs.sort(key=lambda x: x[0])

        result[k] = torch.stack([x[1] for x in efs])

        fv_amount = len(result[k])
        clip = uid_to_video_clips[k]
        expected_fvs = num_fvs(clip, config.inference_config)
        if expected_fvs != fv_amount:
            if assert_feature_size:
                raise AssertionError(
                    f"""
{k} should have {expected_fvs} fvs, but has {fv_amount}
"""
                )

        total_num += fv_amount

    if max_examples > 0:
        assert total_num == max_examples

    return FeatureExtractionResult(
        result=result,
        time_stats=TimeStats(
            to_load=time_to_load,
            transfer_device=time_transfer_device,
            forward_pass=time_forward_pass,
        ),
    )


def perform_feature_extraction(
    videos: List[Video], config: FeatureExtractConfig
) -> TimeStats:
    os.makedirs(config.io.out_path, exist_ok=True)

    # TODO batch this
    time_stats = TimeStats(
        to_load=[],
        transfer_device=[],
        forward_pass=[],
        overall=0,
        to_save=0,
    )

    o1 = time.time()
    for vid in tqdm(videos, desc="videos"):
        gc.collect()

        # TODO: convert this to be a generator
        feature_extract_result = extract_features([vid], config)
        result = feature_extract_result.result

        t1 = time.time()
        for k, v in result.items():
            torch.save(v, f"{config.io.out_path}/{k}.pt")
        t2 = time.time()

        time_stats.to_save += t2 - t1
        time_stats.to_load.extend(feature_extract_result.time_stats.to_load)
        time_stats.transfer_device.extend(
            feature_extract_result.time_stats.transfer_device
        )
        time_stats.forward_pass.extend(feature_extract_result.time_stats.forward_pass)

    o2 = time.time()

    time_stats.overall = o2 - o1
    return time_stats
