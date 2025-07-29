#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import datetime
import functools
import math
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import hydra

import numpy as np
import submitit
import torch
from ego4d.features.config import (
    FeatureExtractConfig,
    get_videos,
    ScheduleConfig,
    Video,
)
from ego4d.features.extract_features import num_fvs, perform_feature_extraction
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from tqdm import tqdm


def greedy_create_batches(
    videos: List[Video], times: List[float], max_time_per_batch: int
) -> List[List[Tuple[Video, float]]]:
    """
    Greedy bin packing algorithm.

    For each bin:
    - Packs the most amount of big videos into the bin
    - Pack any small videos remaining that fit

    big/small corresponding to the time estimate for completion
    """
    assert len(videos) > 0, "empty videos"
    assert len(videos) == len(times), f"videos={len(videos)} vs. times={len(times)}"

    vt = [(v, t) for v, t in zip(videos, times)]
    vt.sort(key=lambda x: x[1])

    assert vt[-1][1] <= max_time_per_batch, f"""
        can't batch things if the max time is larger than max_time_per_batch
        {vt[-1][-1]} vs. {max_time_per_batch}
    """

    n = len(videos)

    i = 0
    j = n - 1

    batches = []
    curr = []
    curr_time = 0

    while i <= j:
        old_i, old_j = i, j

        while i <= j and curr_time + times[i] <= max_time_per_batch:
            curr.append((videos[i], times[i]))
            curr_time += times[i]
            i += 1

        while j >= i and curr_time + times[j] <= max_time_per_batch:
            curr.append((videos[j], times[j]))
            curr_time += times[j]
            j -= 1

        assert i != old_i or j != old_j, f"""
        Could not batch it up -
            i = {i}, j = {j}
            old_i = {old_i}, old_j = {old_j}
            time_i = {times[i]}, time_j = {times[j]}
            max_time_per_batch = {max_time_per_batch}
        """

        batches.append(curr)
        curr = []
        curr_time = 0

    if len(curr) > 0:
        batches.append(curr)

    return batches


def validate_batches(timeout_minutes, times_per_batch, batched_vids):
    for t, b in zip(times_per_batch, batched_vids):
        print(len(b), t)
        assert t <= timeout_minutes * 60, f"""
            Algorithm to batch videos is incorrect.

            Batch estimated time = {t}
            Timeout minutes = {timeout_minutes}
            Timeout seconds = {timeout_minutes * 60}
        """


def batch_videos(
    videos: List[Video], config: FeatureExtractConfig
) -> List[List[Video]]:
    # estimate each time to extract per sub-clip
    num_forward_passes = [
        np.ceil(
            num_fvs(v, config.inference_config) / config.inference_config.batch_size
        )
        for v in videos
    ]
    times = [
        config.schedule_config.overhead
        * n
        * config.schedule_config.time_per_forward_pass
        for n in num_forward_passes
    ]

    # batch up these videos into config.timeout_min batches
    batched_vt = greedy_create_batches(
        videos, times, max_time_per_batch=config.schedule_config.timeout_min * 60
    )
    batched_vids = [[uid for uid, _ in b] for b in batched_vt]

    times_per_batch = [sum(t for _, t in b) for b in batched_vt]

    indicies = list(range(len(times_per_batch)))
    indicies.sort(key=lambda i: times_per_batch[i])
    batched_vids = [batched_vids[i] for i in indicies]
    times_per_batch = [times_per_batch[i] for i in indicies]

    validate_batches(config.schedule_config.timeout_min, times_per_batch, batched_vids)
    return batched_vids


def print_stats_for_videos(
    config: FeatureExtractConfig, all_videos: List[Video], videos: List[Video]
):
    # stats for uids
    assert isinstance(all_videos[0], Video)
    assert isinstance(videos[0], Video)
    total_secs_uncompleted = sum(v.frame_count * config.fps for v in videos)
    secs_uncompleted = sum(v.frame_count * config.fps for v in all_videos)

    print(
        f"""
    Total Number of Videos = {len(all_videos)}
    Incomplete videos = {len(videos)}

    Total Seconds = {total_secs_uncompleted}
    Incomplete seconds = {secs_uncompleted} = {secs_uncompleted/total_secs_uncompleted * 100:.2f}%
    """
    )


def print_stats_for_scheduling(
    config: FeatureExtractConfig, batch_vids: List[List[Video]]
):
    schedule_time = config.schedule_config.schedule_time_per_node

    n_schedules = math.ceil(
        len(batch_vids) / config.schedule_config.slurm_array_parallelism
    )
    print("n schedules =", n_schedules)
    schedule_overhead = n_schedules * schedule_time * 60

    print(f"Will schedule {n_schedules}")

    sec_to_take = (
        n_schedules * 60 * config.schedule_config.timeout_min + schedule_overhead
    )
    print(sec_to_take)
    print(
        f"""
    {len(batch_vids)} batches
    {config.schedule_config.slurm_array_parallelism} batch of machines.

    Will take: {sec_to_take} seconds
        Schedule overhead: - {schedule_overhead} ({100*schedule_overhead/sec_to_take:.2f}%)
    """
    )


def print_completion_stats(results):
    time_to_load = []
    time_to_transfer = []
    forward_pass_time = []
    print(
        "overall,save_time,avg_total,avg_load_time,avg_transfer_time,avg_forward_pass"
    )
    for result in results:
        time_to_load.extend(result.to_load)
        time_to_transfer.extend(result.transfer_device)
        forward_pass_time.extend(result.forward_pass)

        ttl = torch.Tensor(result.to_load)
        ttt = torch.Tensor(result.transfer_device)
        fpt = torch.Tensor(result.forward_pass)
        mean_sum = ttl.mean() + ttt.mean() + fpt.mean()
        print(
            f"{result.overall},{result.to_save},{mean_sum},{ttl.mean()},{ttt.mean()},{fpt.mean()}"
        )

    print("")
    print("")

    print("Averages")
    print("mean_forward,only_forward_pass,time_to_load,time_to_transfer")
    ttl = torch.Tensor(time_to_load)
    ttt = torch.Tensor(time_to_transfer)
    fpt = torch.Tensor(forward_pass_time)
    mean_sum = ttl.mean() + ttt.mean() + fpt.mean()
    print(f"{mean_sum},{ttl.mean()},{ttt.mean()},{fpt.mean()}")


def create_executor(config: ScheduleConfig):
    if not config.run_locally:
        print("Using slurm/auto executor")
        executor = submitit.AutoExecutor(folder=config.log_folder)
    else:
        print("Using local executor")
        executor = submitit.LocalExecutor(folder=config.log_folder)

    executor.update_parameters(
        timeout_min=config.timeout_min,
        constraint=config.constraint,
        slurm_partition=config.slurm_partition,
        slurm_array_parallelism=config.slurm_array_parallelism,
        gpus_per_node=config.gpus_per_node,
        cpus_per_task=config.cpus_per_task,
    )
    return executor


@hydra.main(config_path="configs", config_name=None)
def schedule_feature_extraction(config: FeatureExtractConfig):
    print("###################### Feature Extraction Config ####################")
    print(OmegaConf.to_yaml(config))
    print("############################################################")

    # Get uids + {uids -> duration}
    videos, all_videos = get_videos(config)
    os.makedirs(config.io.out_path, exist_ok=True)
    with open(f"{config.io.out_path}/config.yaml", "w") as out_f:
        out_f.write(OmegaConf.to_yaml(config))

    print_stats_for_videos(config, all_videos=all_videos, videos=videos)

    # ------ Compute real timeout per batch
    batch_vids = batch_videos(videos=videos, config=config)
    config.schedule_config.slurm_array_parallelism = min(
        config.schedule_config.slurm_array_parallelism, len(batch_vids)
    )

    # ------ Schedule
    executor = create_executor(config.schedule_config)
    print_stats_for_scheduling(config, batch_vids)

    if not config.force_yes:
        print(f"Time is: {datetime.datetime.now()}")
        cont = input("Continue? [y/N]: ")
        if cont != "y":
            print("Exiting...")
            sys.exit(0)

    jobs = executor.map_array(
        functools.partial(perform_feature_extraction, config=config),
        batch_vids,
    )
    print(f"Jobs: {jobs}")

    # TODO: display better stats in tqdm ?
    results = []
    for job in tqdm(jobs):
        results.append(job.result())
    print_completion_stats(results)


if __name__ == "__main__":
    schedule_feature_extraction()
