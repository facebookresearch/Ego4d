# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

import functools
import time
from typing import List, Optional, Tuple

import hydra
import torch
from ego4d.features.config import FeatureExtractConfig, get_videos, load_model, Video
from ego4d.features.extract_features import extract_features, num_fvs
from ego4d.features.slurm import create_executor


def profile_extraction(config: FeatureExtractConfig):
    videos, all_videos = get_videos(config)
    if videos:
        print(
            f"Video 0/{len(videos)} ({videos[0].frame_count} | {videos[0].path}): {videos[0]}"
        )
    # videos = [v for v in all_videos if v.frame_count <= 500]
    videos = [v for v in all_videos if v.frame_count > 2000 and v.frame_count <= 10000]
    # videos = [v for v in all_videos if "f1bdf9f3-4f65-4c70-b8ba-b3d4607c0cff" in v.uid]
    videos = videos[1:2]
    print(videos)

    assert len(videos) > 0, "No videos to process!"

    print(f"Got {len(videos)} videos")

    batch_sizes = [1]
    num_workers = [2]
    prefetch_factor = [2]
    model = load_model(config)

    num_examples = -1

    print(
        "prefetch_factor,batch_size,num_workers,total,mean,forward_pass,to_load,transfer_to_device"  # noqa
    )
    for batch_size, nw, pf in zip(batch_sizes, num_workers, prefetch_factor):
        config.inference_config.batch_size = batch_size
        config.inference_config.num_workers = nw
        config.inference_config.prefetch_factor = pf

        t1 = time.time()
        time_stats = extract_features(
            videos=videos,
            config=config,
            model=model,
            log_info=False,
            max_examples=num_examples,
            silent=False,
        ).time_stats
        t2 = time.time()

        total_time = t2 - t1

        forward_pass = torch.Tensor(time_stats.forward_pass)
        to_load = torch.Tensor(time_stats.to_load)
        transfer_time = torch.Tensor(time_stats.transfer_device)

        assert len(forward_pass.shape) == 1
        assert len(to_load.shape) == 1
        assert len(transfer_time.shape) == 1

        mean_sum = forward_pass.mean() + to_load.mean() + transfer_time.mean()
        mean_sum /= max(1, batch_size)

        if num_examples > 0:
            assert forward_pass.shape[0] * batch_size == num_examples

        if batch_size == 0:
            print(
                f"{pf},{batch_size},{nw},{total_time},{mean_sum},{forward_pass.mean()},{to_load.mean()},{transfer_time.mean()}"  # noqa
            )
        else:
            print(
                f"{pf},{batch_size},{nw},{total_time},{mean_sum},{forward_pass.mean()/batch_size},{to_load.mean()/batch_size},{transfer_time.mean()/batch_size}"  # noqa
            )


@hydra.main(config_path="configs", config_name=None)
def schedule_profile_extraction(config: FeatureExtractConfig):
    if config.schedule_config.run_locally:
        profile_extraction(config)
    else:
        executor = create_executor(config.schedule_config)
        job = executor.submit(functools.partial(profile_extraction, config=config))
        print(f"{job}")

        # wait for the job
        job.result()


if __name__ == "__main__":
    schedule_profile_extraction()  # pyre-ignore
