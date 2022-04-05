# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

import functools
import time
from typing import List, Optional, Tuple

import hydra
import torch
from ego4d.features.config import Video, FeatureExtractConfig, get_videos, load_model
from ego4d.features.extract_features import (
    extract_features,
    num_fvs,
)
from ego4d.features.slurm import (
    create_executor,
)


def profile_extraction(config: FeatureExtractConfig):
    # videos, _ = get_videos(config)
    _, videos = get_videos(config)
    videos = [v for v in videos if v.frame_count > 500 and v.frame_count <= 2000]
    # mfc = (max([v.frame_count for v in videos]))
    # videos = [v for v in videos if v.frame_count == mfc]
    videos = videos[0:1]

    print("Frame count=", videos[0].frame_count)

    print(f"Got {len(videos)} videos")

    batch_sizes = [1]
    num_workers = [15]
    # num_workers = [9]
    model = load_model(config)

    num_examples = -1

    print(
        "num_examples,batch_size,num_workers,total,mean,forward_pass,to_load,transfer_to_device"
    )
    for batch_size, num_workers in zip(batch_sizes, num_workers):
        config.inference_config.batch_size = batch_size
        config.inference_config.num_workers = num_workers

        print(batch_size, num_workers)

        t1 = time.time()
        ef = extract_features(
            videos=videos,
            config=config,
            model=model,
            log_info=False,
            max_examples=num_examples,
        )
        time_stats = ef.time_stats
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
                f"{num_examples},{batch_size},{num_workers},{total_time},{mean_sum},{forward_pass.mean()},{to_load.mean()},{transfer_time.mean()}"
            )
        else:
            print(
                f"{num_examples},{batch_size},{num_workers},{total_time},{mean_sum},{forward_pass.mean()/batch_size},{to_load.mean()/batch_size},{transfer_time.mean()/batch_size}"
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
