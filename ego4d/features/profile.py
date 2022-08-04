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
    _, videos = get_videos(config)
    # videos = [v for v in videos if v.frame_count > 1000 and v.frame_count <= 2000]
    # videos = [v for v in videos if v.is_stereo]
    videos = [
        v
        for v in videos
        if v.frame_count > 1000 and v.frame_count <= 2000
        # if v.uid in (
        #     "952b1fa3-05cc-4c2b-8897-47d02cd598b8",
        #     "01aed4ae-486e-41eb-91f8-4d2e8a46db7d",
        #     "0d271871-c8ba-4249-9434-d39ce0060e58",
        #     "002ad105-bd9a-4858-953e-54e88dc7587e",
        #     "4ca97ff8-bc2a-462b-b4c9-b1a878564bea",
        #     "000a3525-6c98-4650-aaab-be7d2c7b9402",
        #     "04994ce8-9d47-44d6-a7b0-42b395e69390",
        #     "000786a7-3f9d-4fe6-bfb3-045b368f7d44",
        #     "0e7ba211-0dba-40b8-8ace-a3e5932db4fb",
        # )
        # if v.uid in (
        #     "674c6777-406d-4158-9539-b5feb88c033d",
        #     "00eab18b-912b-44ec-bca8-c76e94e9e260",
        #     "003b145d-42d3-470d-b987-8a489c42f2f8",
        #     "337910b4-7ada-4703-aeca-d0b29428ed4e",
        # )
        # if v.frame_count > 1000 and v.frame_count <= 2000
    ]
    videos = videos[0:1]

    print(f"Got {len(videos)} videos")

    batch_sizes = [1, 1, 1, 1, 1]
    num_workers = [9, 10, 10, 10, 10]
    prefetch_factor = [2, 3, 4, 5]
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
            silent=True,
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
                f"{pf},{batch_size},{nw},{mean_sum},{forward_pass.mean()/batch_size},{to_load.mean()/batch_size},{transfer_time.mean()/batch_size}"  # noqa
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
