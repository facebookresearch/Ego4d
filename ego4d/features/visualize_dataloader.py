# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

import functools

import random
import hydra
from ego4d.features.config import FeatureExtractConfig, get_videos, load_model
from ego4d.features.extract_features import extract_features
from ego4d.features.slurm import create_executor


def visualize_extraction(config: FeatureExtractConfig):
    _, videos = get_videos(config)
    videos = [
        v for v in videos
        if v.frame_count > 1000 and v.frame_count <= 2000
    ]
    random.shuffle(videos)
    videos = videos[0:5]

    print("Frame count=", videos[0].frame_count)
    print(f"Got {len(videos)} videos")

    config.io.debug_mode = True
    model = load_model(config)

    extract_features(
        videos=videos,
        config=config,
        model=model,
        log_info=False,
        max_examples=-1,
        silent=True,
    )


@hydra.main(config_path="configs", config_name=None)
def schedule_profile_extraction(config: FeatureExtractConfig):
    if config.schedule_config.run_locally:
        visualize_extraction(config)
    else:
        executor = create_executor(config.schedule_config)
        job = executor.submit(functools.partial(visualize_extraction, config=config))
        print(f"{job}")

        # wait for the job
        job.result()


if __name__ == "__main__":
    schedule_profile_extraction()  # pyre-ignore
