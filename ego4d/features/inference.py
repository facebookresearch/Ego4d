# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

import json
import os
import random
from dataclasses import dataclass
from typing import Optional

import av
import hydra
import torch
from ego4d.features.config import Video, FeatureExtractConfig, load_model
from ego4d.features.extract_features import (
    extract_features,
)


@dataclass
class RunInferenceConfig(FeatureExtractConfig):
    dataset_type: str = "k400"
    dataset_dir: str = "/datasets01/Kinetics400_Frames/videos/"
    set_to_use: str = "val"
    num_examples: int = 5
    top_k: int = 2
    seed: Optional[int] = None


def _video_info(path: str):
    with av.open(path) as container:
        return {
            "num_frames": container.streams.video[0].frames,
            "fps": container.streams.video[0].average_rate,
        }


def _load_kinetics_class_names():
    # wget \
    # https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json
    with open(
        "/private/home/miguelmartin/ego4d/ego4d_public/kinetics_classnames.json", "r"
    ) as f:
        kinetics_classnames = json.load(f)

    # Create an id to label name mapping
    kinetics_id_to_classname = {}
    for k, v in kinetics_classnames.items():
        kinetics_id_to_classname[v] = str(k).replace('"', "")
    return kinetics_id_to_classname


def inference_imagenet(config: RunInferenceConfig):
    raise AssertionError("not implemented")


def inference_k400(config: RunInferenceConfig):
    """
    Follows https://pytorchvideo.org/docs/tutorial_torchhub_inference

    Assumes kinetics is structured as:
    - <basedir>/train
    - <basedir>/val
    - <basedir>/test
    """
    if config.seed is not None:
        random.seed(config.seed)

    video_set_dir = os.path.join(config.dataset_dir, config.set_to_use)

    model = load_model(config, patch_final_layer=False)
    label_mapping = _load_kinetics_class_names()
    labels = os.listdir(video_set_dir)

    for i in range(config.num_examples):
        expected_label = random.sample(labels, 1)[0]

        videos_dir = os.path.join(video_set_dir, expected_label)
        videos_in_dir = [x for x in os.listdir(videos_dir) if x.endswith("mp4")]
        video_path = random.sample(videos_in_dir, 1)[0]  # pick an example
        video_path = f"{videos_dir}/{video_path}"

        input_id = "logit"
        info = _video_info(video_path)
        videos = [Video(input_id, video_path, info["num_frames"])]

        predictions = extract_features(
            videos=videos,
            config=config,
            model=model,
            log_info=False,
            silent=True,
            assert_feature_size=False,
        ).result[input_id]

        pred_classes = torch.nn.Softmax(dim=0)(predictions.mean(0))
        top_k = pred_classes.topk(k=config.top_k)
        predictions_strs = [
            f"- {label_mapping[idx.item()]}: {prob.item():.2%}"
            for prob, idx in zip(top_k.values, top_k.indices)
        ]
        predictions_summary = "\n".join(predictions_strs)

        print(
            f"""Example {i+1}: {video_path}

Expected label: {expected_label}
Predicted labels:
{predictions_summary}
"""
        )


@hydra.main(config_path="configs", config_name=None)
def main(config: RunInferenceConfig):
    assert config.dataset_type in ("k400", "imagenet")
    if config.dataset_type == "k400":
        inference_k400(config)
    else:
        inference_imagenet(config)


if __name__ == "__main__":
    main()  # pyre-ignore
