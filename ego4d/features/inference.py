# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

import json
import os
import random
from dataclasses import dataclass
from typing import Optional

import av
import hydra
import pandas as pd
import torch
import torchvision.transforms as T
from ego4d.features.config import FeatureExtractConfig, load_model, Video
from ego4d.features.extract_features import extract_features
from PIL import Image


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
        if len(container.streams.video) == 0:
            return {
                "num_frames": None,
                "fps": None,
            }
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


def _load_imagenet_label_dir_dict(dataset_dir: str):
    path = os.path.join(dataset_dir, "labels.txt")
    df = pd.read_csv(path, names=["dir_name", "class_name"], header=None)
    return dict(zip(df.dir_name.tolist(), df.class_name.tolist()))


def _load_imagenet_class_names():
    # NOTE
    # this class mapping is for omnivore
    # this may or may not be applicable to you
    # wget https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
    with open(
        "/private/home/miguelmartin/ego4d/ego4d_public/imagenet_class_index.json", "r"
    ) as f:
        imagenet_classnames = json.load(f)

    # Create an id to label name mapping
    imagenet_id_to_classname = {}
    for k, v in imagenet_classnames.items():
        imagenet_id_to_classname[k] = v[1]
    return imagenet_id_to_classname


def inference_imagenet(config: RunInferenceConfig):
    # taken from https://github.com/facebookresearch/omnivore/blob/main/inference_tutorial.ipynb
    label_mapping = _load_imagenet_class_names()
    label_dir_dict = _load_imagenet_label_dir_dict(config.dataset_dir)

    image_set_dir = os.path.join(config.dataset_dir, config.set_to_use)
    labels = os.listdir(image_set_dir)

    for _ in range(config.num_examples):
        expected_label = random.sample(labels, 1)[0]
        expected_label_value = label_dir_dict[expected_label]

        images_dir = os.path.join(image_set_dir, expected_label)
        images_in_dir = os.listdir(images_dir)

        image_path = random.sample(images_in_dir, 1)[0]
        image_path = f"{images_dir}/{image_path}"

        image = Image.open(image_path).convert("RGB")
        image_transform = T.Compose(
            [
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        image = image_transform(image)
        image = image[None, :, None, ...].to(config.inference_config.device)
        model = load_model(config, patch_final_layer=False)
        predictions = model(image)

        pred_classes = torch.nn.Softmax(dim=1)(predictions).squeeze()
        top_k = pred_classes.topk(k=config.top_k)
        predictions_strs = [
            f"- {label_mapping[str(idx.item())]}: {prob.item():.2%}"
            for prob, idx in zip(top_k.values, top_k.indices)
        ]
        predictions_summary = "\n".join(predictions_strs)
        print(
            f"""Example: {image_path}

Expected label: {expected_label_value}
Predicted labels:
{predictions_summary}
    """
        )


def inference_k400(config: RunInferenceConfig):
    """
    Follows https://pytorchvideo.org/docs/tutorial_torchhub_inference

    Assumes kinetics is structured as:
    - <basedir>/train
    - <basedir>/val
    - <basedir>/test
    """
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
    if config.seed is not None:
        random.seed(config.seed)

    assert config.dataset_type in ("k400", "imagenet")
    if config.dataset_type == "k400":
        inference_k400(config)
    else:
        inference_imagenet(config)


if __name__ == "__main__":
    main()  # pyre-ignore
