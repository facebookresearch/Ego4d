import json
from typing import List
import pandas as pd

import torch
from pytorchvideo.data.encoded_video import EncodedVideo
from torch.utils.data import DataLoader
from ego4d.features.dataset import CropIfStereo
from ego4d.features.models.omnivore import get_transform as omnivore_transform
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)

from ego4d.vaclip.config import TrainConfig, InputConfig, TransformConfig


class Ego4DVaClip(torch.utils.data.Dataset):
    # TODO
    # refactor into seperate heads such that we can load features instead of
    # raw inputs for models
    def __init__(
        self,
        uid_subset: List[str],
        config: InputConfig,
        transform,
    ):
        self.narration_json = json.load(open(config.narration_json_path))
        uid_subset = set(uid_subset)

        # TODO: unified way to do this on FAIR Cluster and FB Infra
        self.manifest = pd.read_csv(f"{config.video_dir_path}/manifest.csv")
        self.encoded_videos = {
            uid: EncodedVideo.from_path(
                f"{config.video_dir_path}/{uid}.mp4",
                decode_video=True,
                decode_audio=False,  # TODO: decode_audio set to True
                perform_seek=True,
            )
            for uid in uid_subset
        }
        self.is_uid_stereo = {
            v["video_uid"]: v["is_stereo"]
            for v in json.load(open("/checkpoint/miguelmartin/ego4d/top_level.json"))["videos"]
        }

        self.narrations = [
            (uid, data["narration_text"], data["timestamp_sec"])
            for uid in uid_subset
            for data in self.narration_json[uid].get("narration_pass_2", {"narrations": []})["narrations"]
        ]
        self.narrations += [
            (uid, data["narration_text"], data["timestamp_sec"])
            for uid in uid_subset
            for data in self.narration_json[uid].get("narration_pass_2", {"narrations": []})["narrations"]
        ]
        self.transform = transform

    def __len__(self):
        return len(self.narrations)

    def __getitem__(self, idx):
        # TODO: how to do better negative sampling?
        uid, narration_text, timestamp_sec = self.narrations[idx]

        # get a random clip -4 to +4s
        # TODO: how to support bs > 1?
        # offset = (torch.rand(1) * 4).item()
        # TODO fix OOM?
        offset = 32 / 2 / 30

        t1 = timestamp_sec - offset
        t2 = timestamp_sec + offset
        video_clip = self.encoded_videos[uid].get_clip(t1, t2)

        # TODO: audio as well
        frames = video_clip["video"]

        sample_dict = {
            "video": frames,
            "text": narration_text,
            "is_stereo": self.is_uid_stereo[uid],
        }
        return self.transform(sample_dict)


def create_data_loader(dset, config: TrainConfig):
    return DataLoader(
        dset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
    )


def omni_transform(config: TransformConfig):
    return ApplyTransformToKey(
        key="video",
        transform=Compose([
            Lambda(lambda x: x / 255.0),
            NormalizeVideo(config.mean, config.std),
            ShortSideScale(size=config.side_size),
            CenterCropVideo(config.crop_size),
            UniformTemporalSubsample(32),  # TODO pad
        ]),
    )


def text_transform(config: TransformConfig):
    def sub_tagged_tokens(text: str) -> str:
        text = text.replace("#C", "Camera wearer")
        text = text.replace("#O", "Other person")
        text = text.replace("#unsure", "Something")
        return text

    return ApplyTransformToKey(
        key="text",
        transform=Lambda(lambda x: sub_tagged_tokens(x)),
    )


# TODO: image dataloader as a zip of above + open_clip?
def get_transform(config: TransformConfig):
    omni_t = omni_transform(config)
    txt_t = text_transform(config)
    return Compose([
        CropIfStereo(),
        txt_t,
        omni_t,
    ])
