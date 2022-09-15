import copy

import numpy as np

import torch.nn as nn
from ego4d.features.config import FeatureExtractConfig, Video
from ego4d.features.extract_features import extract_features
from ego4d.features.inference import _video_info as video_info
from ego4d.research.clep.config import TrainConfig
from sentence_transformers import SentenceTransformer


def run_feature_extraction(
    path: str, model: nn.Module, feature_extract_config: FeatureExtractConfig
):
    v_info = video_info(path)
    # pyre-ignore
    vid = Video(
        path,
        path,
        v_info["num_frames"],
        w=None,
        h=None,
        has_audio=False,
        is_stereo=False,
    )
    if vid.frame_count is None:
        return None

    feature_extract_config = copy.deepcopy(feature_extract_config)
    fps = v_info["fps"]
    assert fps is not None
    feature_extract_config.inference_config.fps = int(np.round(float(fps)))
    feature_extract_config.inference_config.stride = int(
        np.round(float(fps * (16 / 30)))
    )
    feature_extract_config.inference_config.frame_window = int(
        np.round(float(fps * (32 / 30)))
    )

    return extract_features(
        videos=[vid],
        config=feature_extract_config,
        model=model,
        log_info=False,
        silent=True,
        assert_feature_size=False,
    )


def get_language_model(config: TrainConfig) -> nn.Module:
    return SentenceTransformer(config.pre_config.ego4d_narr.st_model_name)
