# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

from dataclasses import dataclass
from typing import Tuple, List

import torch
from speechbrain.pretrained import EncoderDecoderASR
from ego4d.features.config import InferenceConfig, BaseModelConfig
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torch.nn import Module, Identity
from torchvision.transforms import Compose, Lambda
from torchaudio.transforms import Resample, MelSpectrogram
from typing import Dict, Optional


@dataclass
class ModelConfig(BaseModelConfig):
    source: str = "speechbrain/asr-crdnn-transformerlm-librispeech"
    savedir: str = "pretrained_models/asr-crdnn-transformerlm-librispeech"


class WrapAsrModel(Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x) -> Optional[List[str]]:
        assert len(x["audio_sample_rate"]) == 1
        if "audio" not in x:
            return []

        # see
        # https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/pretrained/interfaces.py#L525
        inp = self.model.audio_normalizer(x["audio"].permute(1, 0), x["audio_sample_rate"].item())
        inp = inp.unsqueeze(0)
        rel_length = torch.tensor([1.0])
        pred_words, pred_tokens = self.model(inp, rel_length)

        out = pred_words[0]
        if len(out) == 0:
            return None
        return out


def load_model(
    inference_config: InferenceConfig,
    config: ModelConfig,
    patch_final_layer: bool = True,
) -> Module:
    if patch_final_layer:
        print("WARNING: this model outputs text, and patching is not supporting")
    model = EncoderDecoderASR.from_hparams(
        source=config.source,
        savedir=config.savedir,
        run_opts={"device": inference_config.device}
    )
    return WrapAsrModel(model)


def get_transform(inference_config: InferenceConfig, config: ModelConfig):
    # do nothing
    return Compose([])
