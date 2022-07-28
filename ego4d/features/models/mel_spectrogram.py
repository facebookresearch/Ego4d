# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

from dataclasses import dataclass
from typing import Optional

import torch
from ego4d.features.config import BaseModelConfig, InferenceConfig
from torch.nn import Module
from torchaudio.transforms import MelSpectrogram
from torchvision.transforms import Compose


@dataclass
class ModelConfig(BaseModelConfig):
    n_fft: int = 1024
    win_length: Optional[int] = None
    hop_length: int = 160  # 10ms
    n_mels: int = 128


class MelSpectrogramModel(Module):
    def __init__(self, inference_config: InferenceConfig, model_config: ModelConfig):
        super().__init__()
        self.inference_config = inference_config
        self.config = model_config

    def get_mel_spectrogram_transform(self, freq):
        return MelSpectrogram(
            sample_rate=self.inference_config.norm_config.resample_audio_rate,
            n_fft=self.config.n_fft,
            win_length=self.config.win_length,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
        )

    def forward(self, x) -> torch.Tensor:
        assert len(x["audio_sample_rate"]) == 1
        f = self.get_mel_spectrogram_transform(x["audio_sample_rate"][0])
        if "audio" not in x:
            return torch.empty(1, 1)
        return f(x["audio"])


def load_model(
    inference_config: InferenceConfig,
    config: ModelConfig,
    patch_final_layer: bool = True,
) -> Module:
    model = MelSpectrogramModel(inference_config, config)
    # don't need to set to GPU for a Mel spectrogram - but will do anyway
    model = model.eval()
    model = model.to(inference_config.device)
    return model


def get_transform(inference_config: InferenceConfig, config: ModelConfig):
    # do nothing
    return Compose([])
