# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

from dataclasses import dataclass
from typing import Any, Dict

import torch
from ego4d.features.config import BaseModelConfig, InferenceConfig
from speechbrain.pretrained import EncoderDecoderASR
from torch.nn import Module
from torchvision.transforms import Compose


@dataclass
class ModelConfig(BaseModelConfig):
    source: str = "speechbrain/asr-crdnn-transformerlm-librispeech"
    savedir: str = "pretrained_models/asr-crdnn-transformerlm-librispeech"


class WrapAsrModel(Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x) -> Dict[str, Any]:
        assert len(x["audio_sample_rate"]) == 1
        if "audio" not in x:
            return []

        # see
        # https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/pretrained/interfaces.py#L545
        # docstring is out-dated, need to use `audio_normalizer` not `normalizer`
        inp = self.model.audio_normalizer(
            x["audio"].permute(1, 0), x["audio_sample_rate"].item()
        )

        wavs = inp.unsqueeze(0)
        wav_lens = torch.tensor([1.0])
        # https://github.com/speechbrain/speechbrain/blob/598f6eda70f9b0c9ad49b393114ff483add1fd25/speechbrain/pretrained/interfaces.py#L595
        with torch.no_grad():
            wav_lens = wav_lens.to(self.model.device)
            encoder_out = self.model.encode_batch(wavs, wav_lens)
            predicted_tokens, scores = self.model.mods.decoder(encoder_out, wav_lens)
            pred_words = [
                self.model.tokenizer.decode_ids(token_seq)
                for token_seq in predicted_tokens
            ]
        # or...
        # pred_words, pred_tokens = self.model(wavs, wav_lens)

        return {
            "text": pred_words[0] if len(pred_words[0]) > 0 else None,
            "score": scores[0].item(),
        }


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
        run_opts={"device": inference_config.device},
    )
    return WrapAsrModel(model)


def get_transform(inference_config: InferenceConfig, config: ModelConfig):
    # do nothing
    return Compose([])
