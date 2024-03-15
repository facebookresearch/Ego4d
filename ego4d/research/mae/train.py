import time
import json
import os
import copy
import hydra
import submitit
import torch
import torch.nn.functional as F
import torch.optim as optim

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger

from torch.utils.data import DataLoader
from ego4d.research.mae.config import (
    TrainConfig,
    InputConfig,
)
from ego4d.research.dataset import VideoDataset
from ego4d.research.readers import (
    PyAvReader,
    TorchAudioStreamReader,
)
from torch.utils.tensorboard import SummaryWriter
from maws.model_builder import build_model
from ego4d.research.mae.decoder import mae_vit_base_patch16


def create_data_loader(dset, config: TrainConfig, shuffle=True):
    num_workers = config.num_workers
    prefetch_factor = config.prefetch_factor
    if config.input_config.reader_class == "TorchAudioStreamReader":
        if prefetch_factor is not None or num_workers != 0:
            print("WARN: torchaudio does not support num_workers>0 or prefetch_factor not None (for now). Setting to 0 and None respectively.")
            prefetch_factor = None
            num_workers = 0

    return DataLoader(
        dset,
        batch_size=config.batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        shuffle=shuffle,
    )


class EgoMae(pl.LightningModule):
    def __init__(self, config: TrainConfig, model):
        super().__init__()
        self.model = model
        self.config = config

    def training_step(self, batch, batch_idx):
        # opt = self.optimizers()
        # opt.zero_grad()
        img_batch = batch["video"].permute(0, 2, 1, 3, 4).reshape(-1, 3, 224, 224)
        loss, pred, mask = self.model(img_batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss
        # self.manual_backward(loss)
        # opt.step()


    def test_step(self, batch, batch_idx):
        img_batch = batch["video"].permute(0, 2, 1, 3, 4).reshape(-1, 3, 224, 224)
        loss, pred, mask = self.model(img_batch)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        config = self.config
        optimizer = optim.Adam(
            self.parameters(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.wd,
            eps=config.eps,
        )
        return optimizer


def get_dataset(config: InputConfig):
    # TODO: egoexo support

    def video_uid_to_path(video_uid):
        return os.path.join(config.ego4d_input.video_root_dir, f"{video_uid}.mp4") 

    ego4d_meta = json.load(open(config.ego4d_input.metadata_path))
    paths_to_n_frames = {
        video_uid_to_path(x["video_uid"]): x["video_metadata"]["num_frames"]
        for x in ego4d_meta["videos"]
    }
    paths = list(paths_to_n_frames.keys())
    paths = paths[0:1]

    print("Creating dataset")
    t1 = time.time()
    video_class = PyAvReader
    video_class_kwargs={
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "crop": 224,
        "resize": 256,
        "frame_window_size": 4,  # TODO: config
        "stride": 1,  # TODO: config
        "gpu_idx": -1,  # TODO: config
    }
    if config.reader_class == "TorchAudioStreamReader":
        video_class = TorchAudioStreamReader
        video_class_kwargs={
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            # TODO: fixme
            "crop": None,
            "resize": 224,
            "frame_window_size": 4,  # TODO: config
            "stride": 1,  # TODO: config
            "gpu_idx": 0,  # TODO: config
        }
    else:
        assert config.reader_class == "PyAvReader"

    dset = VideoDataset(
        paths=paths, 
        video_class=video_class,
        video_class_kwargs=video_class_kwargs,
        max_num_frames_per_video=None,
        paths_to_n_frames=paths_to_n_frames,
        with_pbar=True,
    )
    t2 = time.time()
    print(f"Took: {t2 - t1:.3f}s")
    return dset


@hydra.main(config_path="configs", config_name=None)
def train(config: TrainConfig):
    # base_model = build_model("vit_2b14_xlmr_l", "maws_clip")
    # base_model = build_model("vit_2b14_xlmr_l", "maws_clip")
    base_model = mae_vit_base_patch16()
    dset = get_dataset(config.input_config)
    train_loader = create_data_loader(dset, config)
    val_loader = train_loader  # TODO

    # print("loading warmup batch")
    # t1 = time.time()
    # for x in train_loader:
    #     break
    # t2 = time.time()
    # print(f"Loaded: {t2-t1:.3f}s")

    mae = EgoMae(config, base_model)
    logger = TensorBoardLogger("tb_logs", name="mae_vit_base_patch16")
    trainer = pl.Trainer(
        logger=logger,
        val_check_interval=0.1,
        max_epochs=1,
        strategy="ddp",
        num_nodes=1,
        devices=1,
    )
    trainer.fit(model=mae, train_dataloaders=train_loader)


if __name__ == "__main__":
    train()
