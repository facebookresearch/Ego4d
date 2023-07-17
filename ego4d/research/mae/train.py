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
import omnimae.omni_mae_model as omnimae

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
        x = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        self.log("test_loss", test_loss)

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
    # TODO
    # for split in [("train", 0.7)

    print("Creating dataset")
    t1 = time.time()
    video_class = PyAvReader
    video_class_kwargs={
        "mean": None,  # TODO: config
        "resize": 256,
        "frame_window_size": 1,  # TODO: config
        "stride": 1,  # TODO: config
        "gpu_idx": -1,  # TODO: config
    }
    if config.reader_class == "TorchAudioStreamReader":
        video_class = TorchAudioStreamReader
        video_class_kwargs={
            "mean": None,  # TODO: config
            "resize": 256,
            "frame_window_size": 1,  # TODO: config
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
    base_model = getattr(omnimae, config.model_config.model_name)()
    dset = get_dataset(config.input_config)
    train_loader = create_data_loader(dset, config)
    val_loader = train_loader  # TODO

    print("loading warmup batch")
    t1 = time.time()
    for x in train_loader:
        break
    t2 = time.time()
    print(f"Loaded: {t2-t1:.3f}s")
    t1 = time.time()
    for x in train_loader:
        break
    t2 = time.time()
    print(f"Loaded: {t2-t1:.3f}s")

    mae = EgoMae(config, base_model)
    # trainer = pl.Trainer(gpus=8, num_nodes=4, accelerator='ddp')
    trainer = pl.Trainer(limit_train_batches=100, max_epochs=1)
    trainer.fit(model=mae, train_dataloaders=train_loader)



if __name__ == "__main__":
    train()
