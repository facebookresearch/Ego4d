import os
import gc
import copy
import torch
import math
import hydra
import time
import submitit
import h5py

from ego4d.vaclip.val import (
    eval_classification,
    eval_multi_class_classification,
)
from ego4d.vaclip.dataset import (
    create_data_loader,
)
from ego4d.vaclip.model import ResBlock, _get_layers_proj_dims

from pytorch_lightning.lite import LightningLite

# TODO import from open_clip
import torch.nn as nn
import torch.distributed.nn
from torch import distributed as dist, nn as nn
from torch.nn import functional as F
from open_clip.loss import ClipLoss
import torch.nn.functional as F

from tqdm.auto import tqdm

from torch.utils.tensorboard import SummaryWriter

from dataclasses import dataclass


@dataclass
class TrainConfig:
    accelerator: str
    devices: int
    batch_size: int
    num_workers: int
    prefetch_factor: int
    num_epochs: int

    run_locally: bool
    wd: float
    lr: float
    beta1: float
    beta2: float
    eps: float

    feature_path: str


@dataclass
class FeatureDataloader:
    def __init__(self, config):
        self.features = h5py.File(config.feature_path)
        self.uid_idx = [
            (key, idx)
            for key in self.features
            for idx in range(len(self.features[key]) - 1)
        ]

    def __len__(self):
        return len(self.uid_idx)

    def __getitem__(self, idx):
        uid, idx = self.uid_idx[idx]
        return self.features[uid][idx], self.features[uid][idx + 1]


class UnsupModel(nn.Module):
    def __init__(self):
        super().__init__()

        # TODO: configure
        proj_dims = [1536, 1536, 1536, 1536, 1536, 1536]
        layers = _get_layers_proj_dims(1536, proj_dims, 1536)
        self.proj = nn.Sequential(*layers)

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(
                module.weight.data,
                gain=torch.nn.init.calculate_gain('relu'),
            )
            module.bias.data.zero_()

    def forward(self, x):
        return self.proj(x)


class Lite(LightningLite):
    def my_setup(self, config: TrainConfig):
        self.model = UnsupModel()
        self.config = config

    def run(self):
        self.optimizer = torch.optim.AdamW(
              [
                  {"params": self.model.parameters(), "weight_decay": self.config.wd},
              ],
              lr=self.config.lr,
              betas=(self.config.beta1, self.config.beta2),
              eps=self.config.eps,
        )

        dset = FeatureDataloader(config=self.config)
        dataloader = create_data_loader(dset, self.config)
        dataloader = self.setup_dataloaders(dataloader)
        model, optimizer = self.setup(self.model, self.optimizer)

        model.train()
        # step = 0
        # num_examples = 0
        # max_steps = 2*len(dataloader)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=0, last_epoch=-1)

        log_dir = os.path.join(self.config.tb_log_dir, self.config.tb_log_name)
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        print(f"log_dir={log_dir}")

        loss = torch.nn.MSELoss()
        num_examples = 0
        for epoch in range(self.config.num_epochs):
            print("Epoch:", epoch)
            for x, gt in tqdm(dataloader, total=len(dataloader)):
                optimizer.zero_grad()

                y = model(x)
                l2_loss = loss(y, gt)
                num_examples += x.shape[0]
                writer.add_scalar("Loss", l2_loss.item(), num_examples)

                self.backward(l2_loss)
                optimizer.step()


def run_train(config: TrainConfig):
    lite = Lite(accelerator=config.accelerator, devices=config.devices)
    lite.my_setup(config)
    lite.run()


@hydra.main(config_path="configs", config_name=None)
def train_model(config: TrainConfig):
    print(config)
    if not config.run_locally:
        executor = submitit.AutoExecutor(folder=config.slurm_log_folder)
        executor.update_parameters(
            timeout_min=1200,
            constraint="volta",
            slurm_partition="pixar",
            gpus_per_node=1,
            cpus_per_task=10,
        )
        job_id = executor.submit(run_train, config)
        print(job_id.job_id)
    else:
        run_train(config)


if __name__ == "__main__":
    train_model()  # pyre-ignore
