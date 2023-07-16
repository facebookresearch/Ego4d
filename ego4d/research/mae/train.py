import os
import copy
import hydra
import submitit
import torch
import torch.optim as optim

import lightning.pytorch as pl
import omnimae.omni_mae_model as omnimae

from torch.utils.data import DataLoader
from ego4d.research.mae.config import TrainConfig
from torch.utils.tensorboard import SummaryWriter

def create_data_loader(dset, config: TrainConfig, shuffle=True):
    return DataLoader(
        dset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        shuffle=shuffle,
    )


class EgoMae(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        self.log("test_loss", test_loss)


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


@hydra.main(config_path="configs", config_name=None)
def train(config: TrainConfig):
    base_model = getattr(omnimae, config.model_config.model_name)()
    dataset = None
    train_loader = create_data_loader(dset, config)

    import IPython; IPython.embed()
    pass

if __name__ == "__main__":
    train()
