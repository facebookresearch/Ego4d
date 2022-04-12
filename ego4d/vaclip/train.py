import torch
import hydra

from ego4d.vaclip.config import TrainConfig
from ego4d.vaclip.dataset import Ego4DVaClip, create_data_loader, get_transform
from ego4d.vaclip.model import EgoLangaugeAssociation

from pytorch_lightning.lite import LightningLite

# TODO import from open_clip
import torch.nn as nn
import torch.distributed.nn
from torch import distributed as dist, nn as nn
from torch.nn import functional as F
from open_clip.loss import ClipLoss


class Lite(LightningLite):
    def run(self, config: TrainConfig):
        model = EgoLangaugeAssociation(config.model_config)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        model, optimizer = self.setup(model, optimizer)  # Scale your model / optimizers

        transform = get_transform(config.transform_config)
        dset = Ego4DVaClip(
            # TODO
            uid_subset=[
                "001e3e4e-2743-47fc-8564-d5efd11f9e90",
                "002c3b5c-ed86-4af3-99a1-4b497b7c8a86",
                "0031d268-818c-4ec4-a804-935be610a61a",
            ],
            config=config.input_config,
            transform=transform,
        )
        dataloader = create_data_loader(dset, config)
        dataloader = self.setup_dataloaders(dataloader)
        clip_loss = ClipLoss()

        model.train()
        for _ in range(config.num_epochs):
            for batch in dataloader:
                optimizer.zero_grad()
                v_f, t_f, logit_scale = model(batch)
                loss = clip_loss(v_f, t_f, logit_scale)
                print("Loss", loss)
                self.backward(loss)  # instead of loss.backward()
                optimizer.step()


@hydra.main(config_path="configs", config_name=None)
def train_model(config: TrainConfig):
    lite = Lite(accelerator=config.accelerator, devices=config.devices)
    lite.run(config)


if __name__ == "__main__":
    train_model()  # pyre-ignore
