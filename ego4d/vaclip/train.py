import os
import torch
import hydra
import submitit

from ego4d.vaclip.config import TrainConfig
from ego4d.vaclip.dataset import Ego4DVaClip, create_data_loader
from ego4d.vaclip.model import EgoLangaugeAssociation

from pytorch_lightning.lite import LightningLite

# TODO import from open_clip
import torch.nn as nn
import torch.distributed.nn
from torch import distributed as dist, nn as nn
from torch.nn import functional as F
from open_clip.loss import ClipLoss
from tqdm.auto import tqdm

from torch.utils.tensorboard import SummaryWriter



class Lite(LightningLite):
    def run(self, config: TrainConfig):
        model = EgoLangaugeAssociation(config.model_config)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        model, optimizer = self.setup(model, optimizer)  # Scale your model / optimizers

        dset = Ego4DVaClip(config=config)
        dataloader = create_data_loader(dset, config)
        dataloader = self.setup_dataloaders(dataloader)
        clip_loss = ClipLoss()

        log_dir = os.path.join(config.tb_log_dir, config.tb_log_name)
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

        model.train()
        step = 0
        for epoch in range(config.num_epochs):
            print("Epoch:", epoch)
            for batch in tqdm(dataloader, total=len(dataloader)):
                optimizer.zero_grad()
                v_f, t_f, logit_scale = model(batch)
                loss = clip_loss(v_f, t_f, logit_scale)
                self.backward(loss)  # instead of loss.backward()
                optimizer.step()
                writer.add_scalar("Loss/train", loss.detach().cpu(), step)
                writer.flush()
                step += 1


def run_train(config: TrainConfig):
    lite = Lite(accelerator=config.accelerator, devices=config.devices)
    lite.run(config)


@hydra.main(config_path="configs", config_name=None)
def train_model(config: TrainConfig):
    executor = submitit.AutoExecutor(folder=config.slurm_log_folder)
    executor.update_parameters(
        timeout_min=720,
        constraint="volta",
        slurm_partition="pixar",
        gpus_per_node=1,
        cpus_per_task=10,
    )
    job_id = executor.submit(run_train, config)
    print(job_id.job_id)
    _ = job_id.result()


if __name__ == "__main__":
    train_model()  # pyre-ignore
