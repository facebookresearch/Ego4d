import os
import gc
import torch
import math
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
# from open_clip.training.scheduler import scheduler

from tqdm.auto import tqdm

from torch.utils.tensorboard import SummaryWriter



class Lite(LightningLite):
    def run(self, config: TrainConfig):
        print("Config=")
        print(config, flush=True)

        model = EgoLangaugeAssociation(config.model_config)
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        named_parameters = list(model.named_parameters())

        # https://github.com/mlfoundations/open_clip/blob/main/src/training/main.py#L163
        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        include = lambda n, p: not exclude(n, p)
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

        optimizer = torch.optim.AdamW(
              [
                  {"params": gain_or_bias_params, "weight_decay": 0.},
                  {"params": rest_params, "weight_decay": config.wd},
              ],
              lr=config.lr,
              betas=(config.beta1, config.beta2),
              eps=config.eps,
        )

        model, optimizer = self.setup(model, optimizer)

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

                # https://github.com/mlfoundations/open_clip/blob/main/src/training/train.py#L100
                with torch.no_grad():
                    model.module.logit_scale.clamp_(0, math.log(100))

                writer.add_scalar("Loss/train", loss.detach().cpu(), step)
                writer.add_scalar("logit_scale", model.module.logit_scale.detach().cpu(), step)
                writer.add_scalar("lr", optimizer.param_groups[0]['lr'], step)
                writer.flush()
                step += 1

                if step % 100 == 0:
                    gc.collect()



def run_train(config: TrainConfig):
    lite = Lite(accelerator=config.accelerator, devices=config.devices)
    lite.run(config)


@hydra.main(config_path="configs", config_name=None)
def train_model(config: TrainConfig):
    if not config.run_locally:
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
    else:
        run_train(config)


if __name__ == "__main__":
    train_model()  # pyre-ignore
