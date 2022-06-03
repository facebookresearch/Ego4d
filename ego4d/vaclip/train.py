import os
import gc
import torch
import math
import hydra
import submitit

from ego4d.vaclip.config import TrainConfig
from ego4d.vaclip.dataset import Ego4DVaClip, KineticsDset, create_data_loader
from ego4d.vaclip.model import EgoLangaugeAssociation

from pytorch_lightning.lite import LightningLite

# TODO import from open_clip
import torch.nn as nn
import torch.distributed.nn
from torch import distributed as dist, nn as nn
from torch.nn import functional as F
from open_clip.loss import ClipLoss
import torch.nn.functional as F
# from open_clip.training.scheduler import scheduler

from tqdm.auto import tqdm

from torch.utils.tensorboard import SummaryWriter


# taken from: https://github.com/mlfoundations/open_clip/blob/main/src/training/zero_shot.py#L29
def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


class Lite(LightningLite):
    def my_setup(self, config: TrainConfig):
        self.model = EgoLangaugeAssociation(config.model_config)
        self.config = config

    def run(self):
        named_parameters = list(self.model.named_parameters())

        # https://github.com/mlfoundations/open_clip/blob/main/src/training/main.py#L163
        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        include = lambda n, p: not exclude(n, p)
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
        self.optimizer = torch.optim.AdamW(
              [
                  {"params": gain_or_bias_params, "weight_decay": 0.},
                  {"params": rest_params, "weight_decay": self.config.wd},
              ],
              lr=self.config.lr,
              betas=(self.config.beta1, self.config.beta2),
              eps=self.config.eps,
        )
        model, optimizer = self.setup(self.model, self.optimizer)

        print("Config=")
        print(self.config, flush=True)

        dset = Ego4DVaClip(config=self.config)
        dataloader = create_data_loader(dset, self.config)
        dataloader = self.setup_dataloaders(dataloader)
        clip_loss = ClipLoss()

        log_dir = os.path.join(self.config.tb_log_dir, self.config.tb_log_name)
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

        model.train()
        step = 0
        for epoch in range(self.config.num_epochs):
            print("Epoch:", epoch)
            for batch in tqdm(dataloader, total=len(dataloader)):
                optimizer.zero_grad()
                v_f, t_f, logit_scale = self.model(batch)
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

                if step % 100 == 0:
                    gc.collect()

                if step % self.config.eval_per_iter == 0:
                    print(f"Eval {step} - {self.config.eval_per_iter}", flush=True)
                    acc1, acc5 = self.run_eval()
                    print(f"acc1={acc1}, acc5={acc5}")
                    writer.add_scalar("Val/Acc 1", acc1)
                    writer.add_scalar("Val/Acc 5", acc5)

                step += 1


    def run_eval(self):
        self.model.eval()

        dset = KineticsDset(self.config)
        val_loader = create_data_loader(dset, self.config)
        val_loader = self.setup_dataloaders(val_loader)

        classifier = F.normalize(
            self.model.text_proj(dset.sent_ordered.to(self.device)).t(),
            dim=-1,
        )

        with torch.no_grad():
            acc1, acc5, n = 0.0, 0.0, 0
            for x, target in tqdm(val_loader):
                v = self.model.visual_proj(x)
                v = F.normalize(v, dim=-1)

                # https://github.com/mlfoundations/open_clip/blob/main/src/training/zero_shot.py#L49
                logits = v @ classifier
                acc1, acc5 = accuracy(logits, target, topk=(1, 5))
                n += x.shape[0]

        acc1 /= n
        acc5 /= n
        self.model.train()
        return 100.0 * acc1, 100.0 * acc5


def run_train(config: TrainConfig):
    lite = Lite(accelerator=config.accelerator, devices=config.devices)
    lite.my_setup(config)
    lite.run()


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
