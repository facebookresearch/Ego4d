import os
import gc
import copy
import torch
import math
import hydra
import time
import submitit

from ego4d.vaclip.val import (
    eval_classification,
    eval_multi_class_classification,
)
from ego4d.vaclip.config import TrainConfig
from ego4d.vaclip.dataset import (
    Ego4DVaClip,
    KineticsDset,
    EgoCharadesDset,
    create_data_loader,
)
from ego4d.vaclip.model import EgoLangaugeAssociation

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


class Lite(LightningLite):
    def my_setup(self, config: TrainConfig):
        self.model = EgoLangaugeAssociation(config.model_config)
        self.config = config
        self.val_config = copy.deepcopy(config)
        self.val_config.batch_size = 1

        self.val_bs_config = copy.deepcopy(config)
        self.val_bs_config.batch_size = 128

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

        log_dir = os.path.join(self.config.tb_log_dir, self.config.tb_log_name)
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

        model.train()
        step = 0
        num_examples = 0

        max_steps = 2*len(dataloader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=0, last_epoch=-1)

        bce_loss = torch.nn.BCEWithLogitsLoss()

        for epoch in range(self.config.num_epochs):
            print("Epoch:", epoch)
            for batch in tqdm(dataloader, total=len(dataloader)):
                if step % self.config.eval_per_iter == 0 and (step > 0 or self.config.eval_init):
                    print()
                    if step > 0:
                        print("Loss=", loss.cpu().item(), flush=True)
                    print(f"Eval {step} - {self.config.eval_per_iter}", flush=True)
                    kv_pairs = self.run_eval()
                    for k, v in kv_pairs.items():
                        print(k, v)
                        writer.add_scalar(k, v, num_examples)
                    print()

                optimizer.zero_grad()
                v_f, t_f, logit_scale = self.model(batch)

                txt = batch["text_no_tag"]

                device = txt.device
                with torch.no_grad():
                    if self.config.use_soft_loss is not None:
                        simm = txt @ txt.t()
                        simm = (simm + 1) / 2
                        if self.config.use_soft_loss:
                            label = simm
                        else:
                            label = torch.ones_like(simm, dtype=torch.float)
                        # TODO: nn.Parameter for the threshold?
                        label[simm < self.config.soft_loss_threshold] = 0.0
                        pos_ex_prop = label.sum() / (label.shape[0]*label.shape[0])
                    else:
                        label = torch.arange(txt.shape[0], device=device)
                        pos_ex_prop = None

                if self.config.norm_logits:
                    v_f = F.normalize(v_f, dim=-1)
                    t_f = F.normalize(t_f, dim=-1)

                if self.config.use_logit_scale:
                    vid2txt = logit_scale * v_f @ t_f.T
                    txt2vid = logit_scale * t_f @ v_f.T
                else:
                    vid2txt = v_f @ t_f.T
                    txt2vid = t_f @ v_f.T

                if self.config.use_bce:
                    assert self.config.use_bce != False  # noqa
                    loss = (
                        bce_loss(vid2txt, label) +
                        bce_loss(txt2vid, label)
                    ) / 2.0
                else:
                    loss = (
                        F.cross_entropy(vid2txt, label) +
                        F.cross_entropy(txt2vid, label)
                    ) / 2.0

                self.backward(loss)  # instead of loss.backward()
                optimizer.step()
                scheduler.step()

                # https://github.com/mlfoundations/open_clip/blob/main/src/training/train.py#L100
                with torch.no_grad():
                    model.module.logit_scale.clamp_(0, math.log(100))

                num_examples += batch["video"].shape[0]

                writer.add_scalar("Loss/train", loss.detach().cpu(), num_examples)
                writer.add_scalar("logit_scale", model.module.logit_scale.detach().cpu(), num_examples)
                writer.add_scalar("lr", optimizer.param_groups[0]['lr'], num_examples)
                if pos_ex_prop is not None:
                    writer.add_scalar("pos_ex_prop", pos_ex_prop, num_examples)
                writer.flush()

                if step % 100 == 0:
                    gc.collect()

                step += 1

    def _run_ego_charades(self, ego_only=True, use_ego_sent=True):
        dset = EgoCharadesDset(self.config, use_ego_sent=use_ego_sent, ego_only=ego_only)
        val_loader = create_data_loader(dset, self.val_config)
        val_loader = self.setup_dataloaders(val_loader)

        classifier = F.normalize(
            self.model.text_proj(dset.sent_ordered.to(self.device)).t(),
            dim=-1,
        )
        res = eval_multi_class_classification(self.model.visual_proj, classifier, val_loader)
        ret = {}
        for k, v in res.items():
            char_name = "Char_"
            if ego_only is not None:
                char_name += f"{ego_only:d}Ego_"
            else:
                char_name += "All_"

            if use_ego_sent is not None:
                char_name += f"{use_ego_sent:d}EgoSent"
            else:
                char_name += "Labels"
            ret[f"Val/{char_name}/{k}"] = v
        return ret

    def _run_eval_kinetics(self):
        dset = KineticsDset(self.config)
        val_loader = create_data_loader(dset, self.val_bs_config)
        val_loader = self.setup_dataloaders(val_loader)

        classifier = F.normalize(
            self.model.text_proj(dset.sent_ordered.to(self.device)).t(),
            dim=-1,
        )
        res = eval_classification(self.model.visual_proj, classifier, val_loader, avg_logits=False)
        return {
            "Val/Kinetics/acc1": res["acc1"],
            "Val/Kinetics/acc5": res["acc5"],
        }

    def run_eval(self):
        self.model.eval()
        result = {}
        result.update(self._run_ego_charades(ego_only=None, use_ego_sent=None))
        result.update(self._run_ego_charades(ego_only=None, use_ego_sent=True))
        result.update(self._run_ego_charades(ego_only=False, use_ego_sent=None))
        result.update(self._run_ego_charades(ego_only=True, use_ego_sent=None))
        result.update(self._run_ego_charades(ego_only=True, use_ego_sent=False))
        result.update(self._run_ego_charades(ego_only=True, use_ego_sent=True))
        result.update(self._run_eval_kinetics())
        self.model.train()
        return result


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
        # _ = job_id.result()
    else:
        run_train(config)


if __name__ == "__main__":
    train_model()  # pyre-ignore
