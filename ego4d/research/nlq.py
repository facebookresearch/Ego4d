import json
import os
import gc
import torch
import hydra

import submitit
import h5py
from dataclasses import dataclass


import torch.nn.functional as F
from ego4d.research.clep.dataset import get_start_end_idx
from torch.utils.data import Dataset

from pytorch_lightning.lite import LightningLite

from torch.utils.tensorboard import SummaryWriter

from ego4d.research.common import SlurmConfig


@dataclass
class InputConfig:
    annotation_dir: str
    features_hdf5_path: str


@dataclass
class TrainConfig:
    input_config: InputConfig

    lr: float
    checkpoint_dir: str
    checkpoint_metric: str
    batch_size: int
    num_workers: int
    prefetch_factor: int

    num_epochs: int
    accelerator: str
    devices: int

    run_locally: bool
    tb_log_dir: str
    tb_log_name: str

    lr: float
    beta1: float
    beta2: float
    wd: float
    eps: float

    eval_per_iter: int
    eval_init: bool

    slurm_log_folder: str

def get_anns(ann_json):
    anns = []
    for vid in ann_json["videos"]:
        for clip in vid["clips"]:
            for ann in clip["annotations"]:
                for query in ann["language_queries"]:
                    anns.append({
                        "video_uid": vid["video_uid"],
                        "query_start_time_sec": clip["video_start_sec"],
                        "query_end_time_sec": clip["video_end_sec"],
                        "query_response_start_time_sec": query["video_start_sec"],
                        "query_response_end_time_sec": query["video_end_sec"],
                        "query_template": query.get("template", None),
                        "query": query.get("query", None),
                    })
    return anns

def augment_ann(ann):
    # TODO
    return ann

class NlqFeatureDloader(Dataset):
    def __init__(self, nlq_json, features):
        self.anns = get_anns(nlq_json)
    
    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        ann = self.anns[idx]

        ann = train_anns[0]
        uid = ann["video_uid"]
        feat = self.features[uid]
        feat = features[uid]

        i1, i2 = get_start_end_idx(ann["query_start_time_sec"], ann["query_end_time_sec"], feature_per_sec=30/16, nf=len(feat))
        n = i2 - i1

        B = 2
        N = 2
        scales = {}
        for x in (B ** (torch.arange(N) + 1)):
            idx = 0
            row = []
            inc = int(np.ceil(n / x))
            while idx < n:
                row.append(
                    torch.tensor(
                        feat[idx + i1 : idx + n // x + i1].mean(0)
                    )
                )
                idx += inc
            print(x, [xx.shape for xx in row])
        return ann
        

class Lite(LightningLite):
    def my_setup(self, config: TrainConfig):
        self.config = config

        a_dir = config.input_config.annotation_dir
        train_json_path = os.path.join(a_dir, "nlq_train.json")
        val_json_path = os.path.join(a_dir, "nlq_val.json")
        train_json = json.load(open(train_json_path))
        val_json = json.load(open(val_json_path))

        self.features = h5py.File(config.features_hdf5_path)
        self.train_dset = NlqFeatureDloader(train_json, self.features)
        self.val_dset = NlqFeatureDloader(val_json, self.features)
        config


    def run(self):
        pass


    def run_eval(self):
        self.model.eval()
        result = {}
        # TODO
        # result.update(self._run_eval_kinetics())
        # result.update(self._run_ego_charades(ego_only=None, use_ego_sent=None))
        # result.update(self._run_ego_charades(ego_only=None, use_ego_sent=True))
        # result.update(self._run_ego_charades(ego_only=False, use_ego_sent=None))
        # result.update(self._run_ego_charades(ego_only=True, use_ego_sent=None))
        # result.update(self._run_ego_charades(ego_only=True, use_ego_sent=False))
        # result.update(self._run_ego_charades(ego_only=True, use_ego_sent=True))
        self.model.train()
        return result


def run_train(config: TrainConfig):
    lite = Lite(accelerator=config.accelerator, devices=config.devices)
    lite.my_setup(config)
    lite.run()


@hydra.main(config_path="configs", config_name="nlq")
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
