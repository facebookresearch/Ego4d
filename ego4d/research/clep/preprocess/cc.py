import functools
import os

from multiprocessing import Pool
from typing import List, Tuple

import h5py

import pandas as pd

import torch
import torchvision.transforms as T
from ego4d.features.config import FeatureExtractConfig, load_model
from ego4d.research.clep.config import CCPreprocessConfig, TrainConfig
from ego4d.research.clep.dataset import create_data_loader
from ego4d.research.clep.preprocess.common import get_language_model
from ego4d.research.common import batch_it, create_executor
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm


class ImagePathDset(torch.utils.data.Dataset):
    def __init__(self, paths, transform):
        super().__init__()
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return {
            "img": img,
            "path": path,
        }


def preprocess_cc(config: TrainConfig, cc: CCPreprocessConfig):
    out_dir = config.pre_config.root_dir
    os.makedirs(out_dir, exist_ok=True)

    train_df = pd.read_csv(cc.in_path, sep="\t")

    examples = dict(zip(train_df.filepath, train_df.title))
    invalid_keys = []
    with Pool(cc.helper_workers) as pool:
        for path, x in tqdm(pool.imap(_get_fs, examples), total=len(examples)):
            if x == 0:
                invalid_keys.append(path)
    invalid_keys = set(invalid_keys)
    print(len(invalid_keys))
    examples = {k: v for k, v in examples.items() if k not in invalid_keys}

    feature_extract_config = OmegaConf.load(
        config.input_config.feature_extract_config_path
    )
    feature_extract_config.model_config.input_type = "image"  # not actually needed

    all_ex = list(examples.items())
    batches = batch_it(all_ex, cc.imgs_per_gpu)
    batches = [(idx, batch) for idx, batch in enumerate(batches)]

    # save off the keys to the dataset
    torch.save(
        [_get_key(p, idx) for idx, batch in batches for p, _ in batch], cc.meta_path
    )

    executor = create_executor(config.pre_config.slurm_config, len(batches))

    jobs = executor.map_array(
        functools.partial(
            _map_cc_batch,
            cc=cc,
            feature_extract_config=feature_extract_config,
        ),
        batches,
    )

    viz_path = os.path.join(out_dir, cc.hdf5_viz_path)
    with h5py.File(viz_path, "w") as out_f:
        for job in tqdm(jobs):
            res = job.result()
            for k, v in res.items():
                out_f.create_dataset(k, data=v)

    print("converting captions")
    jobs = executor.map_array(
        functools.partial(_map_cc_sent_batch, config=config, cc=cc),
        batches,
    )
    sent_path = os.path.join(out_dir, cc.hdf5_sent_path)
    with h5py.File(sent_path, "w") as out_f:
        for job in tqdm(jobs):
            res = job.result()
            for k, v in res.items():
                out_f.create_dataset(k, data=v)


def _get_fs(x):
    return x, os.path.getsize(x)


def _map_cc_batch(
    batch_idx, cc: CCPreprocessConfig, feature_extract_config: FeatureExtractConfig
):
    idx, batch = batch_idx
    paths = [x for x, _ in batch]
    viz_model = load_model(feature_extract_config, patch_final_layer=True)

    image_transform = T.Compose(
        [
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.Lambda(lambda x: x.unsqueeze(0).permute(1, 0, 2, 3)),
        ]
    )

    dset = ImagePathDset(paths, image_transform)
    dloader = create_data_loader(dset, cc, shuffle=False)  # pyre-ignore
    viz_model.eval()

    ret = {}
    with torch.no_grad():
        for xx in tqdm(dloader, total=len(dloader)):
            fv = viz_model({"video": xx["img"].cuda()})
            for p, f in zip(xx["path"], fv.cpu().numpy()):
                ret[_get_key(p, idx)] = f
    return ret


def _map_cc_sent_batch(
    batch_idx: Tuple[int, List[str]], config: TrainConfig, cc: CCPreprocessConfig
):
    idx, batch = batch_idx
    paths = [x for x, _ in batch]
    cap = [x for _, x in batch]

    sent_model = get_language_model(config)
    fvs = sent_model.encode(cap, device="cuda", show_progress_bar=True)
    ret = {}
    for p, f in zip(paths, fvs):
        ret[_get_key(p, idx)] = f
    return ret


def _get_key(p, idx):
    pn = p.replace("/", "_")
    return f"{idx}/{pn}"
