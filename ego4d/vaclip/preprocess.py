import sys
import os
import copy
import json
import random
import functools
import math
import submitit
import logging
from typing import List, Any
from multiprocessing import Pool
from pathlib import Path
from collections import defaultdict

import h5py
import pandas as pd
import torch
import hydra
import numpy as np
import torchvision.transforms as T
from PIL import Image

from sentence_transformers import SentenceTransformer
from ego4d.vaclip.dataset import (
    create_data_loader,
)
from ego4d.vaclip.config import (
    EgoPreprocessFeatureConfig,
    TrainConfig,
    CCPreprocessConfig,
    PreprocessConfig,
    EgoPreprocessNarrConfig,
    K400PreprocessConfig,
    EgoCharadePreprocessConfig,
)
from ego4d.features.inference import _load_kinetics_class_names
from tqdm.auto import tqdm
from hydra import compose, initialize
from omegaconf import OmegaConf


from ego4d.features.config import (
    FeatureExtractConfig,
    Video,
    load_model,
)
from ego4d.features.extract_features import extract_features
from ego4d.features.inference import (
    _video_info as video_info,
)


def batch_it(things: List[Any], batch_size: int) -> List[List[Any]]:
    num_batches: int = math.ceil(len(things) / batch_size)

    result = []
    for i in range(num_batches):
        result.append(things[i * batch_size : (i + 1) * batch_size])
    return result


def remove_tags(text: str) -> str:
    text = text.replace("# C", "")
    text = text.replace("#C", "")
    text = text.replace("C", "")
    text = text.replace("#O", "")
    text = text.replace("O", "")
    text = text.replace("#unsure", "")
    return text


def sub_tagged_tokens(text: str) -> str:
    text = text.replace("#C", "Camera wearer")
    text = text.replace("#O", "Other person")
    text = text.replace("#unsure", "something")
    return text


def filter_narrations(narrations):
    # TODO: config
    meta = json.load(open("/checkpoint/miguelmartin/ego4d_data/ego4d.json"))
    val_set_uids = [
        vid["video_uid"]
        for vid in meta["videos"]
        if len({vid["split_em"], vid["split_fho"], vid["split_av"]} & {"val", "test"}) > 0
    ]

    num_val_filtered = 0
    num_txt_filtered = 0
    ret = []
    for uid, txt, ts in tqdm(narrations):
        if uid in val_set_uids:
            num_val_filtered += 1
            continue
        if len(txt.split(" ")) <= 3:
            num_txt_filtered += 1
            continue
        ret.append((uid, txt, ts))
    print(f"Narrations filtered from {len(narrations)} -> {len(ret)}")
    print(f"""
    Val Filtered = {num_val_filtered} = {num_val_filtered/len(narrations):.2%}
    Txt Filtered = {num_txt_filtered} = {num_txt_filtered/len(narrations):.2%}

    """)
    return ret


def get_narrations(config: EgoPreprocessNarrConfig):
    narration_json = json.load(open(config.narration_json_path))
    uid_subset = set(narration_json.keys())
    narrations = [
        (uid, data["narration_text"], data["timestamp_sec"])
        for uid in uid_subset
        for data in narration_json[uid].get("narration_pass_1", {"narrations": []})["narrations"]
    ]
    narrations += [
        (uid, data["narration_text"], data["timestamp_sec"])
        for uid in uid_subset
        for data in narration_json[uid].get("narration_pass_2", {"narrations": []})["narrations"]
    ]
    narrations.sort(key=lambda x: (x[0], x[-1]))
    return narrations


def map_narrs_on_machine(narrs, config=None):
    model = SentenceTransformer(config.st_model_name)

    narr_op = os.path.join(config.pre_root_dir, config.narration_out_path)
    batches = batch_it(narrs, config.batch_size)

    metas = []
    for batch in tqdm(batches):
        fvs = model.encode(
            [x for _, (_, x, _, _, _) in batch],
            device=config.accelerator,
            show_progress_bar=False,
        )
        fvs_without_tags = model.encode(
            [x for _, (x, _, _, _, _) in batch],
            device=config.accelerator,
            show_progress_bar=False,
        )

        for fv_no_tag, fv, (idx, (no_tag_txt, post_txt, uid, txt, ts)) in zip(fvs_without_tags, fvs, batch):
            od = os.path.join(narr_op, uid)
            os.makedirs(od, exist_ok=True)
            path_to_encode = os.path.join(od, f"{idx}.pt")
            torch.save({
                "fv": fv,
                "fv_no_tag": fv_no_tag,
            }, path_to_encode)
            metas.append({"uid": uid, "txt": txt, "ts": ts, "idx": idx, "post_txt": post_txt, "no_tag_txt": no_tag_txt})
    return metas


def create_executor(config, num_batches: int):
    executor = submitit.AutoExecutor(folder=config.slurm_log_folder)

    executor.update_parameters(
        timeout_min=config.timeout_min,
        slurm_constraint=config.constraint,
        slurm_partition=config.slurm_partition,
        slurm_array_parallelism=min(config.slurm_array_parallelism, num_batches),
        gpus_per_node=config.gpus_per_node,
        cpus_per_task=config.cpus_per_task,
    )
    return executor


def preprocess_ego_features(feature_path: str, pre_config: PreprocessConfig, pre_feature: EgoPreprocessFeatureConfig):
    narr_meta_path = os.path.join(pre_config.ego4d_narr.pre_root_dir, pre_config.ego4d_narr.metadata_out_path)
    narr_meta = torch.load(narr_meta_path)
    video_uids = {x["uid"] for x in narr_meta}
    video_uids = list(video_uids)

    out_path = pre_feature.hdf5_path
    print("=>", out_path, flush=True)
    with h5py.File(out_path, "w") as out_f:
        for uid in tqdm(video_uids, desc="video_uid", leave=True):
            feature_path = os.path.join(feature_path, f"{uid}.pt")
            fv = torch.load(feature_path)
            out_f.create_dataset(uid, data=fv.numpy())


def preprocess_ego_narrations(narr_config: EgoPreprocessNarrConfig):
    os.makedirs(narr_config.pre_root_dir, exist_ok=True)

    narr_op = os.path.join(narr_config.pre_root_dir, narr_config.narration_out_path)
    os.makedirs(narr_op, exist_ok=True)

    narrs = get_narrations(narr_config)
    narrs = filter_narrations(narrs)

    print("Transforming text...")
    narrs_with_idx = list(
        enumerate([
            (remove_tags(txt), sub_tagged_tokens(txt), uid, txt, ts)
            for uid, txt, ts in narrs
        ])
    )
    if narr_config.limit > 0:
        narrs_with_idx = narrs_with_idx[0:narr_config.limit]

    batches = batch_it(narrs_with_idx, narr_config.num_narrs_per_machine)
    print(f"Running txt through transformer with {len(batches)} machines")
    print(f"Num narrs = {len(narrs_with_idx)}", flush=True)

    metas = []
    executor = create_executor(narr_config, len(batches))
    jobs = executor.map_array(
        functools.partial(map_narrs_on_machine, config=narr_config),
        batches,
    )
    print("Jobs", jobs, flush=True)

    for j in tqdm(jobs):
        metas.extend(j.result())

    print("Saving metadata")
    m_op = os.path.join(narr_config.pre_root_dir, narr_config.metadata_out_path)
    torch.save(metas, m_op)


def _extract_features(path, model, feature_extract_config):
    v_info = video_info(path)
    vid = Video(path, path, v_info["num_frames"], w=None, h=None)
    if vid.frame_count is None:
        return None

    feature_extract_config = copy.deepcopy(feature_extract_config)
    feature_extract_config.fps = int(np.round(float(v_info["fps"])))
    feature_extract_config.stride = int(np.round(float(v_info["fps"] * (16/30))))
    feature_extract_config.frame_window = int(np.round(float(v_info["fps"] * (32/30))))

    return extract_features(
        videos=[vid],
        config=feature_extract_config,
        model=model,
        log_info=False,
        silent=True,
        assert_feature_size=False,
    )


def _preprocess_k400_data(video_path_label_pairs, feature_extract_config, viz_dir):
    model = load_model(feature_extract_config, patch_final_layer=True)

    for path, label in tqdm(video_path_label_pairs):
        predictions = _extract_features(path, model, feature_extract_config)
        if predictions is None:
            continue

        name = Path(path).stem
        out_path = os.path.join(viz_dir, f"{name}.pt")
        to_save = {
            "feature": predictions.result[path].mean(0),
            "label": label,
            "all_features": predictions.result[path],
        }
        torch.save(to_save, out_path)


def preprocess_k400_data(config: TrainConfig, k_config: K400PreprocessConfig):
    random.seed(1337)

    pre_dir = os.path.join(k_config.pre_root_dir, k_config.set_to_use)
    viz_dir = os.path.join(pre_dir, k_config.viz_feature_dir)

    os.makedirs(pre_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    # TODO: configure
    val_set = pd.read_csv("/datasets01/kinetics/092121/400/lists/val.csv")

    def process_label(label):
        ret = label.replace('"', '')
        ret = ret.replace("'", '')
        ret = ret.replace("_", ' ')
        return ret

    idx_to_label = list(_load_kinetics_class_names().items())
    idx_to_label.sort(key=lambda x: x[0])
    idx_to_label = dict(idx_to_label)
    assert sorted(idx_to_label.keys()) == list(idx_to_label.keys())

    label_names = list(idx_to_label.values())
    label_to_idx = {v: k for k, v in idx_to_label.items()}
    sentences = [
        f"The person in this video is doing {process_label(label)}"
        for _, label in idx_to_label.items()
    ]
    video_path_label_pairs = [
        (
            # TODO: configure
            f"/datasets01/kinetics/092121/400/val_288px/{row.youtube_id}_{row.time_start:06d}_{row.time_end:06d}.mp4",
            row.label,
        )
        for row in val_set.itertuples()
    ]

    old_len = len(video_path_label_pairs)
    video_path_label_pairs = [val for val in video_path_label_pairs if os.path.exists(val[0])]
    print(f"{old_len} -> {len(video_path_label_pairs)} examples", flush=True)

    feature_extract_config = OmegaConf.load(config.input_config.feature_extract_config_path)
    map_fn = functools.partial(
        _preprocess_k400_data,
        feature_extract_config=feature_extract_config,
        viz_dir=viz_dir
    )
    batches = batch_it(video_path_label_pairs, batch_size=k_config.num_labels_per_machine)

    if config.run_locally:
        for batch in batches:
            map_fn(batch)
    else:
        print(f"To schedule {len(batches)} batches across {k_config.slurm_array_parallelism} machines")
        cont = input("Continue? [y/N]: ")
        if cont != "y":
            print("Exiting...")
            sys.exit(0)
        executor = create_executor(k_config, len(batches))
        jobs = executor.map_array(map_fn, batches)

        # wait for the results
        for job in tqdm(jobs):
            _ = job.result()

    # sentences
    print("Processing labels as sentences", flush=True)
    meta = {
        "label_text": [],
        "label_fv": [],
        "label_name_to_idx": label_to_idx,
        "idx_to_label_name": idx_to_label,
    }
    model = SentenceTransformer(config.pre_config.ego4d_narr.st_model_name)
    for idx, (sent, label_name) in tqdm(enumerate(zip(sentences, label_names)), total=len(sentences)):
        assert label_to_idx[label_name] == idx
        fv = model.encode(sent, show_progress_bar=False)
        meta["label_text"].append(sent)
        meta["label_fv"].append(fv)

    out_meta_path = os.path.join(pre_dir, k_config.metadata_out_path)
    torch.save(meta, out_meta_path)



def _preprocess_ego_charade(video_path_ids, feature_extract_config):
    model = load_model(feature_extract_config, patch_final_layer=True)

    ret = {}
    for path, uid in tqdm(video_path_ids):
        predictions = _extract_features(path, model, feature_extract_config)
        assert predictions is not None
        ret[uid] = {
            "fv": predictions.result[path].mean(0),
            "all_fvs": predictions.result[path],
        }
    return ret


def preprocess_ego_charade(config: TrainConfig, char_config: EgoCharadePreprocessConfig):
    val_set_path = "/datasets01/Charades-ego-v1/101320/charades-ego-v1/CharadesEgo/CharadesEgo_v1_test.csv"
    val_df = pd.read_csv(val_set_path)

    root_path = "/datasets01/Charades-ego-v1/101320/charades-ego-v1/CharadesEgo_v1_480/"
    feature_extract_config = OmegaConf.load(config.input_config.feature_extract_config_path)

    out_path = char_config.out_path

    class_desc_path = "/datasets01/Charades-ego-v1/101320/charades-ego-v1/CharadesEgo/Charades_v1_classes.txt"
    class_name_df = pd.read_csv(class_desc_path, header=None)
    class_names = [" ".join(x[1].split(" ")[1:]) for x in class_name_df.itertuples()]

    def get_label_name(x):
        x.replace("Someone", "")
        x.replace("is", "")
        return x.lower()

    sentences_ego = [
        f"Camera wearer is {get_label_name(clazz)}"
        for clazz in class_names
    ]

    sentences_non_ego = [
        f"The person in this video is {get_label_name(clazz)}"
        for clazz in class_names
    ]
    model = SentenceTransformer(config.pre_config.ego4d_narr.st_model_name)
    label_name_fv = model.encode(
        class_names,
        device="cuda",  # TODO
        show_progress_bar=True,
    )
    sent_ego_fv = model.encode(
        sentences_ego,
        device="cuda",  # TODO
        show_progress_bar=True,
    )
    sent_non_ego = model.encode(
        sentences_non_ego,
        device="cuda",  # TODO
        show_progress_bar=True,
    )
    torch.save({
        "labels": label_name_fv,
        "sent_ego_fv": sent_ego_fv,
        "sent_non_ego_fv": sent_non_ego,
    }, char_config.out_label_path)
    # TODO: add back
    # video_path_ids = [(os.path.join(root_path, f"{row.id}.mp4"), row.id) for row in val_df.itertuples()]
    # video_path_ids = [vp for vp in video_path_ids if os.path.exists(vp[0])]

    # batches = batch_it(video_path_ids, char_config.num_vids_per_machine)
    # executor = create_executor(char_config, len(batches))
    # map_fn = functools.partial(
    #     _preprocess_ego_charade,
    #     feature_extract_config=feature_extract_config,
    # )

    # jobs = executor.map_array(map_fn, batches)

    # with h5py.File(out_path, "w") as out_f:
    #     for j in tqdm(jobs):
    #         feat = j.result()
    #         for uid, ret in feat.items():
    #             out_f.create_dataset(uid, data=ret["all_fvs"].numpy())


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


def _map_cc_batch(batch, cc: PreprocessConfig, feature_extract_config: FeatureExtractConfig):
    paths = [x for x, _ in batch]
    viz_model = load_model(feature_extract_config, patch_final_layer=True)

    image_transform = T.Compose(
        [
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.Lambda(lambda x: x.unsqueeze(0).permute(1, 0, 2, 3))
        ]
    )

    dset = ImagePathDset(paths, image_transform)
    dloader = create_data_loader(dset, cc, shuffle=False)
    viz_model.eval()

    ret = {}
    with torch.no_grad():
        for xx in tqdm(dloader, total=len(dloader)):
            fv = viz_model(xx["img"].cuda())
            for p, f in zip(xx["path"], fv.cpu().numpy()):
                ret[p] = f
    return ret


def _map_cc_sent_batch(batch, config: TrainConfig, cc: PreprocessConfig):
    paths = [x for x, _ in batch]
    cap = [x for _, x in batch]

    sent_model = SentenceTransformer(config.pre_config.ego4d_narr.st_model_name)
    fvs = sent_model.encode(cap, device="cuda", show_progress_bar=True)
    ret = {}
    for p, f in zip(paths, fvs):
        ret[p] = f
    return ret

def _get_fs(x):
    return x, os.path.getsize(x)

def preprocess_cc(config: TrainConfig, cc: CCPreprocessConfig):
    in_path = "/checkpoint/miguelmartin/conceptial_captions/Train_GCC-training_output.csv"
    train_df = pd.read_csv(in_path, sep='\t')

    examples = dict(zip(train_df.filepath, train_df.title))
    invalid_keys = []
    with Pool(20) as pool:
        for path, x in tqdm(pool.imap(_get_fs, examples), total=len(examples)):
            if x == 0:
                invalid_keys.append(path)
    invalid_keys = set(invalid_keys)
    print(len(invalid_keys))
    examples = {k: v for k, v in examples.items() if k not in invalid_keys}

    feature_extract_config = OmegaConf.load(config.input_config.feature_extract_config_path)
    feature_extract_config.model_config.input_type = "image"  # not actually needed

    all_ex = list(examples.items())
    batches = batch_it(all_ex, cc.imgs_per_gpu)

    executor = create_executor(cc, len(batches))
    jobs = executor.map_array(
        functools.partial(_map_cc_batch, cc=cc, feature_extract_config=feature_extract_config),
        batches,
    )

    with h5py.File(cc.hdf5_viz_path, "w") as out_f:
        for job in tqdm(jobs):
            print(job)
            res = job.result()
            for k, v in res.items():
                out_f.create_dataset(k, data=v)

    print("converting captions")
    jobs = executor.map_array(
        functools.partial(_map_cc_sent_batch, config=config, cc=cc),
        batches,
    )
    with h5py.File(cc.hdf5_sent_path, "w") as out_f:
        for job in tqdm(jobs):
            print(job)
            res = job.result()
            for k, v in res.items():
                out_f.create_dataset(k, data=v)


@hydra.main(config_path="configs", config_name=None)
def preprocess(config: TrainConfig):
    # TODO: refactor
    if config.preprocess_mode == "ego":
        preprocess_ego_narrations(config.pre_config.ego4d_narr)
    elif config.preprocess_mode == "ego_features":
        preprocess_ego_features(config.input_config.feature_path, config.pre_config, config.pre_config.ego4d_features)
    elif config.preprocess_mode == "k400":
        preprocess_k400_data(config, config.pre_config.k400)
    elif config.preprocess_mode == "ego_charade":
        preprocess_ego_charade(config, config.pre_config.ego_charade)
    elif config.preprocess_mode == "cc":
        preprocess_cc(config, config.pre_config.cc)
    else:
        raise AssertionError("{config.preprocess_mode} not supported")


if __name__ == "__main__":
    preprocess()  # pyre-ignore
