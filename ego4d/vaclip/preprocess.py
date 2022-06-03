import sys
import os
import json
import functools
import math
import submitit
import logging
from typing import List, Any
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import torch
import hydra
from sentence_transformers import SentenceTransformer
from ego4d.vaclip.config import EgoPreprocessConfig, TrainConfig
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


def sub_tagged_tokens(text: str) -> str:
    text = text.replace("#C", "Camera wearer")
    text = text.replace("#O", "Other person")
    text = text.replace("#unsure", "Something")
    return text


def get_narrations(config: EgoPreprocessConfig):
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
    model = SentenceTransformer(config.ego_pre_config.st_model_name)

    narr_op = os.path.join(config.ego_pre_config.pre_root_dir, config.ego_pre_config.narration_out_path)
    batches = batch_it(narrs, config.ego_pre_config.batch_size)

    metas = []
    for batch in tqdm(batches):
        fvs = model.encode(
            [x for _, (x, _, _, _) in batch],
            device=config.ego_pre_config.accelerator,
            show_progress_bar=False,
        )

        for fv, (idx, (post_txt, uid, txt, ts)) in zip(fvs, batch):
            od = os.path.join(narr_op, uid)
            os.makedirs(od, exist_ok=True)
            path_to_encode = os.path.join(od, f"{idx}.pt")
            torch.save(fv, path_to_encode)
            metas.append({"uid": uid, "txt": txt, "ts": ts, "idx": idx, "post_txt": post_txt})
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


def preprocess_ego_features(config: TrainConfig):
    # TODO: partition by time
    pass


def preprocess_ego_narrations(config: TrainConfig):
    os.makedirs(config.ego_pre_config.pre_root_dir, exist_ok=True)

    narr_op = os.path.join(config.ego_pre_config.pre_root_dir, config.ego_pre_config.narration_out_path)
    os.makedirs(narr_op, exist_ok=True)

    narrs = get_narrations(config.ego_pre_config)

    print("Transforming text...")
    narrs_with_idx = list(enumerate([(sub_tagged_tokens(txt), uid, txt, ts) for uid, txt, ts in narrs]))
    if config.ego_pre_config.limit > 0:
        narrs_with_idx = narrs_with_idx[0:config.ego_pre_config.limit]

    batches = batch_it(narrs_with_idx, config.ego_pre_config.num_narrs_per_machine)
    print(f"Running txt through transformer with {len(batches)} machines")
    print(f"Num narrs = {len(narrs_with_idx)}", flush=True)

    metas = []
    executor = create_executor(config.ego_pre_config, len(batches))
    jobs = executor.map_array(
        functools.partial(map_narrs_on_machine, config=config),
        batches,
    )
    print("Jobs", jobs, flush=True)

    for j in tqdm(jobs):
        metas.extend(j.result())

    print("Saving metadata")
    m_op = os.path.join(config.ego_pre_config.pre_root_dir, config.ego_pre_config.metadata_out_path)
    torch.save(metas, m_op)


def _preprocess_k400_data(video_path_label_pairs, feature_extract_config, viz_dir):
    model = load_model(feature_extract_config, patch_final_layer=True)

    for path, label in tqdm(video_path_label_pairs):
        name = Path(path).stem
        vid = Video(path, path, video_info(path)["num_frames"], w=None, h=None)
        if vid.frame_count is None:
            continue

        predictions = extract_features(
            videos=[vid],
            config=feature_extract_config,
            model=model,
            log_info=False,
            silent=True,
            assert_feature_size=False,
        )

        out_path = os.path.join(viz_dir, f"{name}.pt")
        to_save = {
            "feature": predictions.result[path].mean(0),
            "label": label,
            "all_features": predictions.result[path],
        }
        torch.save(to_save, out_path)


def preprocess_k400_data(config: TrainConfig):
    pre_dir = os.path.join(config.k400_pre_config.pre_root_dir, config.k400_pre_config.set_to_use)
    viz_dir = os.path.join(pre_dir, config.k400_pre_config.viz_feature_dir)

    os.makedirs(pre_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    # TODO: configure
    val_set = pd.read_csv("/datasets01/kinetics/092121/400/lists/val.csv")
    idx_to_label = _load_kinetics_class_names()
    label_to_idx = {v: k for k, v in idx_to_label.items()}

    def process_label(label):
        ret = label.replace('"', '')
        ret = ret.replace("'", '')
        ret = ret.replace("_", ' ')
        return ret

    label_names = list(set(val_set.label))
    sentences = [
        f"The person in this video is doing {process_label(label)}"
        for label in label_names
    ]
    feature_extract_config = OmegaConf.load(config.k400_pre_config.feature_extract_config_path)
    video_path_label_pairs = [
        (
            f"/datasets01/kinetics/092121/400/val_288px/{row.youtube_id}_{row.time_start:06d}_{row.time_end:06d}.mp4", 
            row.label,
        )
        for row in val_set.itertuples()
    ]

    old_len = len(video_path_label_pairs)
    video_path_label_pairs = [val for val in video_path_label_pairs if os.path.exists(val[0])]
    print(f"{old_len} -> {len(video_path_label_pairs)} examples", flush=True)

    map_fn = functools.partial(
        _preprocess_k400_data,
        feature_extract_config=feature_extract_config,
        viz_dir=viz_dir
    )
    batches = batch_it(video_path_label_pairs, batch_size=config.k400_pre_config.num_labels_per_machine)

    if config.run_locally:
        for batch in batches:
            map_fn(batch)
    else:
        print(f"To schedule {len(batches)} batches across {config.k400_pre_config.slurm_array_parallelism} machines")
        cont = input("Continue? [y/N]: ")
        if cont != "y":
            print("Exiting...")
            sys.exit(0)
        executor = create_executor(config.k400_pre_config, len(batches))
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
    model = SentenceTransformer(config.ego_pre_config.st_model_name)
    for sent, label_name in tqdm(zip(sentences, label_names), total=len(sentences)):
        meta["label_text"].append(sent)
        meta["label_fv"].append(model.encode(sent, show_progress_bar=False))

    out_meta_path = os.path.join(pre_dir, config.k400_pre_config.metadata_out_path)
    torch.save(meta, out_meta_path)


@hydra.main(config_path="configs", config_name=None)
def preprocess(config: TrainConfig):
    if config.preprocess_mode == "ego":
        preprocess_ego_narrations(config)
    elif config.preprocess_mode == "k400":
        preprocess_k400_data(config)


if __name__ == "__main__":
    preprocess()  # pyre-ignore
