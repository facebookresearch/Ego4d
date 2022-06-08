import sys
import os
import json
import random
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
    narrations = filter_narrations(narrations)
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


def _save_batch(val, to_save_root):
    idx, batch = val
    to_save_path = os.path.join(to_save_root, f"{idx}.pt")
    torch.save(batch, to_save_path)
    return idx, len(batch), to_save_path


def _segment_ego_features(video_uids, config: TrainConfig):
    ret = {}
    with Pool(512) as pool:
        for uid in tqdm(video_uids, desc="video_uid", leave=True):
            feature_path = os.path.join(config.input_config.feature_path, f"{uid}.pt")
            fv = torch.load(feature_path)
            nf = fv.shape[0]

            res = []
            batches = list(enumerate(batch_it(fv, config.ego_pre_feature_config.num_features_per_file)))

            to_save_root = os.path.join(config.ego_pre_feature_config.pre_root_dir, f"{uid}")
            os.makedirs(to_save_root, exist_ok=True)

            map_fn = functools.partial(_save_batch, to_save_root=to_save_root)
            cum_nf = 0
            # for idx, batch_len, to_save_path in tqdm(pool.imap(map_fn, batches), total=len(batches)):
            for idx, batch_len, to_save_path in pool.imap(map_fn, batches):
                res.append((idx, to_save_path))
                cum_nf += batch_len
            res.sort(key=lambda x: x[0])
            assert cum_nf == nf

            ret[uid] = {
                "idx_path_pairs": res,
                "num_features": nf,
            }

    return ret


def preprocess_ego_features(config: TrainConfig):
    narr_meta_path = os.path.join(config.ego_pre_config.pre_root_dir, config.ego_pre_config.metadata_out_path)
    narr_meta = torch.load(narr_meta_path)
    video_uids = {x["uid"] for x in narr_meta}
    video_uids = list(video_uids)
    video_uids = video_uids[0:100]
    random.shuffle(video_uids)

    print("===")
    print(config.ego_pre_feature_config.pre_root_dir, flush=True)
    print()
    print()
    print()

    all_feature_paths = {}
    map_fn = functools.partial(_segment_ego_features, config=config)

    all_feature_paths = {}
    if config.run_locally:
        all_feature_paths = map_fn(video_uids)
    else:
        batches = batch_it(video_uids, 350)
        print(f"To schedule {len(batches)} batches across {config.ego_pre_feature_config.slurm_array_parallelism} machines")
        cont = input("Continue? [y/N]: ")
        if cont != "y":
            print("Exiting...")
            sys.exit(0)

        executor = create_executor(config.ego_pre_feature_config, len(batches))
        jobs = executor.map_array(
            functools.partial(map_fn, config=config),
            batches,
        )
        for job in tqdm(jobs):
            val = job.result()
            for k, v in val.items():
                all_feature_paths[k] = v

    meta_path = os.path.join(config.ego_pre_feature_config.pre_root_dir, config.ego_pre_feature_config.meta_path)
    torch.save(all_feature_paths, meta_path)


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
    random.seed(1337)

    pre_dir = os.path.join(config.k400_pre_config.pre_root_dir, config.k400_pre_config.set_to_use)
    viz_dir = os.path.join(pre_dir, config.k400_pre_config.viz_feature_dir)

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
            f"/datasets01/kinetics/092121/400/val_288px/{row.youtube_id}_{row.time_start:06d}_{row.time_end:06d}.mp4",
            row.label,
        )
        for row in val_set.itertuples()
    ]

    old_len = len(video_path_label_pairs)
    video_path_label_pairs = [val for val in video_path_label_pairs if os.path.exists(val[0])]
    print(f"{old_len} -> {len(video_path_label_pairs)} examples", flush=True)

    feature_extract_config = OmegaConf.load(config.k400_pre_config.feature_extract_config_path)
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
    for idx, (sent, label_name) in tqdm(enumerate(zip(sentences, label_names)), total=len(sentences)):
        assert label_to_idx[label_name] == idx
        fv = model.encode(sent, show_progress_bar=False)
        meta["label_text"].append(sent)
        meta["label_fv"].append(fv)

    out_meta_path = os.path.join(pre_dir, config.k400_pre_config.metadata_out_path)
    torch.save(meta, out_meta_path)


def preprocess_cc(config: TrainConfig):
    pass


def preprocess_imagenet(config: TrainConfig):
    pass


@hydra.main(config_path="configs", config_name=None)
def preprocess(config: TrainConfig):
    # import IPython
    # IPython.embed()

    if config.preprocess_mode == "ego":
        preprocess_ego_narrations(config)
    elif config.preprocess_mode == "ego_features":
        preprocess_ego_features(config)
    elif config.preprocess_mode == "k400":
        preprocess_k400_data(config)


if __name__ == "__main__":
    preprocess()  # pyre-ignore
