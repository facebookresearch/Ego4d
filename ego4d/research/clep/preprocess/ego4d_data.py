import os
import json
import functools
from typing import Dict, Tuple, List, Any

import torch
import h5py
import numpy as np

from tqdm.auto import tqdm
from ego4d.research.clep.config import (
    EgoPreprocessFeatureConfig,
    EgoPreprocessNarrConfig,
    PreprocessConfig,
)
from ego4d.research.clep.preprocess.common import (
    get_language_model,
)
from ego4d.research.common import (
    create_executor,
    batch_it,
)


def preprocess_ego_narrations(config: PreprocessConfig, narr_config: EgoPreprocessNarrConfig):
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
    narrs_with_idx = narrs_with_idx[0:narr_config.limit]

    batches = batch_it(narrs_with_idx, narr_config.num_narrs_per_machine)
    print(f"Running txt through transformer with {len(batches)} machines")
    print(f"Num narrs = {len(narrs_with_idx)}", flush=True)

    metas = []
    executor = create_executor(config.slurm_config, len(batches))
    jobs = executor.map_array(
        functools.partial(_map_narrs_on_machine, config=narr_config),
        batches,
    )
    print("Jobs", jobs, flush=True)

    for j in tqdm(jobs):
        metas.extend(j.result())

    print("Saving metadata")
    m_op = os.path.join(narr_config.pre_root_dir, narr_config.metadata_out_path)
    torch.save(metas, m_op)


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


def _map_narrs_on_machine(
    narrs: List[Tuple[int, Tuple[str, str, str, str, float]]],
    config: EgoPreprocessNarrConfig, 
) -> Dict[str, Any]:
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


def get_narrations(config: EgoPreprocessNarrConfig) -> List[Tuple[str, str, float]]:
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