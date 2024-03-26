import functools
import json
import os
from typing import Any, Dict, List, Tuple

import submitit.helpers as sh

import torch
from ego4d.research.clep.config import (
    EgoPreprocessFeatureConfig,
    EgoPreprocessNarrConfig,
    InputConfig,
    TrainConfig,
)
from ego4d.research.clep.preprocess.common import get_language_model
from ego4d.research.common import batch_it, create_executor
from ego4d.research.dataset import save_features_to_hdf5

from tqdm.auto import tqdm


def preprocess_ego_narrations(
    config: TrainConfig, narr_config: EgoPreprocessNarrConfig
):
    out_dir = config.pre_config.root_dir
    os.makedirs(out_dir, exist_ok=True)

    narr_od = os.path.join(out_dir, narr_config.narration_out_dir)
    os.makedirs(narr_od, exist_ok=True)

    narrs = get_narrations(narr_config)
    narrs = filter_narrations(narrs, config.input_config, narr_config)

    print("Transforming text...")
    narrs_with_idx = list(
        enumerate(
            [
                (remove_tags(txt), sub_tagged_tokens(txt), uid, txt, ts)
                for uid, txt, ts in narrs
            ]
        )
    )
    narrs_with_idx = narrs_with_idx[0 : narr_config.limit]

    batches = batch_it(narrs_with_idx, narr_config.num_narrs_per_machine)
    print(f"Running txt through transformer with {len(batches)} machines")
    print(f"Num narrs = {len(narrs_with_idx)}", flush=True)

    metas = []
    executor = create_executor(config.pre_config.slurm_config, len(batches))
    jobs = executor.map_array(
        functools.partial(
            _map_narrs_on_machine,
            config=config,
            narr_od=narr_od,
        ),
        batches,
    )
    print("Jobs", jobs, flush=True)

    metas = []
    for j in tqdm(sh.as_completed(jobs), total=len(jobs)):
        res = j.result()
        metas.extend(res)

    print("Saving metadata")
    m_op = os.path.join(out_dir, narr_config.metadata_out_path)
    torch.save(metas, m_op)


def preprocess_ego_features(
    feature_path: str,
    config: TrainConfig,
    pre_feature: EgoPreprocessFeatureConfig,
):
    out_dir = config.pre_config.root_dir
    os.makedirs(out_dir, exist_ok=True)

    meta = json.load(open(config.input_config.metadata_path))
    video_uids = [x["video_uid"] for x in meta["videos"]]

    out_path = os.path.join(config.pre_config.root_dir, pre_feature.hdf5_path)
    print("=>", out_path, flush=True)
    save_features_to_hdf5(
        video_uids=video_uids,
        feature_dir=feature_path,
        out_path=out_path,
    )


def _map_narrs_on_machine(
    narrs: List[Tuple[int, Tuple[str, str, str, str, float]]],
    config: TrainConfig,
    narr_od: str,
) -> List[Dict[str, Any]]:
    model = get_language_model(config)

    batches = batch_it(narrs, config.batch_size)

    metas = []
    for batch in tqdm(batches):
        fvs = model.encode(
            [x for _, (_, x, _, _, _) in batch],
            device="cuda" if config.accelerator == "gpu" else config.accelerator,
            show_progress_bar=False,
        )
        fvs_without_tags = model.encode(
            [x for _, (x, _, _, _, _) in batch],
            device="cuda" if config.accelerator == "gpu" else config.accelerator,
            show_progress_bar=False,
        )

        for fv_no_tag, fv, (idx, (no_tag_txt, post_txt, uid, txt, ts)) in zip(
            fvs_without_tags, fvs, batch
        ):
            od = os.path.join(narr_od, uid)
            os.makedirs(od, exist_ok=True)
            path_to_encode = os.path.join(od, f"{idx}.pt")
            torch.save(
                {
                    "fv": fv,
                    "fv_no_tag": fv_no_tag,
                },
                path_to_encode,
            )
            metas.append(
                {
                    "uid": uid,
                    "txt": txt,
                    "ts": ts,
                    "idx": idx,
                    "post_txt": post_txt,
                    "no_tag_txt": no_tag_txt,
                }
            )
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


def filter_narrations(
    narrations, config: InputConfig, pre_config: EgoPreprocessNarrConfig
):
    meta = json.load(open(config.metadata_path))
    val_set_uids = [
        vid["video_uid"]
        for vid in meta["videos"]
        if len({vid["split_em"], vid["split_fho"], vid["split_av"]} & {"val", "test"})
        > 0
    ]

    num_val_filtered = 0
    num_txt_filtered = 0
    ret = []
    for uid, txt, ts in tqdm(narrations):
        if uid in val_set_uids:
            num_val_filtered += 1
            continue
        if len(txt.split(" ")) < pre_config.min_words:
            num_txt_filtered += 1
            continue
        ret.append((uid, txt, ts))
    print(f"Narrations filtered from {len(narrations)} -> {len(ret)}")
    print(
        f"""
    Val Filtered = {num_val_filtered} = {num_val_filtered/len(narrations):.2%}
    Txt Filtered = {num_txt_filtered} = {num_txt_filtered/len(narrations):.2%}

    """
    )
    return ret


def get_narrations(
    config: EgoPreprocessNarrConfig,
) -> List[Tuple[str, str, float, str]]:
    narration_json = json.load(open(config.narration_json_path))
    uid_subset = set(narration_json.keys())
    narrations = [
        (uid, data["narration_text"], data["timestamp_sec"])
        for uid in uid_subset
        for data in narration_json[uid].get("narration_pass_1", {"narrations": []})[
            "narrations"
        ]
    ]
    narrations += [
        (uid, data["narration_text"], data["timestamp_sec"])
        for uid in uid_subset
        for data in narration_json[uid].get("narration_pass_2", {"narrations": []})[
            "narrations"
        ]
    ]
    narrations.sort(key=lambda x: (x[0], x[-1]))
    return narrations
