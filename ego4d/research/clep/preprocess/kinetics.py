import functools
import os
import random
import sys

import h5py
import pandas as pd
import torch
from ego4d.features.config import load_model
from ego4d.features.inference import _load_kinetics_class_names
from ego4d.research.clep.config import K400PreprocessConfig, TrainConfig
from ego4d.research.clep.preprocess.common import (
    get_language_model,
    run_feature_extraction,
)
from ego4d.research.common import batch_it, create_executor
from omegaconf import OmegaConf
from tqdm.auto import tqdm


def _preprocess_k400_data(video_path_label_pairs, feature_extract_config):
    model = load_model(feature_extract_config, patch_final_layer=True)

    ret = {}
    for path, label in tqdm(video_path_label_pairs):
        predictions = run_feature_extraction(path, model, feature_extract_config)
        if predictions is None:
            continue

        ret[path] = {
            "features": predictions.result[path],
            "label": label,
        }
    return ret


def preprocess_k400_data(config: TrainConfig, k_config: K400PreprocessConfig):
    """
    Assumptions:
        - input split given in a CSV file of the form:
            - <csv_dir>/<set_to_use>.csv
        - videos given in directory of the form:
            - "<youtube_id>_<start_time>_<end_time>.mp4
            - start_time / end_time are padded with up to 6 0's, e.g. --07WQ2iBlw_000001_000011.mp4
    """
    random.seed(1337)

    out_dir = os.path.join(config.pre_config.root_dir, k_config.root_dir)
    os.makedirs(out_dir, exist_ok=True)

    # TODO: configure
    val_set = pd.read_csv(os.path.join(k_config.csv_dir, f"{k_config.set_to_use}.csv"))

    def process_label(label):
        ret = label.replace('"', "")
        ret = ret.replace("'", "")
        ret = ret.replace("_", " ")
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
            f"{k_config.dataset_dir}/{row.youtube_id}_{row.time_start:06d}_{row.time_end:06d}.mp4",
            row.label,
        )
        for row in val_set.itertuples()
    ]

    old_len = len(video_path_label_pairs)
    video_path_label_pairs = [
        val for val in video_path_label_pairs if os.path.exists(val[0])
    ]
    print(f"{old_len} -> {len(video_path_label_pairs)} examples", flush=True)

    feature_extract_config = OmegaConf.load(
        config.input_config.feature_extract_config_path
    )
    map_fn = functools.partial(
        _preprocess_k400_data,
        feature_extract_config=feature_extract_config,
    )
    batches = batch_it(
        video_path_label_pairs, batch_size=k_config.num_labels_per_machine
    )

    label_name_pairs = []
    if config.run_locally:
        raise AssertionError("not supported yet")
    else:
        slurm_config = config.pre_config.slurm_config
        print(
            f"To schedule {len(batches)} batches across {slurm_config.slurm_array_parallelism} machines"
        )
        cont = input("Continue? [y/N]: ")
        if cont != "y":
            print("Exiting...")
            sys.exit(0)
        executor = create_executor(slurm_config, len(batches))
        jobs = executor.map_array(map_fn, batches)

        # wait for the results
        out_path = os.path.join(out_dir, k_config.viz_feature_path)
        with h5py.File(out_path, "w") as out_f:
            for job in tqdm(jobs):
                vs = job.result()
                for k, v in vs.items():
                    label_name_pairs.append((k, v["label"]))
                    out_f.create_dataset(k, data=v["features"].numpy())

    # sentences
    print("Processing labels as sentences", flush=True)
    meta = {
        "labels": label_name_pairs,
        "label_text": [],
        "label_fv": [],
        "label_name_to_idx": label_to_idx,
        "idx_to_label_name": idx_to_label,
    }
    model = get_language_model(config)
    for idx, (sent, label_name) in tqdm(
        enumerate(zip(sentences, label_names)), total=len(sentences)
    ):
        assert label_to_idx[label_name] == idx
        fv = model.encode(sent, show_progress_bar=False)
        meta["label_text"].append(sent)
        meta["label_fv"].append(fv)

    out_meta_path = os.path.join(out_dir, k_config.metadata_out_path)
    torch.save(meta, out_meta_path)
