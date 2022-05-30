import os
import json
import functools
import math
from typing import List, Any
from multiprocessing import Pool

import torch
import hydra
from sentence_transformers import SentenceTransformer
from ego4d.vaclip.config import PreprocessConfig, TrainConfig
from tqdm.auto import tqdm


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


def get_narrations(config: PreprocessConfig):
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


@hydra.main(config_path="configs", config_name=None)
def preprocess_ego_data(config: TrainConfig):
    os.makedirs(config.pre_config.pre_root_dir, exist_ok=True)

    narrs = get_narrations(config.pre_config)
    model = SentenceTransformer(config.pre_config.st_model_name)

    narr_op = os.path.join(config.pre_config.pre_root_dir, config.pre_config.narration_out_path)
    os.makedirs(narr_op, exist_ok=True)

    print("Transforming text...")
    narrs_with_idx = list(enumerate([(sub_tagged_tokens(txt), uid, txt, ts) for uid, txt, ts in narrs]))
    if config.pre_config.limit > 0:
        narrs_with_idx = narrs_with_idx[0:config.pre_config.limit]

    batches = batch_it(narrs_with_idx, config.pre_config.batch_size)

    print("Running txt through transformer")
    metas = []
    for batch in tqdm(batches):
        fvs = model.encode(
            [x for _, (x, _, _, _) in batch],
            device=config.pre_config.accelerator,
            show_progress_bar=False,
        )

        for fv, (idx, (post_txt, uid, txt, ts)) in zip(fvs, batch):
            od = os.path.join(narr_op, uid)
            os.makedirs(od, exist_ok=True)
            path_to_encode = os.path.join(od, f"{idx}.pt")
            torch.save(fv, path_to_encode)
            metas.append({"uid": uid, "txt": txt, "ts": ts, "idx": idx, "post_txt": post_txt})

    print("Saving metadata")
    m_op = os.path.join(config.pre_config.pre_root_dir, config.pre_config.metadata_out_path)
    torch.save(metas, m_op)


@hydra.main(config_path="configs", config_name=None)
def preprocess_gcc_txt(config: TrainConfig):
    # TODO
    pass


@hydra.main(config_path="configs", config_name=None)
def preprocess_gcc_imgs(config: TrainConfig):
    # TODO
    pass


if __name__ == "__main__":
    preprocess_ego_data()  # pyre-ignore
