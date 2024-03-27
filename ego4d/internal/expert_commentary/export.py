import datetime
import json
import os
from collections import defaultdict

from ego4d.internal.expert_commentary.extract import (
    load_uniq_commentaries,
    RAW_EXTRACTED_COMM_ROOT,
)
from tqdm.auto import tqdm


def export(
    takes, released_takes, to_dir_base, raw_extracted_dir=RAW_EXTRACTED_COMM_ROOT
):
    comms = load_uniq_commentaries(raw_extracted_dir=raw_extracted_dir)
    takes_by_take_name = {x["take_name"]: x for x in takes}
    released_tns = {x["take_name"] for x in takes if x["take_uid"] in released_takes}

    take_name_to_comms = defaultdict(list)
    for x in comms:
        temp = os.path.basename(x).split("_")
        tn = "_".join(temp[1:-1])
        expert_name = temp[0]
        take_name_to_comms[tn].append((expert_name, x))

    num_anns = sum(len(xs) for xs in take_name_to_comms.values())
    num_anns_released = sum(
        len(xs) for tn, xs in take_name_to_comms.items() if tn in released_tns
    )
    print(
        {
            "num_anns": num_anns,
            "num_anns_released": num_anns_released,
            "num_takes_covered": len(take_name_to_comms),
            "num_takes_covered_released": len(released_tns & set(take_name_to_comms)),
            "num_takes_released": len(released_tns),
        }
    )

    to_dir_data_base = os.path.join(to_dir_base, "data")

    transc_export = []
    copy_commands = []
    for take_name, xs in tqdm(
        take_name_to_comms.items(), total=len(take_name_to_comms)
    ):
        if take_name not in released_tns:
            continue
        temp = {}
        for (
            expert_name,
            from_dir,
        ) in xs:
            to_dir_rel = os.path.join(take_name, expert_name)
            to_dir = os.path.join(to_dir_data_base, take_name, expert_name)

            trans_path = os.path.join(from_dir, "transcriptions.json")
            data_path = os.path.join(from_dir, "data.json")
            copy_commands.append((data_path, os.path.join(to_dir, "data.json")))
            copy_commands.append(
                (trans_path, os.path.join(to_dir, "transcriptions.json"))
            )

            trans = json.load(open(trans_path))
            data = json.load(open(data_path))
            ann_by_rec = {x["recording_path"]: x for x in data["annotations"]}
            take = takes_by_take_name[take_name]
            comm_data = []
            for rec_id, x in trans.items():
                if rec_id not in ann_by_rec:
                    continue
                copy_commands.append(
                    (
                        os.path.join(from_dir, "recordings", rec_id),
                        os.path.join(to_dir, "recordings", rec_id),
                    )
                )
                ann = ann_by_rec[rec_id]
                comm_data.append(
                    {
                        "recording": rec_id,
                        "video_time": ann["video_time"],
                        "text": x.get("text"),
                        "error": x["error"],
                        "_error_desc": x["error_desc"],
                        "duration_approx": ann["duration_approx"],
                    }
                )
            transc_export.append(
                {
                    "take_name": take_name,
                    "take_uid": take["take_uid"],
                    "task_id": take["task_id"],
                    "task_name": take["task_name"],
                    "commentary": to_dir_rel,
                    "commentary_data": comm_data,
                    "_debug_commentary_name": os.path.basename(from_dir),
                }
            )

    print(f"number of commentaries: {len(transc_export)}")

    by_take = defaultdict(list)
    for x in transc_export:
        by_take[x["take_uid"]].append(x)

    dt_str = "{date:%Y%m%d_%H:%M:%S}".format(date=datetime.datetime.now())
    final_trans_export = {
        "annotations": by_take,
        "ds": dt_str,
    }

    os.makedirs(to_dir_base, exist_ok=True)
    out_path = os.path.join(to_dir_base, "expert_commentary_transc.json")
    json.dump(final_trans_export, open(out_path, "w"), indent=2)

    base_dirs = set()
    for f, t in tqdm(copy_commands):
        assert os.path.exists(f), f
        base_dir = os.path.dirname(t)
        base_dirs.add(base_dir)
    print(len(base_dirs))

    for x in tqdm(base_dirs):
        os.makedirs(x, exist_ok=True)

    with open("copy.txt", "w") as out_f:
        for f, t in tqdm(copy_commands):
            out_f.write(f"cp {f} {t}\n")
    print("Copy with:")
    print("cat copy.txt | parallel --eta")
    return copy_commands


def check_export(copy_commands):
    for _, t in tqdm(copy_commands):
        assert os.path.exists(t), t
    return copy_commands


if __name__ == "__main__":
    date_ver = "240324"
    released_takes_path = "/large_experiments/egoexo/v2/released_takes.json"
    takes_path = "/large_experiments/egoexo/v2/takes.json"
    takes = json.load(open(takes_path))
    released_takes = (
        json.load(open(takes_path))
        if os.path.exists(released_takes_path)
        else {x["take_uid"] for x in takes}
    )
    to_dir_base = f"/checkpoint/miguelmartin/expert_commentary/exports/{date_ver}"

    cc = export(released_takes=released_takes, takes=takes, to_dir_base=to_dir_base)
    check_export(cc)
