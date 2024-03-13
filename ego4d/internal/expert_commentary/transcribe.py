import argparse
import datetime
import functools
import json
import os
import sys
import traceback
from typing import List

import submitit
import torch
import whisper
from ego4d.egoexo.expert_commentary.data import load_all_raw_commentaries

from ego4d.research.common import batch_it
from tqdm.auto import tqdm


def transcribe_commentaries(commentary_folder: str, model):
    # TODO: skip if exists?
    result = {}
    for root, _, files in os.walk(commentary_folder):
        for file in files:
            if file.endswith("webm"):
                webm_path = os.path.join(root, file)
                try:
                    temp = model.transcribe(webm_path)
                    transc = {
                        k: (v.cpu() if isinstance(v, torch.Tensor) else v)
                        for k, v in temp.items()
                        if k not in ("segments",)
                    }
                    transc["error"] = False
                    transc["_error_desc"] = None
                    result[file] = transc
                except Exception:
                    result[file] = {
                        "error": True,
                        "_error_desc": traceback.format_exc(),
                    }

    out_p = os.path.join(commentary_folder, "transcriptions.json")
    with open(out_p, "w") as out_f:
        json.dump(result, out_f, indent=4)
    return result


def run_worker(comms_batch: List[str], model_name: str, device: str):
    # TODO: Use whisperX or another transcription model?
    model = whisper.load_model(model_name, device=device)
    total_num_errs = 0
    total_num_succ = 0
    pbar = tqdm(
        comms_batch, desc=f"Transcribing ({total_num_succ} succ, {total_num_errs} errs)"
    )
    for comm_folder in pbar:
        temp = transcribe_commentaries(comm_folder, model)
        num_errs = sum(x["error"] for x in temp.values())
        num_succ = sum(not x["error"] for x in temp.values())
        total_num_errs += num_errs
        total_num_succ += num_succ
        pbar.set_description(
            f"Transcribing ({total_num_succ} succ, {total_num_errs} errs)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--commentary_root",
        type=str,
        help="Path to expert commentary root directory",
        required=True,
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        help="Size of each batch",
        default=4,
    )
    parser.add_argument(
        "-p",
        "--machine_pool_size",
        type=int,
        help="number of machines to schedule in job array",
        default=128,
    )
    parser.add_argument(
        "--timeout_min",
        type=int,
        help="timeout for job scheduling",
        default=240,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of whisper model",
        default="large-v2",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="device to use",
        default="cuda",
    )
    parser.add_argument(
        "-y",
        "--yes",
        help="Whether to proceed with scheduling without a prompt",
        default=False,
        action="store_true",
    )
    args = parser.parse_args()

    assert os.path.exists(
        args.commentary_root
    ), f"{args.commentary_root} does not exist"
    comms = load_all_raw_commentaries(args.commentary_root)
    comms = [
        x for x in comms if not os.path.exists(os.path.join(x, "transcriptions.json"))
    ]
    job_inputs = batch_it(comms, args.batch_size)

    confirm = False
    if not args.yes:
        response = input(
            f"Total commentaries: {len(comms)}\n"
            f"# batches: {len(job_inputs)}, bs={args.batch_size}\n"
            f"# machines: {args.machine_pool_size}, timeout={args.timeout_min}\n"
            f"Continue? [Y/n]: "
        )
        if response.lower() in ["yes", "y", ""]:
            confirm = True
        else:
            confirm = False
    else:
        confirm = True

    if not confirm:
        print("Exiting...")
        sys.exit(0)

    map_fn = functools.partial(
        run_worker, model_name=args.model_name, device=args.device
    )

    dt_now = datetime.datetime.now().strftime("%y%m%d_%H%M")
    log_dir = f"expert_commentary_log/{dt_now}"
    print(f"Logging to: {log_dir}")
    os.makedirs(log_dir, exist_ok=True)
    executor = submitit.AutoExecutor(folder=log_dir)
    executor.update_parameters(
        timeout_min=args.timeout_min,
        slurm_array_parallelism=min(args.machine_pool_size, len(job_inputs)),
        slurm_constraint="volta",
        slurm_partition="eht",
        gpus_per_node=1,
        cpus_per_task=10,
    )

    print("Scheduling ...")
    jobs = executor.map_array(map_fn, job_inputs)

    print("Waiting...")
    results = []
    for job in tqdm(jobs):
        results.append(job.result())
    print("Done")
