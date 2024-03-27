import datetime
import functools
import json
import math
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

import submitit

import whisper
from ego4d.research.common import batch_it
from tqdm.auto import tqdm


ROOT_DIR = "/large_experiments/egoexo/v2/"
OUT_TAKES_DIR = "/checkpoint/miguelmartin/egoexo/v2/audio/"
PART_BY_UNI = True


def extract_audio_and_run_whisper(x, model):
    # need this here - otherwise pickle error
    from projectaria_tools.core.vrs import extract_audio_track

    base_dir = (
        os.path.join(OUT_TAKES_DIR, x["take_name"])
        if not PART_BY_UNI
        else os.path.join(OUT_TAKES_DIR, x["uni_id"], x["take_name"])
    )
    os.makedirs(base_dir, exist_ok=True)
    audio_outpath = os.path.join(base_dir, "audio", f"{x['vrs_base_name']}.wav")
    transcribe_outpath = os.path.join(
        base_dir,
        "audio",
        f"{x['vrs_base_name']}_transcriptions.json",
    )
    if not os.path.exists(audio_outpath):
        _ = extract_audio_track(
            x["vrs_path"],
            audio_outpath,
        )
        os.remove(audio_outpath + ".json")

    if os.path.exists(audio_outpath) and not os.path.exists(transcribe_outpath):
        temp = model.transcribe(audio_outpath, word_timestamps=True)
        json.dump(temp, open(transcribe_outpath, "w"), indent=2)


def process_all(xs):
    model_name = "large-v3"
    device = "cuda"
    model = whisper.load_model(model_name, device=device)
    map_fn = functools.partial(extract_audio_and_run_whisper, model=model)
    for x in tqdm(xs):
        map_fn(x)


def main():
    # TODO: argparse
    num_machines: int = 256
    root_dir: str = ROOT_DIR
    out_dir: str = OUT_TAKES_DIR

    takes_to_process = json.load(open(os.path.join(ROOT_DIR, "takes.json")))

    map_values = []
    completed = 0
    num_vids = 0
    for take in takes_to_process:
        td = os.path.join(root_dir, take["root_dir"])
        fs = os.listdir(td)
        fs = [f for f in fs if "noimagestream" in f]
        if len(fs) == 0:
            continue
        vrs_f = fs[0]
        vrs_file_path = os.path.join(root_dir, take["root_dir"], vrs_f)
        map_values.append(
            {
                "vrs_path": vrs_file_path,
                "vrs_base_name": os.path.splitext(vrs_f)[0].split("_")[0],
                "take_dir": td,
                "take_name": take["take_name"],
                "uni_id": take["university_id"],
            }
        )
        num_vids += 1

    print(
        f"# to process: {len(map_values)} / {num_vids} [{completed} / {1 - (len(map_values) / num_vids):.2%} completed]"
    )
    os.makedirs(out_dir, exist_ok=True)
    job_inputs = batch_it(
        map_values, batch_size=math.ceil(len(map_values) / num_machines)
    )
    num_machines = min(num_machines, len(job_inputs))

    dt_now = datetime.datetime.now().strftime("%y%m%d_%H%M")
    log_dir = f"extract_audio_and_run_whisper/{dt_now}"
    print(f"Logging to: {log_dir}")
    os.makedirs(log_dir, exist_ok=True)
    executor = submitit.AutoExecutor(folder=log_dir)

    executor.update_parameters(
        timeout_min=3200,
        slurm_array_parallelism=num_machines,
        slurm_constraint="volta",
        slurm_partition="eht",
        gpus_per_node=1,
        cpus_per_task=10,
    )
    jobs = executor.map_array(process_all, job_inputs)

    print("Waiting...")
    results = []
    for job in tqdm(jobs):
        results.append(job.result())
    print("Done")


if __name__ == "__main__":
    main()
