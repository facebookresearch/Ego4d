"""
Performs downscaling of takes videos with the SLURM cluster.

Credits to Santhosh Kumar Ramakrishnan for providing the original snippet this
code originates from.
"""
import datetime
import json
import math
import os
import subprocess as sp
from concurrent.futures import ThreadPoolExecutor

import submitit
from ego4d.research.common import batch_it
from tqdm.auto import tqdm


ROOT_DIR = "/checkpoint/miguelmartin/egoexo_data/dev/"
DS_TAKES_DIR = "/checkpoint/miguelmartin/egoexo/downscaled_takes/"
DEMONSTRATOR_TRAIN_DIR = (
    "/checkpoint/miguelmartin/demonstrator_cvpr_dataset/demonstrator_cvpr_train.json"
)
DEMONSTRATOR_VAL_DIR = (
    "/checkpoint/miguelmartin/demonstrator_cvpr_dataset/demonstrator_cvpr_val.json"
)
DEMONSTRATOR_TEST_DIR = (
    "/checkpoint/miguelmartin/demonstrator_cvpr_dataset/demonstrator_cvpr_test.json"
)


def call_ffmpeg(paths):
    src_path, tgt_path = paths
    assert os.path.exists(src_path)
    # https://docs.nvidia.com/video-technologies/video-codec-sdk/12.0/ffmpeg-with-nvidia-gpu/index.html
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        # TODO: try with cuda encoding for faster processing
        # '-hwaccel', 'cuda',
        # '-hwaccel_output_format', 'cuda',
        "-i",
        src_path,
        # This sweet conditional is thanks to ChatGPT :)
        "-vf",
        "scale=w=if(lt(iw\,ih)\,448\,-2):h=if(lt(iw\,ih)\,-2\,448)",  # noqa
        # '-c:a', 'copy',
        # '-c:v', 'h264_nvenc',
        # '-b:v', '5M',
        tgt_path,
        "-y",
    ]
    print(" ".join(cmd))
    os.makedirs(os.path.dirname(tgt_path), exist_ok=True)
    sp.call(cmd)


def process_all(paths):
    global ROOT_DIR
    global DS_TAKES_DIR
    map_fn = call_ffmpeg
    with ThreadPoolExecutor(5) as pool:
        for _ in tqdm(
            pool.map(map_fn, paths), total=len(paths), desc="Processing takes"
        ):
            continue


def main():
    with open(DEMONSTRATOR_TRAIN_DIR, "r") as f:
        train_data = json.load(f)
    with open(DEMONSTRATOR_VAL_DIR, "r") as f:
        val_data = json.load(f)
    with open(DEMONSTRATOR_TEST_DIR, "r") as f:
        test_data = json.load(f)

    required_takes = []
    takes_to_process = train_data + val_data + test_data
    for datum in takes_to_process:
        take_dir = os.path.dirname(datum["video_paths"]["ego"])
        assert take_dir.startswith("takes/")
        take_dir = take_dir[len("takes/") :]
        take_dir = os.path.join(DS_TAKES_DIR, take_dir)
        required_takes.append(datum)

    num_machines = 50
    root_dir: str = ROOT_DIR
    ds_take_dir: str = DS_TAKES_DIR

    map_values = []
    for take_datum in required_takes:
        for _, path in take_datum["video_paths"].items():
            assert path.startswith("takes/")
            src_path = os.path.join(root_dir, path)
            tgt_path = os.path.join(ds_take_dir, path.replace("takes/", ""))
            assert os.path.exists(src_path)
            map_values.append((src_path, tgt_path))

    print(f"# videos to process: {len(map_values)}")
    job_inputs = batch_it(
        map_values, batch_size=math.ceil(len(map_values) / num_machines)
    )
    num_machines = min(num_machines, len(job_inputs))

    dt_now = datetime.datetime.now().strftime("%y%m%d_%H%M")
    log_dir = f"downscale/{dt_now}"
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
