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


ROOT_DIR = "/large_experiments/egoexo/v2/"
DS_TAKES_DIR = "/checkpoint/miguelmartin/egoexo/v2/downscaled_takes/takes_by_uni"


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
    map_fn = call_ffmpeg
    with ThreadPoolExecutor(5) as pool:
        for _ in tqdm(
            pool.map(map_fn, paths), total=len(paths), desc="Processing takes"
        ):
            continue


def main():
    num_machines: int = 50
    root_dir: str = ROOT_DIR
    ds_take_dir: str = DS_TAKES_DIR

    takes_to_process = json.load(open(os.path.join(root_dir, "takes.json")))

    map_values = []
    completed = 0
    num_vids = 0
    for take in takes_to_process:
        for _, streams in take["frame_aligned_videos"].items():
            for _, stream in streams.items():
                rel_path = stream["relative_path"]
                if rel_path is None:
                    continue
                if stream["is_collage"]:
                    continue
                src_path = os.path.join(root_dir, take["root_dir"], rel_path)
                dst_path = os.path.join(
                    ds_take_dir, take["university_id"], take["take_name"], rel_path
                )
                assert os.path.exists(src_path), src_path
                num_vids += 1
                if os.path.exists(dst_path):
                    completed += 1
                    continue
                map_values.append((src_path, dst_path))

    print(
        f"# videos to process: {len(map_values)} / {num_vids} [{completed} / {1 - (len(map_values) / num_vids):.2%} completed]"
    )
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
