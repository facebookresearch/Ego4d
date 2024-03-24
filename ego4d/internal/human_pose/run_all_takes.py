import os
import sys
import pandas as pd

batch_job_name="project_retriangulation_production_v2"
frame_start=0
frame_end=1000000000
min_area_ratio=0.003
storage_level=30
USER="suyogjain"
cache_root_dir=f"/large_experiments/egoexo/egopose/{USER}/project_retriangulation_production_v2/"
data_dir="/private/home/suyogjain/egoexo/jan24/data"
partition="learnaccel"
run_type="submitit"
steps="extract_camera_info"

def chunk_list(input_list, chunk_size):
    # Create a list of lists, where each sublist has chunk_size elements
    return [
        input_list[i : i + chunk_size] for i in range(0, len(input_list), chunk_size)
    ]

def get_cmd(take_list):
    take_name = '+'.join(take_list)
    take_name = take_name.lstrip('+')
    cmd = list()
    cmd.append(f"python launch_main.py")
    cmd.append(f"--base_work_dir /checkpoint/{USER}/flow/ego4d")
    cmd.append(f"--batch_job_name {batch_job_name}")
    cmd.append(f"--config-name dev_release_base")
    cmd.append(f"--partition {partition}")
    cmd.append(f"--run_type {run_type}")
    cmd.append(f"--steps {steps}")
    cmd.append(f"--take_name {take_name}")
    cmd.append(f"cache_root_dir={cache_root_dir}")
    cmd.append(f"data_dir={data_dir}")
    cmd.append(f"inputs.from_frame_number={frame_start}")
    cmd.append(f"inputs.to_frame_number={frame_end}")
    cmd.append(f"inputs.sample_interval=")
    cmd.append(f"inputs.subclip_json_dir=/checkpoint/jinghuang/datasets/EgoExoPose/subclip_info/dummy")
    cmd.append(f"mode_preprocess.vrs_bin_path=/private/home/jinghuang/code/vrs/build/tools/vrs/vrs")
    cmd.append(f"mode_bbox.min_area_ratio={min_area_ratio}")
    cmd.append(f"outputs.storage_level={storage_level}")
    cmd.append(f"repo_root_dir=/private/home/{USER}/code/Ego4d")

    cmd = ' '.join(cmd)
    print(cmd)
    return cmd

release_takes = pd.read_csv('/private/home/suyogjain/egoexo/jan24/release_take_metadata.csv')
take_names = list(release_takes['take_name'])
take_names = [x for x in take_names if 'bouldering' not in x]

take_chunks = chunk_list(take_names, 75)

for idx, take_list in enumerate(take_chunks):    
    print(idx)
    cmd = get_cmd(take_list)
    os.system(cmd)
    user_input = input("Please enter to kick of the next batch:")
    print(f"Enqueing next batch now..")    




