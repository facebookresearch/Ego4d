#!/bin/bash
####################################
EGO4D_ROOT=/home/jinxu/code/Ego4d
batch_job_name=egobodypose_gt_gen
cfg=dev_release_base_jinxu
frame_start=5000
frame_end=5100
take_name=georgiatech_covid_02_2+iiith_cooking_30_1+nus_cpr_24_2+uniandes_dance_007_3
cache_root_dir=/media/jinxu/New\ Volume/ego4dData
data_dir=/media/jinxu/New\ Volume/ego4dData
partition=learnaccel
run_type=submitit  # Note: to run/debug locally, use run_type=local instead
steps=preprocess+body_bbox+
###################################


cd ${EGO4D_ROOT}/ego4d/internal/human_pose

python launch_main.py \
--batch_job_name ${batch_job_name} \
--config-name ${cfg} \
--partition ${partition} \
--run_type ${run_type} \
--steps ${steps} \
--take_name ${take_name} \
cache_root_dir=${cache_root_dir} \
data_dir=${data_dir} \
inputs.from_frame_number=${frame_start} \
inputs.to_frame_number=${frame_end} \
mode_preprocess.vrs_bin_path=/home/jinxu/code/vrs/build/tools/vrs/vrs \
repo_root_dir=/home/jinxu/code/Ego4d