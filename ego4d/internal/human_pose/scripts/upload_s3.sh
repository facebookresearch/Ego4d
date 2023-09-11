#!/bin/bash

data=$1

# date=20230910
date=$2

# queue=pilot
queue=production

set -e

# cache_root_dir="/checkpoint/${USER}/datasets/EgoExoPose/tmp_0_80000_1"
cache_root_dir="/large_experiments/egoexo/egopose/${USER}/tmp_0_80000_1"

aws s3 sync "${cache_root_dir}/cache/${data}/body/halo" "s3://ego4d-fair/egopose/${queue}/${date}/${data}/body"

aws s3 sync "${cache_root_dir}/cache/${data}/hand/halo" "s3://ego4d-fair/egopose/${queue}/${date}/${data}/hand"

aws s3 sync "${cache_root_dir}/cache/${data}/vis_pose3d" "s3://ego4d-fair/egopose/${queue}/${date}/${data}"
