#!/bin/bash

data=$1

date=20230831

# queue=pilot
queue=production

set -e

aws s3 sync "/checkpoint/${USER}/datasets/EgoExoPose/tmp_0_80000_1/cache/${data}/body/halo" "s3://ego4d-fair/egopose/${queue}/${date}/${data}/body"

aws s3 sync "/checkpoint/${USER}/datasets/EgoExoPose/tmp_0_80000_1/cache/${data}/hand/halo" "s3://ego4d-fair/egopose/${queue}/${date}/${data}/hand"

aws s3 sync "/checkpoint/${USER}/datasets/EgoExoPose/tmp_0_80000_1/cache/${data}/vis_pose3d" "s3://ego4d-fair/egopose/${queue}/${date}/${data}"
