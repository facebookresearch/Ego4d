#!/bin/bash
# This script is supposed to run on devserver

data=$1

aws_date=$2
# date=20230910
date=$2

# queue=pilot
queue=production

manifold mkdir "ego4d_fair/tree/egoexo/egopose/${queue}/${date}"
manifold mkdir "ego4d_fair/tree/egoexo/egopose/${queue}/${date}/${data}"

set -e

export https_proxy=fwdproxy:8080
aws s3 sync "s3://ego4d-fair/egopose/${queue}/${aws_date}/${data}" ~/egopose/"${queue}/${date}/${data}"

manifold putr --ignoreExisting -j 10 --threads 1 ~/egopose/"${queue}/${date}/${data}/body" "ego4d_fair/tree/egoexo/egopose/${queue}/${date}/${data}/body"

manifold putr --ignoreExisting -j 10 --threads 1 ~/egopose/"${queue}/${date}/${data}/hand" "ego4d_fair/tree/egoexo/egopose/${queue}/${date}/${data}/hand"

rm -rf ~/egopose/"${queue}/${date}/${data}/body"
rm -rf ~/egopose/"${queue}/${date}/${data}/hand"

manifold put ~/egopose/"${queue}/${date}/${data}/body_bbox.mp4" "ego4d_fair/tree/egoexo/egopose/${queue}/${date}/${data}/body_bbox.mp4"
manifold put ~/egopose/"${queue}/${date}/${data}/body_pose2d.mp4" "ego4d_fair/tree/egoexo/egopose/${queue}/${date}/${data}/body_pose2d.mp4"
manifold put ~/egopose/"${queue}/${date}/${data}/body_pose3d.mp4" "ego4d_fair/tree/egoexo/egopose/${queue}/${date}/${data}/body_pose3d.mp4"
manifold put ~/egopose/"${queue}/${date}/${data}/hand_pose2d.mp4" "ego4d_fair/tree/egoexo/egopose/${queue}/${date}/${data}/hand_pose2d.mp4"
manifold put ~/egopose/"${queue}/${date}/${data}/hand_pose3d.mp4" "ego4d_fair/tree/egoexo/egopose/${queue}/${date}/${data}/hand_pose3d.mp4"
