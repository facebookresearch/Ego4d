#!/bin/bash

############### MODIFY ###############
CONFIG_NAME="iiith_cooking_01_1"
TAKE_NAME="iiith_cooking_01_1"
MODE="body_pose2d
      body_pose3d
      wholebodyHand_pose3d
      hand_pose2d_exo
      hand_pose2d_ego
      hand_pose3d_exo
      hand_pose3d_egoexo
      "
######################################

# Set-up
very_start=`date +%s`

# Iterate each mode in the pipeline
for mode in $MODE
do
    echo ============================ $mode starts ============================
    start=`date +%s`
    python3 ego4d/internal/human_pose/main.py --config-name $CONFIG_NAME mode=$mode
    end=`date +%s`
    curr_time=`expr $end - $start`
    echo $mode: "$(($curr_time / 3600))hrs $((($curr_time / 60) % 60))min $(($curr_time % 60))sec"
    echo ============================ $mode finished ============================ $'\n\n'
done
