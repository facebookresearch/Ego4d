#!/bin/bash
#############################################################################################
# This bash file runs all body&hand pose 2d&3d estimation using Jinxu's pipeline, and
# tracks the running time of each mode and store the summary in [SAVE_PATH]/[TAKE_NAME].txt
#############################################################################################

############### MODIFY ###############
CONFIG_NAME="sfu_cooking_007_3"
TAKE_NAME="sfu_cooking_007_3"
SAVE_PATH='handPose_time_log/'
# MODE="preprocess
#       body_bbox
#       body_pose2d
#       wholebodyHand_pose3d
#       hand_pose2d_exo
#       hand_pose2d_ego
#       hand_pose3d_exo
#       hand_pose3d_egoexo
#       "
MODE="body_pose2d
      wholebodyHand_pose3d
      hand_pose2d_exo
      hand_pose2d_ego"
######################################

# Set-up
mkdir -p $SAVE_PATH
very_start=`date +%s`

# Iterate each mode in the pipeline
for mode in $MODE
do
    echo ============================ $mode starts ============================
    start=`date +%s`
    python3 ego4d/internal/human_pose/main.py --config-name $CONFIG_NAME mode=$mode
    end=`date +%s`
    curr_time=`expr $end - $start`
    echo $mode: "$(($curr_time / 3600))hrs $((($curr_time / 60) % 60))min $(($curr_time % 60))sec" >> $SAVE_PATH$TAKE_NAME.txt
    echo ============================ $mode finished ============================ $'\n\n'
done

# # Total running time
# very_end=`date +%s`
# total_time=`expr $very_end - $very_start`
# final_summary="\nElapsed: $(($total_time / 3600))hrs $((($total_time / 60) % 60))min $(($total_time % 60))sec"
# echo -e $final_summary >> $SAVE_PATH$CONFIG_NAME.txt