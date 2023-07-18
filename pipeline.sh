#!/bin/bash
############### MODIFY ###############
CONFIG_NAME="iu_music_jinxu"
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
MODE="body_bbox
      body_pose2d
      wholebodyHand_pose3d
      hand_pose2d_exo
      hand_pose2d_ego
      hand_pose3d_exo
      hand_pose3d_egoexo
      "
######################################
very_start=`date +%s`

# Iterate each mode in the pipeline
for mode in $MODE
do
    echo ================= Running $mode ==================
    start=`date +%s`
    python3 ego4d/internal/human_pose/main_jinxu.py --config-name $CONFIG_NAME mode=$mode
    end=`date +%s`
    curr_time=`expr $end - $start`
    echo $mode: "$(($curr_time / 3600))hrs $((($curr_time / 60) % 60))min $(($curr_time % 60))sec" >> $SAVE_PATH$CONFIG_NAME.txt
done

# # Total running time
# very_end=`date +%s`
# total_time=`expr $very_end - $very_start`
# final_summary="\nElapsed: $(($total_time / 3600))hrs $((($total_time / 60) % 60))min $(($total_time % 60))sec"
# echo -e $final_summary >> $SAVE_PATH$CONFIG_NAME.txt
