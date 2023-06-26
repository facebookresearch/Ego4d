cd ../..

##--------------------------------------------------------------
DEVICES=0,
RUN_FILE=ego4d/internal/human_pose/main.py
# CONFIG=unc_T1_rawal
CONFIG=iu_bike_rawal
# CONFIG=iu_music_rawal
# CONFIG=cmu_soccer_rawal; DEVICES=0,


# ##--------------------------------------------------------------
# CUDA_VISIBLE_DEVICES=${DEVICES} python main.py --config-name $CONFIG mode=multi_view_vis_bbox
# CUDA_VISIBLE_DEVICES=${DEVICES} python main.py --config-name $CONFIG mode=multi_view_vis_pose2d
CUDA_VISIBLE_DEVICES=${DEVICES} python main.py --config-name $CONFIG mode=multi_view_vis_pose3d
