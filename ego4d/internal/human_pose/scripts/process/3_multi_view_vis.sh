cd ../..

##--------------------------------------------------------------
DEVICES=0,
RUN_FILE=ego4d/internal/human_pose/main.py
# CONFIG=unc_T1_rawal
CONFIG=iu_bike_rawal
# CONFIG=iu_music_rawal


# ##--------------------------------------------------------------
CUDA_VISIBLE_DEVICES=${DEVICES} python main.py --config-name $CONFIG mode=multi_view_vis