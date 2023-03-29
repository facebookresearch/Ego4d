cd ../..

##--------------------------------------------------------------
RUN_FILE=ego4d/internal/human_pose/main.py

CONFIG=unc_T1_rawal; DEVICES=0,
# CONFIG=iu_bike_rawal; DEVICES=0,
# CONFIG=iu_music_rawal; DEVICES=0,
# CONFIG=cmu_soccer_rawal; DEVICES=0,

# # ##--------------------------------------------------------------
# CUDA_VISIBLE_DEVICES=${DEVICES} python main.py --config-name $CONFIG mode=pose2d


# ##---------------------parallel process---------------------------
DEVICES=0,; CAMERA_NAME=cam01; CUDA_VISIBLE_DEVICES=${DEVICES} python main_single_camera.py --config-name $CONFIG mode=pose2d exo_camera_name=$CAMERA_NAME &
DEVICES=0,; CAMERA_NAME=cam02; CUDA_VISIBLE_DEVICES=${DEVICES} python main_single_camera.py --config-name $CONFIG mode=pose2d exo_camera_name=$CAMERA_NAME &
DEVICES=1,; CAMERA_NAME=cam03; CUDA_VISIBLE_DEVICES=${DEVICES} python main_single_camera.py --config-name $CONFIG mode=pose2d exo_camera_name=$CAMERA_NAME &
DEVICES=1,; CAMERA_NAME=cam04; CUDA_VISIBLE_DEVICES=${DEVICES} python main_single_camera.py --config-name $CONFIG mode=pose2d exo_camera_name=$CAMERA_NAME &
