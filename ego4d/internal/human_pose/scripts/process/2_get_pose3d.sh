cd ../..

##--------------------------------------------------------------
RUN_FILE=ego4d/internal/human_pose/main.py

CONFIG=unc_T1_rawal; DEVICES=0,
# CONFIG=iu_bike_rawal; DEVICES=0,
# CONFIG=iu_music_rawal; DEVICES=0,
# CONFIG=cmu_soccer_rawal; DEVICES=0,

# ##--------------------------------------------------------------
# CUDA_VISIBLE_DEVICES=${DEVICES} python main.py --config-name $CONFIG mode=pose3d

# ##--------------------------------------------------------------
## read the yaml file and get the start and end frame
start_frame=$(python -c "import yaml; file = open('configs/${CONFIG}.yaml', 'r'); config = yaml.safe_load(file); file.close(); print(config['inputs']['from_frame_number'])")
end_frame=$(python -c "import yaml; file = open('configs/${CONFIG}.yaml', 'r'); config = yaml.safe_load(file); file.close(); print(config['inputs']['to_frame_number'])")
TOTAL_FRAMES=$(($end_frame-$start_frame + 1))

NUM_JOBS=8

## start num_jobs, each job runs for 1/num_jobs of the total frames, set the start and end frame accordingly
## start from 0 to TOTAL_FRAMES

for ((i=0; i<$NUM_JOBS; i++))
do
    start=$(($i*$TOTAL_FRAMES/$NUM_JOBS))
    end=$(($(($i+1))*$TOTAL_FRAMES/$NUM_JOBS))
    echo "start: $start, end: $end"
    CUDA_VISIBLE_DEVICES=${DEVICES} python main.py --config-name $CONFIG mode=pose3d mode_pose3d.start_frame=$start mode_pose3d.end_frame=$end &
done
