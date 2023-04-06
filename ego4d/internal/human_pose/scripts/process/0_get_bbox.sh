cd ../..

##--------------------------------------------------------------
DEVICES=0,
RUN_FILE=ego4d/internal/human_pose/main.py
CONFIG=unc_T1_rawal

# ##--------------------------------------------------------------
# CUDA_VISIBLE_DEVICES=${DEVICES} python $RUN_FILE \
#                     --sequence_path ${SEQUENCE_PATH} \
#                     --output_path $OUTPUT_DIR \
#                     --start_time $START_TIME \
#                     --end_time $END_TIME \
#                     --choosen_time $TIMESTAMPS \


CUDA_VISIBLE_DEVICES=${DEVICES} python main.py --config-name $CONFIG mode=bbox