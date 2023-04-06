cd ../../..

## need to export NCCL_P2P_DISABLE=1 to make it work on A6000 gpus. export to bashrc

###--------------------------------------------------------------
# DEVICES=0,1,
DEVICES=0,1,2,3,4,5,6,7,

RUN_FILE='./tools/dist_test.sh'
PORT=$(( ((RANDOM<<15)|RANDOM) % 63001 + 2000 ))

# ###-----------------------------w32 256 x 192---------------------------
CONFIG_FILE='configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/center_ViTPose_base_coco_256x192.py'
OUTPUT_DIR='/home/rawalk/Desktop/ego/vitpose/Outputs/test/center_vitpose_base_coco_256x192'
TEST_BATCH_SIZE_PER_GPU=64 
CHECKPOINT='/home/rawalk/Desktop/ego/vitpose/Outputs/train/center_vitpose_base_coco_256x192/best_AP_epoch_210.pth'

###--------------------------------------------------------------
OPTIONS="$(echo "data.samples_per_gpu=$TEST_BATCH_SIZE_PER_GPU")"

# # #####---------------------multi-gpu training---------------------------------
NUM_GPUS_STRING_LEN=${#DEVICES}
NUM_GPUS=$((NUM_GPUS_STRING_LEN/2))

LOG_FILE="$(echo "${OUTPUT_DIR}/log.txt")"
mkdir -p ${OUTPUT_DIR}; touch ${LOG_FILE}

CUDA_VISIBLE_DEVICES=${DEVICES} PORT=${PORT} ${RUN_FILE} ${CONFIG_FILE} $CHECKPOINT\
			${NUM_GPUS} \
			--work-dir ${OUTPUT_DIR} \
			--cfg-options ${OPTIONS} \
			| tee ${LOG_FILE}


# # # # #####---------------------debugging on a single gpu---------------------------------
# TEST_BATCH_SIZE_PER_GPU=8 ## works for single gpu
# OPTIONS="$(echo "data.samples_per_gpu=${TEST_BATCH_SIZE_PER_GPU} data.workers_per_gpu=0")"

# CUDA_VISIBLE_DEVICES=${DEVICES} python tools/test.py ${CONFIG_FILE} $CHECKPOINT --work-dir ${OUTPUT_DIR} --cfg-options ${OPTIONS} \
