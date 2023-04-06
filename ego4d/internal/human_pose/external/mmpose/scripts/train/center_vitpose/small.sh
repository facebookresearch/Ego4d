cd ../../..

## need to export NCCL_P2P_DISABLE=1 to make it work on A6000 gpus. export to bashrc

###--------------------------------------------------------------
# DEVICES=0,1,
# DEVICES=0,1,2,3,
# DEVICES=4,5,6,7,
DEVICES=0,1,2,3,4,5,6,7,

RUN_FILE='./tools/dist_train.sh'
PORT=$(( ((RANDOM<<15)|RANDOM) % 63001 + 2000 ))

# ###-----------------------------w32 256 x 192---------------------------
CONFIG_FILE='configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/center_ViTPose_small_coco_256x192.py'
# PRETRAINED='/media/rawalk/disk1/rawalk/vitpose/pretrained/small_pretrained.pth'
PRETRAINED='/media/rawalk/disk1/rawalk/vitpose/checkpoints/vitpose_small.pth'


###--------------------------------------------------------------
## add timestamp to output dir
OUTPUT_DIR="$(echo "Outputs/train/center_vitpose_small_coco_256x192/$(date +"%m-%d-%Y_%H:%M:%S")")"

# TRAIN_BATCH_SIZE_PER_GPU=64 ### default
TRAIN_BATCH_SIZE_PER_GPU=128
# TRAIN_BATCH_SIZE_PER_GPU=224

###--------------------------------------------------------------
OPTIONS="$(echo "model.pretrained=$PRETRAINED data.samples_per_gpu=$TRAIN_BATCH_SIZE_PER_GPU")"

##--------------------------------------------------------------
# mode='debug'
mode='multi-gpu'

##--------------------------------------------------------------
## if mode is multi-gpu, then run the following
## else run the debugging on a single gpu
if [ "$mode" = "debug" ]; then
    TRAIN_BATCH_SIZE_PER_GPU=8 ## works for single gpu
    OPTIONS="$(echo "model.pretrained=$PRETRAINED data.samples_per_gpu=${TRAIN_BATCH_SIZE_PER_GPU} data.workers_per_gpu=0")"

    CUDA_VISIBLE_DEVICES=${DEVICES} python tools/train.py ${CONFIG_FILE} --work-dir ${OUTPUT_DIR} --no-validate --cfg-options ${OPTIONS}

elif [ "$mode" = "multi-gpu" ]; then
    NUM_GPUS_STRING_LEN=${#DEVICES}
    NUM_GPUS=$((NUM_GPUS_STRING_LEN/2))
    SEED='0'

    LOG_FILE="$(echo "${OUTPUT_DIR}/log.txt")"
    mkdir -p ${OUTPUT_DIR}; touch ${LOG_FILE}

    CUDA_VISIBLE_DEVICES=${DEVICES} PORT=${PORT} ${RUN_FILE} ${CONFIG_FILE} \
                ${NUM_GPUS} \
                --work-dir ${OUTPUT_DIR} \
                --seed ${SEED} \
                --cfg-options ${OPTIONS} \
                | tee ${LOG_FILE}
fi

