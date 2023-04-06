cd ../..

# RUN_FILE='demo/image_demo.py'
RUN_FILE='demo/image_folder_demo.py'

##-------------------------------------------------------------------------
CONFIG='configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py'
CHECKPOINT='https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth'

IMAGE_SIZE=1024

DEVICES=0,1,2,3,

##-------------------------------------------------------------------------
# CAMERAS=('cam01' 'cam02' 'cam03' 'cam04' 'cam05' 'cam06' 'cam07' 'cam08' 'cam09' 'cam10' 'cam11' 'cam12' 'cam13' 'cam14' 'cam15' 'cam16' 'cam17' 'cam18' 'cam19' 'cam20' 'cam21' 'cam22')
# CAMERAS=('cam01' 'cam02' 'cam03' 'cam04' 'cam05' 'cam06' 'cam07' 'cam08')
# CAMERAS=('cam09' 'cam10' 'cam11' 'cam12' 'cam13' 'cam14' 'cam15' 'cam16')
CAMERAS=('cam17' 'cam18' 'cam19' 'cam20' 'cam21' 'cam22')

NUM_GPUS=$(echo $DEVICES | tr -cd , | wc -c)

# PER_GPU_CAMERAS=$(( ${#CAMERAS[@]} / $NUM_GPUS ))
PER_GPU_CAMERAS=2

echo "Num GPUs: $NUM_GPUS, Total Cameras: ${#CAMERAS[@]}, Per GPU Cameras: $PER_GPU_CAMERAS"

## run background processes per gpu using the split cameras
for (( i=0; i<$NUM_GPUS; i++ ))
do
    START_CAMERA=$(( $i * $PER_GPU_CAMERAS ))

    ## if last gpu, then add the remaining cameras
    if [ $i -eq $(( $NUM_GPUS - 1 )) ]
    then
        PER_GPU_CAMERAS=$(( $PER_GPU_CAMERAS + ${#CAMERAS[@]} % $NUM_GPUS ))
    fi

    END_CAMERA=$(( $START_CAMERA + $PER_GPU_CAMERAS ))
    echo "GPU $i: ${CAMERAS[@]:$START_CAMERA:$PER_GPU_CAMERAS}"

    ## inner camera loop from START_CAMERA to END_CAMERA
    for (( j=$START_CAMERA; j<$END_CAMERA; j++ ))
    do
        ## if camera is null, then skip
        if [ -z "${CAMERAS[$j]}" ]
        then
            continue
        fi

        IMG_FOLDER=/media/rawalk/disk1/rawalk/datasets/ego_exo/main/13_hugging/001_hugging/exo/${CAMERAS[$j]}/images
        OUT_FOLDER=/media/rawalk/disk1/rawalk/datasets/ego_exo/main/13_hugging/001_hugging/processed_data/vis_mask/${CAMERAS[$j]}/rgb

        echo "Processing ${CAMERAS[$j]}"

        CUDA_VISIBLE_DEVICES=$i python $RUN_FILE $CONFIG $CHECKPOINT --img_folder $IMG_FOLDER --out_folder $OUT_FOLDER --max_image_size $IMAGE_SIZE &
    done

done

