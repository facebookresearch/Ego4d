cd ../..

###--------------------------------------------------------------
DEVICES=3,
RUN_FILE='demo/custom_bottom_up_img_demo.py'

CONFIG_FILE='configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/higherhrnet_w48_coco_512x512_udp.py'
CHECKPOINT='https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet48_coco_512x512_udp-7cad61ef_20210222.pth'

LINE_THICKNESS=5 ## 1 is default

###--------------------------------------------------------------
READ_DIR='/media/rawalk/disk1/rawalk/datasets/ego_exo/main/13_hugging/001_hugging/exo'
OUT_DIR='/media/rawalk/disk1/rawalk/datasets/ego_exo/main/13_hugging/001_hugging/processed_data/vis_higherhrnet_poses2d'

CAMERAS=('cam01' 'cam02' 'cam03' 'cam04' 'cam05' 'cam06' 'cam07' 'cam08' 'cam09' 'cam10' 'cam11' 'cam12' 'cam13' 'cam14' 'cam15' 'cam16' 'cam17' 'cam18' 'cam19' 'cam20' 'cam21' 'cam22')
# CAMERAS=('cam01')

for CAMERA in "${CAMERAS[@]}"
do
    IMAGE_ROOT="${READ_DIR}/${CAMERA}/images/"
    OUTPUT_IMAGES="${OUT_DIR}/${CAMERA}/rgb"
    OUTPUT_POSES="${OUT_DIR}/${CAMERA}/pose2d/"

    echo $IMAGE_ROOT
    echo $OUTPUT_IMAGES

    # ###--------------------------------------------------------------
    CUDA_VISIBLE_DEVICES=${DEVICES} python ${RUN_FILE} \
        ${CONFIG_FILE} \
        ${CHECKPOINT} \
        --img-root ${IMAGE_ROOT} \
        --out-img-root ${OUTPUT_IMAGES} \
        --out-pose-root ${OUTPUT_POSES} \
        --thickness ${LINE_THICKNESS} 

done

