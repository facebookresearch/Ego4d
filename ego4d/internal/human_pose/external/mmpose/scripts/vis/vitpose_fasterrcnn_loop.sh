cd ../..

###--------------------------------------------------------------
DEVICES=3,
RUN_FILE='demo/custom_top_down_img_demo_with_mmdet.py'

DETECTION_CONFIG_FILE='demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
DETECTION_CHECKPOINT='https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# CONFIG_FILE='configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py'
# CHECKPOINT='https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth'

# CONFIG_FILE='configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitPose+_huge_coco+aic+mpii+ap10k+apt36k+wholebody_256x192_udp.py'
# CHECKPOINT='/media/rawalk/disk1/rawalk/vitpose/checkpoints/vitpose+_huge.pth'

CONFIG_FILE='configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py'
CHECKPOINT='/media/rawalk/disk1/rawalk/vitpose/checkpoints/vitpose-h-multi-coco.pth'

LINE_THICKNESS=3 ## 1 is default

###--------------------------------------------------------------
READ_DIR='/media/rawalk/disk1/rawalk/datasets/ego_exo/main/13_hugging/001_hugging/exo'
OUT_DIR='/media/rawalk/disk1/rawalk/datasets/ego_exo/main/13_hugging/001_hugging/processed_data/vis_vitpose_poses2d'

CAMERAS=('cam01' 'cam02' 'cam03' 'cam04' 'cam05' 'cam06' 'cam07' 'cam08' 'cam09' 'cam10' 'cam11' 'cam12' 'cam13' 'cam14' 'cam15' 'cam16' 'cam17' 'cam18' 'cam19' 'cam20' 'cam21' 'cam22')


for CAMERA in "${CAMERAS[@]}"
do
    IMAGE_ROOT="${READ_DIR}/${CAMERA}/images/"
    OUTPUT_IMAGES="${OUT_DIR}/${CAMERA}/rgb"
    OUTPUT_POSES="${OUT_DIR}/${CAMERA}/pose2d/"

    echo $IMAGE_ROOT
    echo $OUTPUT_IMAGES

    # ###--------------------------------------------------------------
    CUDA_VISIBLE_DEVICES=${DEVICES} python ${RUN_FILE} \
        ${DETECTION_CONFIG_FILE} \
        ${DETECTION_CHECKPOINT} \
        ${CONFIG_FILE} \
        ${CHECKPOINT} \
        --img-root ${IMAGE_ROOT} \
        --out-img-root ${OUTPUT_IMAGES} \
        --out-pose-root ${OUTPUT_POSES} \
        --thickness ${LINE_THICKNESS} 

done

