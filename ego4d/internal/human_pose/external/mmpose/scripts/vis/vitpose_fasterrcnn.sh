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
IMAGE_ROOT='/home/rawalk/Desktop/ego/cliff/data/test_samples/hugging1/imgs'
OUTPUT_IMAGES='/home/rawalk/Desktop/ego/cliff/data/test_samples/hugging1/vis_2dpose/'
OUTPUT_POSES='/home/rawalk/Desktop/ego/cliff/data/test_samples/hugging1/2dpose'

###--------------------------------------------------------------
CUDA_VISIBLE_DEVICES=${DEVICES} python ${RUN_FILE} \
    ${DETECTION_CONFIG_FILE} \
    ${DETECTION_CHECKPOINT} \
    ${CONFIG_FILE} \
    ${CHECKPOINT} \
    --img-root ${IMAGE_ROOT} \
    --out-img-root ${OUTPUT_IMAGES} \
    --out-pose-root ${OUTPUT_POSES} \
    --thickness ${LINE_THICKNESS} \