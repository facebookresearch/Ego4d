import json
import os
import pickle
import numpy as np

########### Parameter ###########
test_case = 'iu_music'
test_dataset = 'dataset'
#################################

# data_dir = '/media/jinxu/New Volume/ego4dData'
# dataset_dir = '/media/jinxu/New Volume/ego4dData/cache/iu_music/dataset'

# dset = SyncedEgoExoCaptureDset(
#     data_dir='/media/jinxu/New Volume/ego4dData',
#     dataset_json_path=f'/media/jinxu/New Volume/ego4dData/cache/{test_case}/{test_dataset}/data.json',
#     read_frames=False,
# )

# # Body pose2d model
# pose_config = "external/mmlab/mmpose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py"
# pose_checkpoint = "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth"
# body_pose_model = PoseModel(
#     pose_config, 
#     pose_checkpoint, 
#     rgb_keypoint_thres=0.7, 
#     rgb_keypoint_vis_thres=0.7)


bbox_dir = os.path.join(f'/media/jinxu/New Volume/ego4dData/cache/{test_case}/{test_dataset}', 'body/bbox')
bbox_file = os.path.join(bbox_dir, "bbox.pkl")
with open(bbox_file, "rb") as f:
    bboxes = pickle.load(f)


print(bboxes[0])

