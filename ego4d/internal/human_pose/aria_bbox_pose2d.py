import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import cv2
import hydra
import glob
import numpy as np
from PIL import Image
import pandas as pd
from ego4d.internal.colmap.preprocess import download_andor_generate_streams

from ego4d.internal.human_pose.bbox_detector import DetectorModel
from ego4d.internal.human_pose.camera import (
    batch_xworld_to_yimage,
    create_camera,
    create_camera_data,
    get_aria_camera_models,
)
from ego4d.internal.human_pose.config import Config
from ego4d.internal.human_pose.dataset import (
    get_synced_timesync_df,
    SyncedEgoExoCaptureDset,
)
from ego4d.internal.human_pose.pose_estimator import PoseModel
from ego4d.internal.human_pose.readers import read_frame_idx_set
from ego4d.internal.human_pose.triangulator import Triangulator
from ego4d.internal.human_pose.utils import (
    check_and_convert_bbox,
    draw_bbox_xyxy,
    draw_points_2d,
    get_exo_camera_plane,
    get_region_proposal,
    left_right_bboxes_div
)

from iopath.common.file_io import PathManager
from iopath.common.s3 import S3PathHandler
from tqdm.auto import tqdm

from mmdet.apis import inference_detector, init_detector

pathmgr = PathManager()



########### Parameter ###########
test_case = 'iu_music'
test_dataset = 'dataset'
#################################

dset = SyncedEgoExoCaptureDset(
        data_dir='/mnt/shared/ego4dData',
        dataset_json_path=f'/mnt/shared/ego4dData/cache/{test_case}/{test_dataset}/data.json',
        read_frames=False,
    )
# Aria rotate save directory
rotImg_output_path = f'/mnt/shared/ego4dData/cache/{test_case}/{test_dataset}/frames/aria01/aria_rgb'
os.makedirs(rotImg_output_path, exist_ok=True)

# Load hand bounding box detector
config_file = 'external/mmlab/mmpose/demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_1class.py'
checkpoint_file = 'https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth'
detector = init_detector(config_file, checkpoint_file, device='cuda:0')

# Load hand pose estimation model
###### hrnetv2_w18_coco_wholebody_hand_256x256_dark ######
# hand_pose_config = 'external/mmlab/mmpose/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/coco_wholebody_hand/hrnetv2_w18_coco_wholebody_hand_256x256_dark.py'
# hand_pose_ckpt = 'https://download.openmmlab.com/mmpose/hand/dark/hrnetv2_w18_coco_wholebody_hand_256x256_dark-a9228c9c_20210908.pth'
###### res50_onehand10k_256x256 ######
hand_pose_config = 'external/mmlab/mmpose/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/onehand10k/res50_onehand10k_256x256.py'
hand_pose_ckpt = 'https://download.openmmlab.com/mmpose/top_down/resnet/res50_onehand10k_256x256-e67998f6_20200813.pth'
###### hrnetv2_dark_onehand10k ######
# hand_pose_config = 'external/mmlab/mmpose/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/onehand10k/hrnetv2_w18_onehand10k_256x256_dark.py'
# hand_pose_ckpt = 'https://download.openmmlab.com/mmpose/hand/dark/hrnetv2_w18_onehand10k_256x256_dark-a2f80c64_20210330.pth'
hand_pose_model = PoseModel(hand_pose_config, hand_pose_ckpt, rgb_keypoint_thres=0.3, rgb_keypoint_vis_thres=0.3)



################################ Find hand bbox ################################
# all_img_dir = glob.glob(rotImg_output_path + '/*')
bbox_output_path = f'/mnt/shared/ego4dData/cache/{test_case}/{test_dataset}/hand/bbox'
vis_bbox_path = f'/mnt/shared/ego4dData/cache/{test_case}/{test_dataset}/hand/vis_bbox/aria01'
os.makedirs(bbox_output_path, exist_ok=True)
os.makedirs(vis_bbox_path, exist_ok=True)

def get_two_small_bboxes(bboxes):
    """
    Since the hand bbox detection might contain multiple results where one bbox capture both hands, while we want each bbox
    to contain only one hand, we select most two small bboxes from multiple results.
    """
    bbox_area = (bboxes[:,2] - bboxes[:,0]) * (bboxes[:,3] - bboxes[:,1])
    min_idx = np.argsort(bbox_area)[:2]
    return bboxes[min_idx,:4]

# Iterate through all aria rotated images
bboxes = {}
for idx in tqdm(range(len(dset))):
    image_path = os.path.join(rotImg_output_path, os.path.basename(dset[idx]['aria_rgb']['abs_frame_path']))
    # Inference 
    det_results = inference_detector(detector, image_path)
    
    # Refine detected bboxes to only two bboxes
    refined_two_bboxes = get_two_small_bboxes(det_results[0])
    right_left_bboxes = left_right_bboxes_div(refined_two_bboxes) # (2,4)

    # Visualization
    vis_bbox_img = cv2.imread(image_path)
    color_map = [(255,0,0),(0,255,0)] 
    for ii, curr_hand_bbox in enumerate(right_left_bboxes):
        vis_bbox_img = draw_bbox_xyxy(vis_bbox_img, curr_hand_bbox, color=color_map[ii])
    cv2.imwrite(os.path.join(vis_bbox_path, f'{idx:06d}.jpg'), vis_bbox_img)

    # Save bbox result
    bboxes[idx] = [bbox for bbox in right_left_bboxes] # Follow same convention as mode_hand_pose2d to save hand bbox

# save the bboxes as a pickle file
with open(os.path.join(bbox_output_path, "aria_bbox.pkl"), "wb") as f:
    pickle.dump(bboxes, f)
####################################################################################


################################ Hand 2d estimation ################################
# Load in bbox and all image path
bbox_file = f'/mnt/shared/ego4dData/cache/{test_case}/{test_dataset}/hand/bbox/aria_bbox.pkl'
with open(bbox_file, "rb") as f:
    bboxes = pickle.load(f)

# Save directory
pose2d_dir = f'/mnt/shared/ego4dData/cache/{test_case}/{test_dataset}/hand/pose2d'
os.makedirs(pose2d_dir, exist_ok=True)
vis_pose2d_dir = f'/mnt/shared/ego4dData/cache/{test_case}/{test_dataset}/hand/vis_pose2d/aria01'
os.makedirs(vis_pose2d_dir, exist_ok=True)

poses2d = {}
# Iterate thorugh every aria images 
for idx in tqdm(range(len(dset))):
    image_path = os.path.join(rotImg_output_path, os.path.basename(dset[idx]['aria_rgb']['abs_frame_path']))
    two_hand_bboxes = [{'bbox':np.append(curr_hand_bbox,1)} for curr_hand_bbox in bboxes[idx]]

    # Hand pose estimation
    pose_results = hand_pose_model.get_poses2d(
                        bboxes=two_hand_bboxes,
                        image_name=image_path,
                    )
    
    # Save result
    curr_pose2d_kpts = np.array([res['keypoints'] for res in pose_results])
    poses2d[idx] = curr_pose2d_kpts

    # Visualization
    save_path = f'{vis_pose2d_dir}/{idx:06d}.jpg'
    vis_twoHand = cv2.imread(image_path)
    hand_pose_model.draw_poses2d([pose_results[0]], vis_twoHand, save_path)
    vis_twoHand = cv2.imread(save_path)
    hand_pose_model.draw_poses2d([pose_results[1]], vis_twoHand, save_path)

# save poses2d key points result
with open(os.path.join(pose2d_dir, "aria_pose2d.pkl"), "wb") as f:
    pickle.dump(poses2d, f)
####################################################################################