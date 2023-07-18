import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import cv2
import numpy as np
import pandas as pd
import glob
from ego4d.internal.colmap.preprocess import download_andor_generate_streams
from ego4d.internal.human_pose.bbox_detector import DetectorModel
from ego4d.internal.human_pose.camera import (
    batch_xworld_to_yimage,
    create_camera,
    create_camera_data,
    create_customized_camera,
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
    # draw_points_2d,
    get_exo_camera_plane,
    get_region_proposal,
    get_bbox_fromKpts,
    get_two_small_bboxes,
    aria_rotate_kpts,
    left_right_bboxes_div,
    extri_intri_from_json,
    get_largest_bboxes
)
from iopath.common.file_io import PathManager
from iopath.common.s3 import S3PathHandler
from tqdm.auto import tqdm
from mmdet.apis import inference_detector, init_detector


def mode_body_bbox(args):
    """
    Detect human body bbox for only exo cameras for later body pose2d estimation
    """
    # Directory setup and load parameters
    dataset_root = args.dataset_root
    dataset_dir = os.path.join(dataset_root, f'{args.case}', 'dataset')
    calib_result_dir = os.path.join(dataset_root, f'{args.case}', 'dataset/calib_results')
    exo_cam_names = args.exo_cam_names

    # Human body bbox detector
    detector_config = "ego4d/internal/human_pose/external/mmlab/mmpose/demo/mmdetection_cfg/faster_rcnn_x101_64x4d_fpn_mstrain_3x_coco.py"
    detector_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_x101_64x4d_fpn_mstrain_3x_coco/faster_rcnn_x101_64x4d_fpn_mstrain_3x_coco_20210524_124528-26c63de6.pth'
    detector = init_detector(detector_config, detector_checkpoint, device='cuda:0')
    
    # Directory to store bbox result and visualization
    bbox_dir = os.path.join(dataset_dir, 'body/bbox')
    vis_bbox_dir = os.path.join(dataset_dir, 'body/vis_bbox')
    os.makedirs(bbox_dir, exist_ok=True)
    os.makedirs(vis_bbox_dir, exist_ok=True)

    # Set up image path loader
    dset = {curr_cam : sorted(glob.glob(os.path.join(dataset_dir, 'frames', curr_cam, '*'))) for curr_cam in exo_cam_names}
    # check number of images are consistent across different camera groups
    num_img = len(dset[next(iter(dset))])
    assert all(len(dset[k]) == num_img for k in dset), "Not all cameras have same number of images"


    # Iterate through every frames and generate body bbox
    bboxes = {}
    for time_stamp in tqdm(range(num_img), total=num_img):
        
        # Iterate through every cameras
        bboxes[time_stamp] = {}
        for exo_camera_name in exo_cam_names:
            # Load in image
            image_path = dset[exo_camera_name][time_stamp]
            
            # bbox visualization save dir
            vis_bbox_cam_dir = os.path.join(vis_bbox_dir, exo_camera_name)
            if not os.path.exists(vis_bbox_cam_dir):
                os.makedirs(vis_bbox_cam_dir)

            # Inference
            det_results = inference_detector(detector, image_path)
            curr_bbox = det_results[0].flatten()[:4] ######### Assume single person per frame!!!
            
            ############# Heuristics: Get the largest bbox as the one of interest #################
            # curr_bbox = get_largest_bboxes(det_results[0])
            #######################################################################################

            # Append result
            bboxes[time_stamp][exo_camera_name] = curr_bbox

            # Save visualization result
            original_img = cv2.imread(image_path)
            bbox_img = draw_bbox_xyxy(original_img, curr_bbox)
            cv2.imwrite(os.path.join(vis_bbox_cam_dir, f"{time_stamp:05d}.jpg"), bbox_img)

    # save the bboxes as a pickle file
    with open(os.path.join(bbox_dir, "bbox.pkl"), "wb") as f:
        pickle.dump(bboxes, f)
    print('=============== mode_bbox_jinxu finished ===============\n')



def mode_body_pose2d(args):
    # Directory setup and load parameters
    dataset_root = args.dataset_root
    dataset_dir = os.path.join(dataset_root, f'{args.case}', 'dataset')
    calib_result_dir = os.path.join(dataset_root, f'{args.case}', 'dataset/calib_results')
    exo_cam_names = args.exo_cam_names

    # Load body keypoints estimation model
    pose_config = "ego4d/internal/human_pose/external/mmlab/mmpose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py"
    pose_checkpoint = "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth"
    pose_model = PoseModel(
        pose_config=pose_config, pose_checkpoint=pose_checkpoint
    )

    # Create directory to store body pose2d results and visualization
    pose2d_dir = os.path.join(dataset_dir, 'body/pose2d')
    if not os.path.exists(pose2d_dir):
        os.makedirs(pose2d_dir)
    vis_pose2d_dir = os.path.join(dataset_dir, 'body/vis_pose2d')
    if not os.path.exists(vis_pose2d_dir):
        os.makedirs(vis_pose2d_dir)

    # load bboxes from bbox_dir/bbox.pkl
    bbox_dir = os.path.join(dataset_dir, 'body/bbox')
    bbox_file = os.path.join(bbox_dir, "bbox.pkl")
    if not os.path.exists(bbox_file):
        print(f"bbox path does not exist: {bbox_file}")
        print("NOTE: run mode_bbox")
        sys.exit(1)
    with open(bbox_file, "rb") as f:
        bboxes = pickle.load(f)
    
    # Set up image path loader
    dset = {curr_cam : sorted(glob.glob(os.path.join(dataset_dir, 'frames', curr_cam, '*'))) for curr_cam in exo_cam_names}
    # check number of images are consistent across different camera groups
    num_img = len(dset[next(iter(dset))])
    assert all(len(dset[k]) == num_img for k in dset), "Not all cameras have same number of images"

    # Iterate through every frame
    poses2d = {}
    for time_stamp in tqdm(range(num_img), total=num_img):
        poses2d[time_stamp] = {}
        # Iterate through every cameras
        for exo_camera_name in exo_cam_names:
            image_path = dset[exo_camera_name][time_stamp]
            image = cv2.imread(image_path)
            
            # Directory to store body kpts visualization for current camera
            vis_pose2d_cam_dir = os.path.join(vis_pose2d_dir, exo_camera_name)
            if not os.path.exists(vis_pose2d_cam_dir):
                os.makedirs(vis_pose2d_cam_dir)
            
            # Load in body bbox 
            bbox_xyxy = bboxes[time_stamp][exo_camera_name]  # x1, y1, x2, y2
            if bbox_xyxy is not None:
                # add confidence score to the bbox
                bbox_xyxy = np.append(bbox_xyxy, 1.0)
                
                # Inference to get body 2d kpts
                pose_results = pose_model.get_poses2d(
                    bboxes=[{"bbox": bbox_xyxy}],
                    image_name=image_path,
                )
                assert len(pose_results) == 1
                # Save results and visualization
                save_path = os.path.join(vis_pose2d_cam_dir, f"{time_stamp:05d}.jpg")
                pose_model.draw_poses2d(pose_results, image, save_path)
                pose_result = pose_results[0]
                pose2d = pose_result["keypoints"]
            else:
                pose2d = None
                save_path = os.path.join(vis_pose2d_cam_dir, f"{time_stamp:05d}.jpg")
                cv2.imwrite(save_path, image)

            poses2d[time_stamp][exo_camera_name] = pose2d

    # save poses2d to pose2d_dir/pose2d.pkl
    with open(os.path.join(pose2d_dir, "pose2d.pkl"), "wb") as f:
        pickle.dump(poses2d, f)



def mode_exo_hand_pose2d(args):
    """
    Hand pose2d estimation for exo cameras based on wholebody hand kpts as heuristic bbox input
    """
    # Directory setup and load parameters
    dataset_root = args.dataset_root
    dataset_dir = os.path.join(dataset_root, f'{args.case}', 'dataset')
    calib_result_dir = os.path.join(dataset_root, f'{args.case}', 'dataset/calib_results')
    exo_cam_names = args.exo_cam_names

    #####################################
    kpts_vis_threshold = 0.3
    #####################################

    # Hand pose estimation model
    ###### Onehand-10k ######
    # hand_pose_config = 'ego4d/internal/human_pose/external/mmlab/mmpose/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/onehand10k/res50_onehand10k_256x256.py'
    # hand_pose_ckpt = 'https://download.openmmlab.com/mmpose/top_down/resnet/res50_onehand10k_256x256-e67998f6_20200813.pth'
    ### COCOWholebody hand ###
    hand_pose_config = 'ego4d/internal/human_pose/external/mmlab/mmpose/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/coco_wholebody_hand/hrnetv2_w18_coco_wholebody_hand_256x256_dark.py'
    hand_pose_ckpt = 'https://download.openmmlab.com/mmpose/hand/dark/hrnetv2_w18_coco_wholebody_hand_256x256_dark-a9228c9c_20210908.pth'
    hand_pose_model = PoseModel(
        hand_pose_config, 
        hand_pose_ckpt,
        rgb_keypoint_thres=kpts_vis_threshold, 
        rgb_keypoint_vis_thres=kpts_vis_threshold,
        refine_bbox=False,
        thickness=3)
    
    # Directory to store bbox result and visualization
    bbox_dir = os.path.join(dataset_dir, f'hand/bbox')
    vis_bbox_dir = os.path.join(dataset_dir, f'hand/vis_bbox')
    os.makedirs(bbox_dir, exist_ok=True)
    os.makedirs(vis_bbox_dir, exist_ok=True)
    # Directory to store pose2d result and visualization
    pose2d_dir = os.path.join(dataset_dir, f'hand/pose2d')
    if not os.path.exists(pose2d_dir):
        os.makedirs(pose2d_dir)
    vis_pose2d_dir = os.path.join(dataset_dir, f'hand/vis_pose2d/visThresh={kpts_vis_threshold}')
    if not os.path.exists(vis_pose2d_dir):
        os.makedirs(vis_pose2d_dir)

    # Load human body keypoints result from mode_pose2d
    body_pose2d_path = os.path.join(dataset_dir, 'body/pose2d', "pose2d.pkl")
    assert os.path.exists(body_pose2d_path), f"{body_pose2d_path} does not exist"
    with open(body_pose2d_path, "rb") as f:
        body_poses2d = pickle.load(f)

    # Set up image path loader
    dset = {curr_cam : sorted(glob.glob(os.path.join(dataset_dir, 'frames', curr_cam, '*'))) for curr_cam in exo_cam_names}
    # check number of images are consistent across different camera groups
    num_img = len(dset[next(iter(dset))])
    assert all(len(dset[k]) == num_img for k in dset), "Not all cameras have same number of images"

    # Iterate through every frame
    poses2d = {}
    bboxes = {}
    for time_stamp in tqdm(range(num_img), total=num_img):
        poses2d[time_stamp] = {}
        bboxes[time_stamp] = {}
        # Iterate through every cameras
        for exo_camera_name in exo_cam_names:
            image_path = dset[exo_camera_name][time_stamp]
            image = cv2.imread(image_path)
            
            # Directory to store hand kpts visualization for exo cameras
            vis_pose2d_cam_dir = os.path.join(vis_pose2d_dir, exo_camera_name)
            if not os.path.exists(vis_pose2d_cam_dir):
                os.makedirs(vis_pose2d_cam_dir)
            # bbox visualization save dir
            vis_bbox_cam_dir = os.path.join(vis_bbox_dir, exo_camera_name)
            if not os.path.exists(vis_bbox_cam_dir):
                os.makedirs(vis_bbox_cam_dir)
            
            # Extract left and right hand hpts from wholebody kpts estimation
            body_pose_kpts = body_poses2d[time_stamp][exo_camera_name]
            # Right hand kpts
            right_hand_kpts_index = list(range(112,132))
            right_hand_kpts = body_pose_kpts[right_hand_kpts_index,:]
            # Left hand kpts
            left_hand_kpts_index = list(range(91,111))
            left_hand_kpts = body_pose_kpts[left_hand_kpts_index,:]

            ############## Hand bbox ##############
            img_H, img_W = image.shape[:2]
            right_hand_bbox = get_bbox_fromKpts(right_hand_kpts, img_W, img_H, padding=25)
            left_hand_bbox = get_bbox_fromKpts(left_hand_kpts, img_W, img_H, padding=25)
            # Append result NOTE: Right hand first
            bboxes[time_stamp][exo_camera_name] = [right_hand_bbox, left_hand_bbox]
            # Save hand bbox visualization
            vis_bbox_img = image.copy()
            vis_bbox_img = draw_bbox_xyxy(vis_bbox_img, right_hand_bbox, color=(255,0,0)) # Blue for right hand 
            vis_bbox_img = draw_bbox_xyxy(vis_bbox_img, left_hand_bbox, color=(0,0,255))  # Red for left hand
            cv2.imwrite(os.path.join(vis_bbox_cam_dir, f"{time_stamp:05d}.jpg"), vis_bbox_img)

            ############## Hand pose 2d ##############
            # Append confience score to bbox 
            bbox_xyxy_right = np.append(right_hand_bbox, 1)
            bbox_xyxy_left = np.append(left_hand_bbox, 1)
            two_hand_bboxes=[{"bbox": bbox_xyxy_right},
                             {"bbox": bbox_xyxy_left}]
            # Hand pose estimation
            pose_results = hand_pose_model.get_poses2d(
                                bboxes=two_hand_bboxes,
                                image_name=image_path,
                            )
            
            # ###### Heuristic check of pose2d: If two hand's result are too close then drop the one with lower confidence ######
            # right_hand_pos2d_kpts, left_hand_pos2d_kpts = pose_results[0]['keypoints'], pose_results[1]['keypoints']
            # pairwise_conf_dis = np.linalg.norm(left_hand_pos2d_kpts[:,:2] - right_hand_pos2d_kpts[:,:2],axis=1) * \
            #                     right_hand_pos2d_kpts[:,2] * \
            #                     left_hand_pos2d_kpts[:,2]
            # # Drop lower kpts result if pairwise_conf_dis is too low
            # if np.mean(pairwise_conf_dis) < 10:
            #     right_conf_mean = np.mean(right_hand_pos2d_kpts[:,2])
            #     left_conf_mean = np.mean(left_hand_pos2d_kpts[:,2])
            #     if right_conf_mean < left_conf_mean:
            #         right_hand_pos2d_kpts[:,:] = 0
            #     else:
            #         left_hand_pos2d_kpts[:,:] = 0
            # pose_results[0]['keypoints'] = right_hand_pos2d_kpts
            # pose_results[1]['keypoints'] = left_hand_pos2d_kpts
            # ####################################################################################################################

            # Save result and visualization
            save_path = os.path.join(vis_pose2d_cam_dir, f"{time_stamp:05d}.jpg")
            vis_pose2d_img = image.copy()
            hand_pose_model.draw_poses2d([pose_results[0]], vis_pose2d_img, save_path)
            vis_pose2d_img = cv2.imread(save_path)
            hand_pose_model.draw_poses2d([pose_results[1]], vis_pose2d_img, save_path)
            # Save 2d hand pose estimation result ~ (2,13,3)
            curr_pose2d_kpts = np.array([res['keypoints'] for res in pose_results])
            poses2d[time_stamp][exo_camera_name] = curr_pose2d_kpts

    # save poses2d key points result
    with open(os.path.join(pose2d_dir, "exo_pose2d.pkl"), "wb") as f:
        pickle.dump(poses2d, f)
    
    # save the bboxes as a pickle file
    with open(os.path.join(bbox_dir, "exo_bbox.pkl"), "wb") as f:
        pickle.dump(bboxes, f)



def mode_ego_hand_pose2d(args):
    """
    Hand bbox detection + Hand pose2d estimation for ego camera
    NOTE: 
        1. Hand bbox detection relies on pretrained hand detector
        2. Simple heuristic is used to 
            i.  Extract two smallest hand bbox if multiple hand bboxes detected
            ii. Assign left/right hand bbox based on x-coordinate
    """
    # Directory setup and load parameters
    dataset_root = args.dataset_root
    dataset_dir = os.path.join(dataset_root, f'{args.case}', 'dataset')
    calib_result_dir = os.path.join(dataset_root, f'{args.case}', 'dataset/calib_results')
    ego_cam_name = args.ego_cam_name

    ################# Modified as needed #####################
    kpts_vis_threshold = 0.3
    ##########################################################

    # Load hand bounding box detector
    config_file = 'ego4d/internal/human_pose/external/mmlab/mmpose/demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_1class.py'
    checkpoint_file = 'https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth'
    detector = init_detector(config_file, checkpoint_file, device='cuda:0')
    # Load hand pose estimation model
    ###### Onehand-10k ######
    # hand_pose_config = 'ego4d/internal/human_pose/external/mmlab/mmpose/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/onehand10k/res50_onehand10k_256x256.py'
    # hand_pose_ckpt = 'https://download.openmmlab.com/mmpose/top_down/resnet/res50_onehand10k_256x256-e67998f6_20200813.pth'
    ###### COCO wholebody hand ######
    hand_pose_config = 'ego4d/internal/human_pose/external/mmlab/mmpose/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/coco_wholebody_hand/hrnetv2_w18_coco_wholebody_hand_256x256_dark.py'
    hand_pose_ckpt = 'https://download.openmmlab.com/mmpose/hand/dark/hrnetv2_w18_coco_wholebody_hand_256x256_dark-a9228c9c_20210908.pth'
    hand_pose_model = PoseModel(hand_pose_config, 
                                hand_pose_ckpt, 
                                rgb_keypoint_thres=kpts_vis_threshold, 
                                rgb_keypoint_vis_thres=kpts_vis_threshold,
                                refine_bbox=False,
                                thickness=3)

    # All ego-view images directory
    dset = sorted(glob.glob(os.path.join(dataset_dir, 'frames', ego_cam_name, '*')))
    num_img = len(dset)

    # Directory to store bbox result and visualization for ego camera
    bbox_dir = os.path.join(dataset_dir, f'hand/bbox')
    vis_bbox_dir = os.path.join(dataset_dir, f'hand/vis_bbox', ego_cam_name)
    os.makedirs(bbox_dir, exist_ok=True)
    os.makedirs(vis_bbox_dir, exist_ok=True)
    # Directory to store pose2d result and visualization
    pose2d_dir = os.path.join(dataset_dir, f'hand/pose2d')
    if not os.path.exists(pose2d_dir):
        os.makedirs(pose2d_dir)
    vis_pose2d_dir = os.path.join(dataset_dir, f'hand/vis_pose2d/visThresh={kpts_vis_threshold}', ego_cam_name)
    if not os.path.exists(vis_pose2d_dir):
        os.makedirs(vis_pose2d_dir)

    # Iterate through all ego camera images
    bboxes = {}
    poses2d = {}

    prev_det = None # Please make sure first frame has good detected hand bbox
    manual_idx = []
    for idx in tqdm(range(num_img)):
        # Image path
        image_path = dset[idx]

        ############### Ego camera hand bbox detection ###############
        det_results = inference_detector(detector, image_path)

        # ##### Heuristics: Select two smallest bbox and assign right/left hand bbox based on x-coor ##########
        # refined_two_bboxes = get_two_small_bboxes(det_results[0])
        # right_left_bboxes = left_right_bboxes_div(refined_two_bboxes) # (2,4)
        # #####################################################################################################
        
        ########### Heuristic: Select two bboxes with smaller x-coor (For 0630_Cooking_4 ego view) ###########
        if len(det_results[0]) < 2: 
            # Method 1: If less than two hands are detected, then drop this and manually correct it later
            right_left_bboxes = np.zeros((2,4))
            manual_idx.append(idx)
            # Method 2: If less than two hands are detected, then use pervious frame's bbox result
            det_results = prev_det
            bbox_x_center = (det_results[0][:,0] + det_results[0][:,2]) / 2
            sort_order = np.argsort(bbox_x_center)
            two_smallest_bbox = det_results[0][sort_order[:2], :4]
            right_left_bboxes = left_right_bboxes_div(two_smallest_bbox) # (2,4)
        else:
            bbox_x_center = (det_results[0][:,0] + det_results[0][:,2]) / 2
            sort_order = np.argsort(bbox_x_center)
            two_smallest_bbox = det_results[0][sort_order[:2], :4]
            right_left_bboxes = left_right_bboxes_div(two_smallest_bbox) # (2,4)
        #######################################################################################################
        prev_det = det_results.copy()

        # Visualization
        vis_bbox_img = cv2.imread(image_path)
        color_map = [(255,0,0),(0,0,255)] 
        for ii, curr_hand_bbox in enumerate(right_left_bboxes):
            vis_bbox_img = draw_bbox_xyxy(vis_bbox_img, curr_hand_bbox, color=color_map[ii])
        cv2.imwrite(os.path.join(vis_bbox_dir, f'{idx:06d}.jpg'), vis_bbox_img)
        # Save bbox result
        bboxes[idx] = [bbox for bbox in right_left_bboxes] # Follow same convention as mode_hand_pose2d to save hand bbox

        ################### Ego camera hand pose2d ###################
        # Format hand bbox
        two_hand_bboxes = [{'bbox':np.append(curr_hand_bbox,1)} for curr_hand_bbox in right_left_bboxes]
        # Hand pose estimation
        pose_results = hand_pose_model.get_poses2d(
                            bboxes=two_hand_bboxes,
                            image_name=image_path,
                        )
        # Save result
        curr_pose2d_kpts = np.array([res['keypoints'] for res in pose_results])
        poses2d[idx] = curr_pose2d_kpts
        # Visualization
        save_path = os.path.join(vis_pose2d_dir, f'{idx:06d}.jpg')
        vis_twoHand = cv2.imread(image_path)
        hand_pose_model.draw_poses2d([pose_results[0]], vis_twoHand, save_path)
        vis_twoHand = cv2.imread(save_path)
        hand_pose_model.draw_poses2d([pose_results[1]], vis_twoHand, save_path)

    # Print image index that needs manual correction
    print('Less than two hand bboxes detected image index:\n',manual_idx)

    # save poses2d key points result
    with open(os.path.join(pose2d_dir, "ego_pose2d.pkl"), "wb") as f:
        pickle.dump(poses2d, f)
    # save the bboxes as a pickle file
    with open(os.path.join(bbox_dir, "ego_bbox.pkl"), "wb") as f:
        pickle.dump(bboxes, f)



def mode_exo_hand_pose3d(args):
    # Directory setup and load parameters
    dataset_root = args.dataset_root
    dataset_dir = os.path.join(dataset_root, f'{args.case}', 'dataset')
    calib_result_dir = os.path.join(dataset_root, f'{args.case}', 'dataset/calib_results')
    exo_cam_names = args.exo_cam_names

    ################# Modified as needed #####################
    tri_threshold = 0.3
    ##########################################################

    # Hand pose2d estimator (Only for visualization in here)
    hand_pose_config = 'ego4d/internal/human_pose/external/mmlab/mmpose/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/coco_wholebody_hand/hrnetv2_w18_coco_wholebody_hand_256x256_dark.py'
    hand_pose_ckpt = 'https://download.openmmlab.com/mmpose/hand/dark/hrnetv2_w18_coco_wholebody_hand_256x256_dark-a9228c9c_20210908.pth'
    hand_pose_model = PoseModel(
        hand_pose_config, 
        hand_pose_ckpt, 
        rgb_keypoint_thres=tri_threshold, 
        rgb_keypoint_vis_thres=tri_threshold,
        thickness=3)

    # Set up image path loader
    dset = {curr_cam : sorted(glob.glob(os.path.join(dataset_dir, 'frames', curr_cam, '*'))) for curr_cam in exo_cam_names}
    # check number of images are consistent across different camera groups
    num_img = len(dset[next(iter(dset))])
    assert all(len(dset[k]) == num_img for k in dset), "Not all cameras have same number of images"
    img_shape = cv2.imread(dset[exo_cam_names[0]][0]).shape[:2]

    # Create exo cameras object
    exo_cameras = {}
    for curr_cam_name in exo_cam_names:
        extri_dir = os.path.join(calib_result_dir, f'{curr_cam_name}_extri.json')
        intri_dir = os.path.join(calib_result_dir, f'{curr_cam_name}_intri.json')
        extrinsics, intrinsics = extri_intri_from_json(extri_dir, intri_dir)
        # Camera data template
        camera_data_keys = ['name', 'center', 'T_device_world', 'T_world_device', 'T_device_camera', 'T_camera_device', 'camera_type', 'device_row']
        curr_camera_data = {k:None for k in camera_data_keys}
        curr_camera_data['name'] = curr_cam_name
        curr_camera_data['extrinsics'] = extrinsics
        curr_camera_data['camera_type'] = 'colmap'
        # Create camera based on parameter
        exo_cameras[curr_cam_name] = create_customized_camera(curr_camera_data, intrinsics, img_shape)

    # Directory to store pose3d result and visualization
    pose3d_dir = os.path.join(dataset_dir, f'hand/pose3d')
    if not os.path.exists(pose3d_dir):
        os.makedirs(pose3d_dir)
    vis_pose3d_dir = os.path.join(dataset_dir, f'hand/vis_pose3d','exo_camera',f'triThresh={tri_threshold}')
    if not os.path.exists(vis_pose3d_dir):
        os.makedirs(vis_pose3d_dir)
    # Load hand pose2d keypoints from exo cameras
    cam_pose2d_file = os.path.join(dataset_dir, f'hand/pose2d', "exo_pose2d.pkl")
    assert os.path.exists(cam_pose2d_file), f"{cam_pose2d_file} does not exist"
    with open(cam_pose2d_file, "rb") as f:
        cam_poses2d = pickle.load(f)

    # Triangulation starts
    poses3d = {}
    for time_stamp in tqdm(range(num_img), total=num_img):
        # Collect hand pose2d kpts for this timestamp
        # multi_view_pose2d = {
        #     exo_camera_name: cam_poses2d[time_stamp][exo_camera_name].reshape(-1,3)
        #     for exo_camera_name in exo_cam_names
        # }
        ########### Heuristic Check: Hardcode hand wrist kpt conf to be 1 ################################
        multi_view_pose2d = {}
        for exo_camera_name in exo_cam_names:
            curr_exo_hand_pose2d_kpts = cam_poses2d[time_stamp][exo_camera_name].reshape(-1,3)
            # curr_exo_hand_pose2d_kpts[[0,21],2] = 1
            if np.mean(curr_exo_hand_pose2d_kpts[:,-1]) > 0.3:
                curr_exo_hand_pose2d_kpts[[0,21],2] = 1
            multi_view_pose2d[exo_camera_name] = curr_exo_hand_pose2d_kpts
        ##################################################################################################

        ###### Heuristic Check: If two hands are too close, then drop the one with lower confidence ######
        ###### TODO: Instead of dropping one with lower confidence, input both hand's kpts during triangulation and rely on RANSAC to choose the best
        for exo_camera_name in exo_cam_names:
            right_hand_pos2d_kpts, left_hand_pos2d_kpts = multi_view_pose2d[exo_camera_name][:21,:], multi_view_pose2d[exo_camera_name][21:,:]
            pairwise_conf_dis = np.linalg.norm(left_hand_pos2d_kpts[:,:2] - right_hand_pos2d_kpts[:,:2],axis=1) * \
                                right_hand_pos2d_kpts[:,2] * \
                                left_hand_pos2d_kpts[:,2]
            # Drop lower kpts result if pairwise_conf_dis is too low
            if np.mean(pairwise_conf_dis) < 5:
                right_conf_mean = np.mean(right_hand_pos2d_kpts[:,2])
                left_conf_mean = np.mean(left_hand_pos2d_kpts[:,2])
                if right_conf_mean < left_conf_mean:
                    right_hand_pos2d_kpts[:,:] = 0
                else:
                    left_hand_pos2d_kpts[:,:] = 0
            multi_view_pose2d[exo_camera_name][:21] = right_hand_pos2d_kpts
            multi_view_pose2d[exo_camera_name][21:] = left_hand_pos2d_kpts
        ###################################################################################################

        # Triangulation
        triangulator = Triangulator(
            time_stamp, 
            exo_cam_names, 
            exo_cameras, 
            multi_view_pose2d, 
            keypoint_thres=tri_threshold, 
            num_keypoints=42
        )
        pose3d = triangulator.run(debug=False)  ## N x 4 (x, y, z, confidence)
        poses3d[time_stamp] = pose3d

        # visualize pose3d triangulation result
        for camera_name in exo_cam_names:
            image_path = dset[camera_name][time_stamp]
            image = cv2.imread(image_path)
            curr_camera = exo_cameras[camera_name]

            vis_pose3d_cam_dir = os.path.join(vis_pose3d_dir, camera_name)
            if not os.path.exists(vis_pose3d_cam_dir):
                os.makedirs(vis_pose3d_cam_dir)

            projected_pose3d = batch_xworld_to_yimage(pose3d[:, :3], curr_camera)
            projected_pose3d = np.concatenate(
                [projected_pose3d, pose3d[:, 3].reshape(-1, 1)], axis=1
            )  ## N x 3 (17 for body,; 42 for hand)

            save_path = os.path.join(vis_pose3d_cam_dir, f"{time_stamp:05d}.jpg")
            hand_pose_model.draw_projected_poses3d([projected_pose3d[:21], projected_pose3d[21:]], image, save_path)
    
    # Save pose3d result
    with open(os.path.join(pose3d_dir, "exo_pose3d.pkl"), "wb") as f:
        pickle.dump(poses3d, f)



def mode_egoexo_hand_pose3d(args):
    # Directory setup and load parameters
    dataset_root = args.dataset_root
    dataset_dir = os.path.join(dataset_root, f'{args.case}', 'dataset')
    calib_result_dir = os.path.join(dataset_root, f'{args.case}', 'dataset/calib_results')
    # Camera name list
    exo_cam_names = args.exo_cam_names
    ego_cam_name = args.ego_cam_name
    ego_exo_cam_names = args.exo_cam_names + [args.ego_cam_name]

    ################# Modified as needed #####################
    tri_threshold = 0.3
    ##########################################################

    # Hand pose2d estimator (Only for visualization in here)
    hand_pose_config = 'ego4d/internal/human_pose/external/mmlab/mmpose/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/coco_wholebody_hand/hrnetv2_w18_coco_wholebody_hand_256x256_dark.py'
    hand_pose_ckpt = 'https://download.openmmlab.com/mmpose/hand/dark/hrnetv2_w18_coco_wholebody_hand_256x256_dark-a9228c9c_20210908.pth'
    hand_pose_model = PoseModel(
        hand_pose_config, 
        hand_pose_ckpt, 
        rgb_keypoint_thres=tri_threshold, 
        rgb_keypoint_vis_thres=tri_threshold)

    # Set up image path loader for both exo and ego cameras 
    dset = {curr_cam : sorted(glob.glob(os.path.join(dataset_dir, 'frames', curr_cam, '*'))) for curr_cam in ego_exo_cam_names}
    # check number of images are consistent across different camera groups
    num_img = len(dset[next(iter(dset))])
    assert all(len(dset[k]) == num_img for k in dset), "Not all cameras have same number of images"
    exo_img_shape = cv2.imread(dset[exo_cam_names[0]][0]).shape[:2]
    ego_img_shape = cv2.imread(dset[ego_cam_name][0]).shape[:2]

    # Create exo cameras object
    ego_exo_cameras = {}
    for curr_cam_name in exo_cam_names:
        extri_dir = os.path.join(calib_result_dir, f'{curr_cam_name}_extri.json')
        intri_dir = os.path.join(calib_result_dir, f'{curr_cam_name}_intri.json')
        extrinsics, intrinsics = extri_intri_from_json(extri_dir, intri_dir)
        # Camera data template
        camera_data_keys = ['name', 'center', 'T_device_world', 'T_world_device', 'T_device_camera', 'T_camera_device', 'camera_type', 'device_row']
        curr_camera_data = {k:None for k in camera_data_keys}
        curr_camera_data['name'] = curr_cam_name
        curr_camera_data['extrinsics'] = extrinsics
        curr_camera_data['camera_type'] = 'colmap'
        # Create camera based on parameter
        ego_exo_cameras[curr_cam_name] = create_customized_camera(curr_camera_data, intrinsics, exo_img_shape)

    # Directory to store pose3d result and visualization
    pose3d_dir = os.path.join(dataset_dir, f'hand/pose3d')
    if not os.path.exists(pose3d_dir):
        os.makedirs(pose3d_dir)
    vis_pose3d_dir = os.path.join(dataset_dir, f'hand/vis_pose3d','ego_exo_camera', f'triThresh={tri_threshold}')
    if not os.path.exists(vis_pose3d_dir):
        os.makedirs(vis_pose3d_dir)
    # Load hand pose2d keypoints from exo camera
    exo_cam_pose2d_file = os.path.join(dataset_dir, f'hand/pose2d', "exo_pose2d.pkl")
    assert os.path.exists(exo_cam_pose2d_file), f"{exo_cam_pose2d_file} does not exist"
    with open(exo_cam_pose2d_file, "rb") as f:
        exo_cam_poses2d = pickle.load(f)
    # Load hand pose2d kpypoints from ego camera
    ego_cam_pose2d_file = os.path.join(dataset_dir, f'hand/pose2d', "ego_pose2d.pkl")
    assert os.path.exists(ego_cam_pose2d_file), f"{ego_cam_pose2d_file} does not exist"
    with open(ego_cam_pose2d_file, "rb") as f:
        ego_cam_poses2d = pickle.load(f)

    # Triangulation starts
    poses3d = {}
    for time_stamp in tqdm(range(num_img), total=num_img):
        # Collect hand pose2d kpts at this timestamp
        ########### Heuristic Check: Hardcode hand wrist kpt conf to be 1 #############
        multi_view_pose2d = {}
        # Add exo camera keypoints
        for exo_camera_name in exo_cam_names:
            curr_exo_hand_pose2d_kpts = exo_cam_poses2d[time_stamp][exo_camera_name].reshape(-1,3)
            if np.mean(curr_exo_hand_pose2d_kpts[:,-1]) > 0.3:
                curr_exo_hand_pose2d_kpts[[0,21],2] = 1
            multi_view_pose2d[exo_camera_name] = curr_exo_hand_pose2d_kpts
        # Add ego camera keypoints
        ego_hand_pose2d_kpts = ego_cam_poses2d[time_stamp].reshape(-1,3)
        if np.mean(ego_hand_pose2d_kpts[:,-1]) > 0.3:
            ego_hand_pose2d_kpts[[0,21],2] = 1
        multi_view_pose2d[ego_cam_name] = ego_hand_pose2d_kpts
        ###############################################################################
        
        # Add ego camera at this timestamp
        extri_dir = os.path.join(calib_result_dir, f'{ego_cam_name}_extri.json')
        intri_dir = os.path.join(calib_result_dir, f'{ego_cam_name}_intri.json')
        extrinsics, intrinsics = extri_intri_from_json(extri_dir, intri_dir, time_stamp, ego=True)
        # Camera data template
        camera_data_keys = ['name', 'center', 'T_device_world', 'T_world_device', 'T_device_camera', 'T_camera_device', 'camera_type', 'device_row']
        ego_camera_data = {k:None for k in camera_data_keys}
        ego_camera_data['name'] = ego_cam_name
        ego_camera_data['extrinsics'] = extrinsics
        ego_camera_data['camera_type'] = 'colmap'
        # Create camera based on parameter
        ego_exo_cameras[ego_cam_name] = create_customized_camera(ego_camera_data, intrinsics, ego_img_shape)

        ###### Heuristic Check: If two hands are too close, then drop the one with lower confidence ######
        ###### TODO: Instead of dropping one with lower confidence, input both hand's kpts during triangulation and rely on RANSAC to choose the best
        for exo_camera_name in exo_cam_names:
            right_hand_pos2d_kpts, left_hand_pos2d_kpts = multi_view_pose2d[exo_camera_name][:21,:], multi_view_pose2d[exo_camera_name][21:,:]
            pairwise_conf_dis = np.linalg.norm(left_hand_pos2d_kpts[:,:2] - right_hand_pos2d_kpts[:,:2],axis=1) * \
                                right_hand_pos2d_kpts[:,2] * \
                                left_hand_pos2d_kpts[:,2]
            # Drop lower kpts result if pairwise_conf_dis is too low
            if np.mean(pairwise_conf_dis) < 5:
                right_conf_mean = np.mean(right_hand_pos2d_kpts[:,2])
                left_conf_mean = np.mean(left_hand_pos2d_kpts[:,2])
                if right_conf_mean < left_conf_mean:
                    right_hand_pos2d_kpts[:,:] = 0
                else:
                    left_hand_pos2d_kpts[:,:] = 0
            multi_view_pose2d[exo_camera_name][:21] = right_hand_pos2d_kpts
            multi_view_pose2d[exo_camera_name][21:] = left_hand_pos2d_kpts
        ###################################################################################################

        # Triangulation
        triangulator = Triangulator(
            time_stamp, 
            ego_exo_cam_names, 
            ego_exo_cameras,
            multi_view_pose2d, 
            keypoint_thres=tri_threshold, 
            num_keypoints=42
        )
        pose3d = triangulator.run(debug=False)  ## N x 4 (x, y, z, confidence)
        poses3d[time_stamp] = pose3d

        # visualize pose3d triangulation result
        for camera_name in ego_exo_cam_names:
            image_path = dset[camera_name][time_stamp]
            image = cv2.imread(image_path)
            curr_camera = ego_exo_cameras[camera_name]

            vis_pose3d_cam_dir = os.path.join(vis_pose3d_dir, camera_name)
            if not os.path.exists(vis_pose3d_cam_dir):
                os.makedirs(vis_pose3d_cam_dir)

            projected_pose3d = batch_xworld_to_yimage(pose3d[:, :3], curr_camera)
            projected_pose3d = np.concatenate(
                [projected_pose3d, pose3d[:, 3].reshape(-1, 1)], axis=1
            )  ## N x 3 (17 for body,; 42 for hand)

            save_path = os.path.join(vis_pose3d_cam_dir, f"{time_stamp:05d}.jpg")
            hand_pose_model.draw_projected_poses3d([projected_pose3d[:21], projected_pose3d[21:]], image, save_path)
    
    # Save pose3d result
    with open(os.path.join(pose3d_dir, "egoexo_pose3d.pkl"), "wb") as f:
        pickle.dump(poses3d, f)





def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default='/mnt/shared/ego4dData/cache',
                        help="root path for all dataset")
    parser.add_argument('--case', type=str, default=None,
                        help="path for one specific case")
    parser.add_argument('--mode', type=str, default=None,
                        help="mode to run the code e.g. body_bbox, body_pose2d etc.")
    parser.add_argument('--exo_cam_names', default=['cam01','cam02','cam03','cam04'], nargs='+')
    parser.add_argument('--ego_cam_name', type=str, default='cam05',
                        help="Name of the ego view camera e.g. cam05")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    # Run based on different mode
    if args.mode == 'body_bbox':
        mode_body_bbox(args)
    elif args.mode == 'body_pose2d':
        mode_body_pose2d(args)
    elif args.mode == 'hand_pose2d_exo':
        mode_exo_hand_pose2d(args)
    elif args.mode == 'hand_pose2d_ego':
        mode_ego_hand_pose2d(args)
    elif args.mode == 'hand_pose3d_exo':
        mode_exo_hand_pose3d(args)
    elif args.mode == 'hand_pose3d_ego_exo':
        mode_egoexo_hand_pose3d(args)
    else:
        raise AssertionError(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    main()