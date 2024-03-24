import argparse
import json
import os
import pickle
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional
import getpass

import cv2

import hydra
import numpy as np
import pandas as pd
from projectaria_tools.core import data_provider
from ego4d.internal.human_pose.camera import (
    batch_xworld_to_yimage,
    batch_xworld_to_yimage_check_camera_z,
    create_camera,
    create_camera_simple,
    create_camera_data,
    get_aria_camera_models,
)
from ego4d.internal.human_pose.config import Config
from ego4d.internal.human_pose.triangulator import Triangulator

from ego4d.internal.human_pose.camera_utils import (
    body_keypoints_list,    
    hand_keypoints_list,    
    process_exocam_data,
    get_aria_extrinsics,
    get_aria_intrinsics,    
)

from ego4d.internal.human_pose.utils import (
    aria_extracted_to_original,
    aria_original_to_extracted,    
)

from iopath.common.file_io import PathManager
from iopath.common.s3 import S3PathHandler
from projectaria_tools.core import data_provider
from tqdm.auto import tqdm

pathmgr = PathManager()
pathmgr.register_handler(S3PathHandler(profile="default"))

cvpr_data_dir="/large_experiments/egoexo/cvpr"
project_root_dir = "/large_experiments/egoexo/egopose/suyogjain/project_retriangulation_production_v2/"

annotation_base_dir = os.path.join(project_root_dir, "ego_pose_latest")
annotation_output_base_dir = os.path.join(project_root_dir, "ego_pose_post_triangulation")

@dataclass
class Context:
    data_dir: str    
    repo_root_dir: str
    cache_dir: str
    cache_rel_dir: str
    metadata_json: str
    dataset_dir: str
    dataset_json_path: str
    dataset_rel_dir: str
    frame_dir: str
    ego_cam_names: List[str]
    exo_cam_names: List[str]
    bbox_dir: str
    vis_bbox_dir: str
    pose2d_dir: str
    vis_pose2d_dir: str
    pose3d_dir: str
    vis_pose3d_dir: str
    detector_config: str
    detector_checkpoint: str
    pose_config: str
    pose_checkpoint: str
    dummy_pose_config: str
    dummy_pose_checkpoint: str
    hand_pose_config: str
    hand_pose_ckpt: str
    human_height: float = 1.5
    human_radius: float = 0.3
    min_bbox_score: float = 0.7
    pose3d_start_frame: int = 0
    pose3d_end_frame: int = -1
    refine_pose3d_dir: Optional[str] = None
    vis_refine_pose3d_dir: Optional[str] = None
    take: Optional[Dict[str, Any]] = None
    all_cams: Optional[List[str]] = None
    frame_rel_dir: str = None
    storage_level: int = 30

def get_context(config: Config) -> Context:    
    take_json_path = os.path.join(config.data_dir, "takes.json")
    takes = json.load(open(take_json_path))
    #take = [t for t in takes if t["root_dir"] == config.inputs.take_name]
    take = [t for t in takes if t["take_name"] == config.inputs.take_name]
    if len(take) != 1:
        print(f"Take: {config.inputs.take_name} does not exist")
        sys.exit(1)
    take = take[0]    

    if not config.data_dir.startswith("/"):
        config.data_dir = os.path.join(config.repo_root_dir, config.data_dir)
        print(f"Using data dir: {config.data_dir}")

    data_dir = config.data_dir
    cache_rel_dir = os.path.join(
        "cache",
        take["take_name"],
    )
    cache_dir = os.path.join(
        config.cache_root_dir,
        cache_rel_dir,
    )
    # Initialize exo cameras from calibration file since sometimes some exo camera is missing
    traj_dir = os.path.join(
        data_dir, take["capture"]["root_dir"], "trajectory"
    )
    exo_traj_path = os.path.join(traj_dir, "gopro_calibs.csv")
    exo_traj_df = pd.read_csv(exo_traj_path)
    exo_cam_names = list(exo_traj_df["cam_uid"])

    ego_cam_names = [x["cam_id"] for x in take["capture"]["cameras"] if x["is_ego"]]
    assert len(ego_cam_names) > 0, "No ego cameras found!"
    if len(ego_cam_names) > 1:
        print(
            f"[Warning] {len(ego_cam_names)} ego cameras: {ego_cam_names} filtering ..."
        )
        ego_cam_names = [
            cam for cam in ego_cam_names if cam in take["frame_aligned_videos"].keys()
        ]
        assert len(ego_cam_names) > 0, "No frame-aligned ego cameras found!"
        if len(ego_cam_names) > 1:
            print(
                f"[Warning] Still {len(ego_cam_names)} cameras: {ego_cam_names} filtering ..."
            )
            ego_cam_names_filtered = [
                cam for cam in ego_cam_names if "aria" in cam.lower()
            ]
            if len(ego_cam_names_filtered) == 1:
                ego_cam_names = ego_cam_names_filtered
        assert (
            len(ego_cam_names) == 1
        ), f"Found too many ({len(ego_cam_names)}) ego cameras: {ego_cam_names}"

    all_cams = ego_cam_names + exo_cam_names
    dataset_dir = cache_dir    
    frame_rel_dir = os.path.join(cache_rel_dir, "frames")

    return Context(
        data_dir=data_dir,        
        repo_root_dir=config.repo_root_dir,
        cache_dir=cache_dir,
        cache_rel_dir=cache_rel_dir,
        metadata_json=None,  # pyre-ignore
        dataset_dir=dataset_dir,
        dataset_json_path=os.path.join(dataset_dir, "data.json"),
        dataset_rel_dir=os.path.join(
            cache_rel_dir,
            config.mode_preprocess.dataset_name,
        ),
        frame_dir=os.path.join(dataset_dir, "frames"),
        frame_rel_dir=frame_rel_dir,
        ego_cam_names=ego_cam_names,
        exo_cam_names=exo_cam_names,
        all_cams=all_cams,
        bbox_dir=os.path.join(dataset_dir, "bbox"),
        vis_bbox_dir=os.path.join(dataset_dir, "vis_bbox"),
        pose2d_dir=os.path.join(dataset_dir, "pose2d"),
        vis_pose2d_dir=os.path.join(dataset_dir, "vis_pose2d"),
        pose3d_dir=os.path.join(dataset_dir, "pose3d"),
        vis_pose3d_dir=os.path.join(dataset_dir, "vis_pose3d"),
        detector_config=config.mode_bbox.detector_config,
        detector_checkpoint=config.mode_bbox.detector_checkpoint,
        pose_config=config.mode_pose2d.pose_config,
        pose_checkpoint=config.mode_pose2d.pose_checkpoint,
        dummy_pose_config=config.mode_pose2d.dummy_pose_config,
        dummy_pose_checkpoint=config.mode_pose2d.dummy_pose_checkpoint,
        human_height=config.mode_bbox.human_height,
        human_radius=config.mode_bbox.human_radius,
        min_bbox_score=config.mode_bbox.min_bbox_score,
        pose3d_start_frame=config.mode_pose3d.start_frame,
        pose3d_end_frame=config.mode_pose3d.end_frame,
        refine_pose3d_dir=os.path.join(dataset_dir, "refine_pose3d"),
        vis_refine_pose3d_dir=os.path.join(dataset_dir, "vis_refine_pose3d"),
        take=take,
        hand_pose_config=config.mode_pose2d.hand_pose_config,
        hand_pose_ckpt=config.mode_pose2d.hand_pose_ckpt,
        storage_level=config.outputs.storage_level,
    )

def mode_preprocess(config: Config):
    ctx = get_context(config)
    assert config.mode_preprocess.download_video_files, "must download files"
    #os.makedirs(ctx.frame_dir, exist_ok=True)
    
    mps_capture_dir = os.path.join(
        ctx.data_dir, "captures", ctx.take["capture"]["root_dir"]
    )
    print(mps_capture_dir)
    traj_dir = os.path.join(mps_capture_dir, "trajectory")
    aria_traj_path = os.path.join(traj_dir, "closed_loop_trajectory.csv")
    exo_traj_path = os.path.join(traj_dir, "gopro_calibs.csv")
    aria_traj_df = pd.read_csv(aria_traj_path)
    exo_traj_df = pd.read_csv(exo_traj_path)
    all_timesync_df = pd.read_csv(os.path.join(mps_capture_dir, "timesync.csv"))

    # TODO: confirm that this is correct?
    i1, i2 = ctx.take["timesync_start_idx"], ctx.take["timesync_end_idx"] - 1
    synced_df = all_timesync_df.iloc[i1:i2]

    # Note: start_frame and end_frame are relative to i1 (i.e., synced_df)
    start_frame = config.inputs.from_frame_number
    end_frame = config.inputs.to_frame_number

    frame_window_size = 1
    if end_frame is None or end_frame > len(synced_df) - frame_window_size:
        end_frame = len(synced_df) - frame_window_size

    print(i1, i2, start_frame, end_frame, ctx.ego_cam_names, ctx.exo_cam_names)

    stream_name_to_id = {
        "et": "211-1",
        "rgb": "214-1",
        "slam-left": "1201-1",
        "slam-right": "1201-2",
    }

    output = dict()
    for cam_id in ctx.exo_cam_names:
        key = (cam_id, "0")
        key_str = "_".join(key)                
        cam_data = exo_traj_df[exo_traj_df.cam_uid == cam_id].iloc[0].to_dict()

        output[key_str] = {            
            "t": None,
            "camera_data": create_camera_data(
                device_row=cam_data,
                name=cam_id,
                camera_model=None,
                device_row_key="cam",
            ),
            "_raw_camera": cam_data,
        }
  
    capture_dir = os.path.join(
        cvpr_data_dir, "captures", ctx.take["capture"]["root_dir"]
    )
    aria_dir = os.path.join(capture_dir, "videos")
    aria_path = os.path.join(aria_dir, f"{ctx.ego_cam_names[0]}.vrs")
    assert os.path.exists(aria_path), f"Cannot find {aria_path}"
    aria_camera_models = get_aria_camera_models(aria_path)

    assert config.inputs.exo_timesync_name_to_calib_name is None

    
    for idx in range(start_frame, end_frame):
            row = {}
            row_df = synced_df.iloc[idx]
            skip_frame = False
            for stream_name in config.inputs.aria_streams:
                if stream_name!='rgb':
                    continue
            
                key = (ctx.ego_cam_names[0], stream_name)
                key_str = "_".join(key)                
                if key_str not in output:
                    output[key_str] = dict()                    
                
                stream_id = stream_name_to_id[stream_name]
                try:
                    frame_num = int(
                        row_df[f"{ctx.ego_cam_names[0]}_{stream_id}_frame_number"]
                    )
                except ValueError as e:
                    # cannot convert float NaN to integer
                    print(f"[Warning] Skip idx {idx} due to exception:\n{str(e)}")
                    skip_frame = True
                    break

                frame_t = (
                    row_df[f"{ctx.ego_cam_names[0]}_{stream_id}_capture_timestamp_ns"]
                    / 1e9
                )
                aria_t = (
                    row_df[f"{ctx.ego_cam_names[0]}_{stream_id}_capture_timestamp_ns"]
                    / 1e3
                )
                frame_t = f"{frame_t:.3f}"
                aria_pose = aria_traj_df.iloc[
                    (aria_traj_df.tracking_timestamp_us - aria_t)
                    .abs()
                    .argsort()
                    .iloc[0]
                ].to_dict()
                output[key_str][idx] = {                    
                    "frame_number": idx,
                    "capture_frame_number": frame_num,
                    "t": aria_t,
                    "camera_data": create_camera_data(
                        device_row=aria_pose,
                        name=stream_id,
                        camera_model=aria_camera_models[stream_id],
                        device_row_key="device",
                    ),
                    "_raw_camera": aria_pose,
                }

    dataset_json = {
        "cache_dir": ctx.cache_rel_dir,        
        "dataset_dir": ctx.dataset_rel_dir,
        "frames": output,
    }
    print(f"Saving data info to {ctx.dataset_json_path}")
    json.dump(dataset_json, open(ctx.dataset_json_path, "w"))


def extract_camera_data(config: Config):
    ctx = get_context(config)
    assert config.mode_preprocess.download_video_files, "must download files"
    
    mps_capture_dir = os.path.join(
        ctx.data_dir, ctx.take["capture"]["root_dir"]
    )    
    print(mps_capture_dir)
    traj_dir = os.path.join(mps_capture_dir, "trajectory")
    aria_traj_path = os.path.join(traj_dir, "closed_loop_trajectory.csv")
    exo_traj_path = os.path.join(traj_dir, "gopro_calibs.csv")
    aria_traj_df = pd.read_csv(aria_traj_path)
    exo_traj_df = pd.read_csv(exo_traj_path)
    all_timesync_df = pd.read_csv(os.path.join(mps_capture_dir, "timesync.csv"))

    # TODO: confirm that this is correct?
    i1, i2 = ctx.take["timesync_start_idx"], ctx.take["timesync_end_idx"] - 1
    synced_df = all_timesync_df.iloc[i1:i2]

    # Note: start_frame and end_frame are relative to i1 (i.e., synced_df)
    start_frame = config.inputs.from_frame_number
    end_frame = config.inputs.to_frame_number

    frame_window_size = 1
    if end_frame is None or end_frame > len(synced_df) - frame_window_size:
        end_frame = len(synced_df) - frame_window_size

    print(i1, i2, start_frame, end_frame, ctx.ego_cam_names, ctx.exo_cam_names)

    stream_name_to_id = {
        "et": "211-1",
        "rgb": "214-1",
        "slam-left": "1201-1",
        "slam-right": "1201-2",
    }

    capture_dir = os.path.join(
        cvpr_data_dir,  ctx.take["capture"]["root_dir"]
    )
    print(capture_dir)
    aria_dir = os.path.join(capture_dir, "videos")
    aria_path = os.path.join(aria_dir, f"{ctx.ego_cam_names[0]}.vrs")
    assert os.path.exists(aria_path), f"Cannot find {aria_path}"    
    aria_camera_models = get_aria_camera_models(aria_path)
    
    assert config.inputs.exo_timesync_name_to_calib_name is None

    camera_info = dict()
    exo_cameras = dict()
    camera_info["metadata"] = dict()
    camera_info["metadata"]['take_name'] = config.inputs.take_name
    camera_info["metadata"]['take_uid']  = ctx.take["take_uid"]

    for cam_id in ctx.exo_cam_names:        
        key_str = cam_id                   
        cam_data = exo_traj_df[exo_traj_df.cam_uid == cam_id].iloc[0].to_dict()
        
        exo_cameras[key_str] = {                            
            "camera_data": create_camera_data(
                device_row=cam_data,
                name=cam_id,
                camera_model=None,
                device_row_key="cam",
            ),
            "_raw_camera": cam_data,
        }
        
        # Process exo camera params        
        processed_cam_data = process_exocam_data(exo_cameras[key_str])
        camera_info[key_str] = processed_cam_data
            
    for idx in range(start_frame, end_frame):        
        row_df = synced_df.iloc[idx]
        skip_frame = False
        for stream_name in config.inputs.aria_streams:            
            if stream_name!='rgb':
                continue
            
            key_str = ctx.ego_cam_names[0]
            if key_str not in camera_info:
                camera_info[key_str] = dict()
                camera_info[key_str]['camera_intrinsics'] = get_aria_intrinsics()
                camera_info[key_str]['camera_extrinsics'] = dict()

            stream_id = stream_name_to_id[stream_name]
            try:
                frame_num = int(
                    row_df[f"{ctx.ego_cam_names[0]}_{stream_id}_frame_number"]
                )
            except ValueError as e:
                # cannot convert float NaN to integer
                print(f"[Warning] Skip idx {idx} due to exception:\n{str(e)}")
                skip_frame = True
                break

            frame_t = (
                row_df[f"{ctx.ego_cam_names[0]}_{stream_id}_capture_timestamp_ns"]
                / 1e9
            )
            aria_t = (
                row_df[f"{ctx.ego_cam_names[0]}_{stream_id}_capture_timestamp_ns"]
                / 1e3
            )
            frame_t = f"{frame_t:.3f}"
            aria_pose = aria_traj_df.iloc[
                (aria_traj_df.tracking_timestamp_us - aria_t)
                .abs()
                .argsort()
                .iloc[0]
            ].to_dict()
            
            aria_data = {                
                "frame_number": idx,
                "capture_frame_number": frame_num,
                "t": aria_t,
                "camera_data": create_camera_data(
                    device_row=aria_pose,
                    name=stream_id,
                    camera_model=aria_camera_models[stream_id],
                    device_row_key="device",
                ),
                "_raw_camera": aria_pose,
            }
            aria_extrinsics = get_aria_extrinsics(aria_data)
            camera_info[key_str]['camera_extrinsics'][idx] = aria_extrinsics            
            if (idx+1) % 100 == 0:
                print(f"[Info] Saved {idx+1}/{end_frame} frames for {stream_name}")
            

        if skip_frame:
            continue       

    #print(json.dumps(camera_info, indent=2))
    camera_info_output_dir = os.path.join(ctx.dataset_dir, 'camera_pose')
    os.makedirs(camera_info_output_dir, exist_ok=True)
    camera_info_json_path=os.path.join(camera_info_output_dir, ctx.take["take_uid"]+".json")    
    print(f"Saving camera info to {camera_info_json_path}")
    json.dump(camera_info, open(camera_info_json_path, "w"))

def convert_to_array(pose2d, keypoints_list, exo_cam_list):
    pose2d_transformed = dict()
    for cam_name in pose2d:
        if cam_name not in exo_cam_list:
            continue
        pose_list = pose2d[cam_name]    
        pose_array = list()
        for kp in keypoints_list:
            kp_name = kp["label"].lower()
            if kp_name in pose_list:
                ann = pose_list[kp_name]                
                confidence = 1
                if ann['placement']=='auto':
                    confidence = 0
                pose_array.append([ann['x'], ann['y'], confidence])
            else:
                pose_array.append([0, 0, 0])

        pose_array = np.array(pose_array)
        pose2d_transformed[cam_name] = pose_array
    return pose2d_transformed

def get_inlier_by_camera(ctx, inlier_views):
    inlier_kp = dict()
    for exo_camera_name in ctx.exo_cam_names:
        inlier_kp[exo_camera_name] = list()
            
    for kp_id in inlier_views:
        cam_names = inlier_views[kp_id]
        for cam_name in cam_names:
            inlier_kp[cam_name].append(kp_id)
    
    return inlier_kp


def calculate_reprojection_errors(ctx, exo_cameras, pose3d_new, multi_view_pose2d, inlier_kp):
    error_vector = list()
    projected_2d = dict()
    print("2D Reprojection Info")
    for exo_camera_name in ctx.exo_cam_names:            
        exo_camera = exo_cameras[exo_camera_name]            
        projected_pose3d = batch_xworld_to_yimage(pose3d_new[:, :3], exo_camera)
        if exo_camera_name not in multi_view_pose2d:
            continue
        original_pose2d = multi_view_pose2d[exo_camera_name][:, :2]
        reprojection_error = np.linalg.norm(
            (original_pose2d - projected_pose3d), ord=1, axis=1
        )
        projected_2d[exo_camera_name] = dict()
        for kp_id in inlier_kp[exo_camera_name]:
            kp_name = body_keypoints_list[kp_id]['label'].lower()
            print(exo_camera_name, kp_id, kp_name, reprojection_error[kp_id], original_pose2d[kp_id,:], projected_pose3d[kp_id, :])
            error_vector.append(reprojection_error[kp_id])
            projected_2d[exo_camera_name][kp_name] = dict()
            projected_2d[exo_camera_name][kp_name]['x'] = projected_pose3d[kp_id, 0]
            projected_2d[exo_camera_name][kp_name]['y'] = projected_pose3d[kp_id, 1]
    
    error_vector = np.array(error_vector)
    print()
    return np.mean(error_vector), projected_2d 


def transform(pose3d_new, pose3d, body_keypoints_list):
    keypoints_3d = dict()
    print("3D pose updates")
    for kp_id in range(len(body_keypoints_list)):
        kp_name = body_keypoints_list[kp_id]['label'].lower()
        if kp_name in pose3d:            
            keypoints_3d[kp_name] = {
                "x": pose3d_new[kp_id, 0],
                "y": pose3d_new[kp_id, 1],
                "z": pose3d_new[kp_id, 2],
                "num_views_for_3d": pose3d[kp_name]['num_views_for_3d']                
            }
            print(kp_name, pose3d[kp_name], pose3d_new[kp_id, :])
        else:
            print(kp_name, {}, pose3d_new[kp_id, :])

    print()
    return keypoints_3d

def transform_basic(pose3d_new, body_keypoints_list):
    keypoints_3d = dict()
    print("3D pose updates")
    for kp_id in range(len(body_keypoints_list)):
        kp_name = body_keypoints_list[kp_id]['label'].lower()
        is_valid = np.sum(pose3d_new[kp_id,:])!=0
        if is_valid:            
            keypoints_3d[kp_name] = {
                "x": pose3d_new[kp_id, 0],
                "y": pose3d_new[kp_id, 1],
                "z": pose3d_new[kp_id, 2],
                "num_views_for_3d": 2                
            }
        print(kp_name, is_valid, pose3d_new[kp_id, :])        

    print()
    return keypoints_3d


def convert_to_array_hand(pose2d, keypoints_list, all_cam_list):
    pose2d_transformed = dict()    
    for cam_name in pose2d:
        target_cam_name = cam_name
        if cam_name.find('aria')!=-1:
            target_cam_name = f"{cam_name}_rgb"
        
        if target_cam_name not in all_cam_list:
            continue
        
        pose_list = pose2d[cam_name]    
        pose_array = list()
        for kp in keypoints_list:
            kp_name = kp["label"].lower()
            if kp_name in pose_list:
                ann = pose_list[kp_name]                
                confidence = 1
                if ann['placement']=='auto':
                    confidence = 0
                else:
                    print('Manual annotation (x, y):', cam_name, kp_name, ann['x'], ann['y'])
                pose_array.append([ann['x'], ann['y'], confidence])
            else:
                pose_array.append([0, 0, 0])

        pose_array = np.array(pose_array)
        if cam_name.find('aria')!=-1: 
            #pose_array = aria_extracted_to_original(pose_array)
            # Transform annotations to ARIA rotated frame
            print("Skip Transformed original keypoints to ARIA rotated points")            
            #pose_array = aria_original_to_extracted(pose_array)                                               
        print()        
        
        pose2d_transformed[target_cam_name] = pose_array
    return pose2d_transformed

def get_inlier_by_camera_hand(all_cam_list, inlier_views):
    inlier_kp = dict()
    for egoexo_camera_name in all_cam_list:
        inlier_kp[egoexo_camera_name] = list()
            
    for kp_id in inlier_views:
        cam_names = inlier_views[kp_id]
        for cam_name in cam_names:
            inlier_kp[cam_name].append(kp_id)
    
    return inlier_kp

           


def calculate_reprojection_errors_hand(all_cam_list, egoexo_cameras, pose3d_new, multi_view_pose2d, inlier_kp):
    error_vector = list()
    projected_2d = dict()
    print("2D Reprojection Info")
    for egoexo_camera_name in all_cam_list:
        egoexo_camera = egoexo_cameras[egoexo_camera_name]   
                 
        projected_pose3d = batch_xworld_to_yimage(pose3d_new[:, :3], egoexo_camera)
        if egoexo_camera_name not in multi_view_pose2d:
            continue
        
        original_pose2d = multi_view_pose2d[egoexo_camera_name][:, :2]
        reprojection_error = np.linalg.norm(
            (original_pose2d - projected_pose3d), ord=1, axis=1
        )
        projected_2d[egoexo_camera_name] = dict()
        for kp_id in inlier_kp[egoexo_camera_name]:
            kp_name = hand_keypoints_list[kp_id]['label'].lower()
            print(egoexo_camera_name, kp_id, kp_name, reprojection_error[kp_id], original_pose2d[kp_id,:], projected_pose3d[kp_id, :])
            error_vector.append(reprojection_error[kp_id])
            projected_2d[egoexo_camera_name][kp_name] = dict()
            projected_2d[egoexo_camera_name][kp_name]['x'] = projected_pose3d[kp_id, 0]
            projected_2d[egoexo_camera_name][kp_name]['y'] = projected_pose3d[kp_id, 1]
    
    error_vector = np.array(error_vector)
    print()
    return np.mean(error_vector), projected_2d 



def mode_body_pose3d(config: Config, annot_type='annotation'):
    """
    Body pose3d estimation with exo cameras, only uses first 17 body kpts for faster speed
    """    
    skel_type = "body"

    # Load dataset info
    ctx = get_context(config)
    with open(ctx.dataset_json_path, 'r') as f:
        dset = json.load(f)
        
    # Load exo cameras
    exo_cameras = {
        exo_camera_name: create_camera(
            dset["frames"][f"{exo_camera_name}_0"]["camera_data"], None
        )
        for exo_camera_name in ctx.exo_cam_names
    }    
    
    annotation_dir = os.path.join(annotation_base_dir, skel_type, annot_type)    
    annotation_json_path=os.path.join(annotation_dir, ctx.take["take_uid"]+".json")    
    print(f"Loading annotation from {annotation_json_path}")
    with open(annotation_json_path, 'r') as f:
        annotation = json.load(f)
    
    projected_2d_annotation = dict()
    print(f"Num annotations found:", len(annotation))    
    count = 0
    for frame_number in annotation:        
        num_annotators = len(annotation[frame_number])
        if count>5:
            break
        count+=1
        print(frame_number, num_annotators)
        projected_2d_annotation[frame_number] = list()
        for annotation_index in range(num_annotators):
            frame_data = annotation[frame_number][annotation_index]
            pose2d = frame_data["annotation2D"]
            multi_view_pose2d = convert_to_array(pose2d, body_keypoints_list, ctx.exo_cam_names)
            print(multi_view_pose2d.keys())
            if len(multi_view_pose2d.keys())==0:
                print("No camera annotations found")                
                pose3d_new = {}                
            else:                                                      
                # triangulate
                triangulator = Triangulator(
                    frame_number, ctx.exo_cam_names, exo_cameras, multi_view_pose2d
                )            
                pose3d_new, inlier_views, reprojection_error_vector  = triangulator.run(debug=False)  ## 17 x 4 (x, y, z, confidence) 
                if "annotation3D" not in frame_data:
                    pose3d = {}
                else:           
                    pose3d = frame_data["annotation3D"]
                
                inlier_kp = get_inlier_by_camera(ctx, inlier_views)

                proj_error, projected_2d = calculate_reprojection_errors(ctx, exo_cameras, pose3d_new, multi_view_pose2d, inlier_kp)                
                
                projected_2d_annotation[frame_number].append(projected_2d)
                print(f"Average Projection Error: {config.inputs.take_name}-{frame_number}: {proj_error}\n")                 
                
                if len(pose3d)==0:
                    pose3d_new = transform_basic(pose3d_new, body_keypoints_list)
                else:
                    pose3d_new = transform(pose3d_new, pose3d, body_keypoints_list)
                                
            annotation[frame_number][annotation_index]["annotation3D"] = pose3d_new
                                            
        print('-'*80)

    
    annotation_output_dir = os.path.join(annotation_output_base_dir, skel_type, annot_type)    
    os.makedirs(annotation_output_dir, exist_ok=True)

    annotation_output_json_path=os.path.join(annotation_output_dir, ctx.take["take_uid"]+".json")    
    print(f"Saving retriangulated annotations to {annotation_output_json_path}")
    json.dump(annotation, open(annotation_output_json_path, "w"))            

    annotation_output_json_path=os.path.join(annotation_output_dir, ctx.take["take_uid"]+"_projected.json")    
    print(f"Saving retriangulated annotations to {annotation_output_json_path}")
    json.dump(projected_2d_annotation, open(annotation_output_json_path, "w"))            


   
def mode_hand_pose3d(config: Config, annot_type='annotation'):
    """
    Hand pose3d estimation with exo+ego cameras
    """    
    skel_type = "hand"
    

    # Load dataset info
    ctx = get_context(config)
    with open(ctx.dataset_json_path, 'r') as f:
        dset = json.load(f)
    
    ego_cam_names = [f"{cam}_rgb" for cam in ctx.ego_cam_names]    
    exo_cam_names = (ctx.exo_cam_names) 
    all_used_cam = exo_cam_names + ego_cam_names
    #all_used_cam = exo_cam_names 
    tri_threshold = 0.3
    
    # Load exo cameras
    aria_exo_cameras = {
        exo_camera_name: create_camera(
            dset["frames"][f"{exo_camera_name}_0"]["camera_data"], None
        )
        for exo_camera_name in ctx.exo_cam_names
    }    

    capture_dir = os.path.join(
        cvpr_data_dir, "captures", ctx.take["capture"]["root_dir"]
    )
    print(capture_dir)
    aria_dir = os.path.join(capture_dir, "videos")
    aria_path = os.path.join(aria_dir, f"{ctx.ego_cam_names[0]}.vrs")
    assert os.path.exists(aria_path), f"Cannot find {aria_path}"    
    print(aria_path)
    aria_camera_models = get_aria_camera_models(aria_path)
    stream_name_to_id = {
        f"{ctx.ego_cam_names[0]}_rgb": "214-1",
        f"{ctx.ego_cam_names[0]}_slam-left": "1201-1",
        f"{ctx.ego_cam_names[0]}_slam-right": "1201-2",
    }
    
    annotation_dir = os.path.join(annotation_base_dir, skel_type, annot_type)    
    annotation_json_path=os.path.join(annotation_dir, ctx.take["take_uid"]+".json")    
    print(f"Loading annotation from {annotation_json_path}")
    with open(annotation_json_path, 'r') as f:
        annotation = json.load(f)
    
    projected_2d_annotation = dict()
    print(f"Num annotations found:", len(annotation))   
    count = 0 
    for frame_number in annotation:        
        num_annotators = len(annotation[frame_number])        
        print(frame_number, num_annotators)
        projected_2d_annotation[frame_number] = list()
        
        for annotation_index in range(num_annotators):
            frame_data = annotation[frame_number][annotation_index]
            pose2d = frame_data["annotation2D"]
            print(f"Loading annotation for cams {all_used_cam}\n")          
            multi_view_pose2d = convert_to_array_hand(pose2d, hand_keypoints_list, all_used_cam)                   
            if len(multi_view_pose2d.keys())==0:
                print("No camera annotations found")                
                pose3d_new = {}                
            else:                                                      
                # triangulate
                # Add ego camera for the frame in the camera dict
                
                for ego_cam_name in ego_cam_names:
                    ego_cam_data = dset['frames'][ego_cam_name][frame_number]["camera_data"]                    
                    aria_exo_cameras[ego_cam_name] = create_camera(ego_cam_data,                         
                        aria_camera_models[stream_name_to_id[ego_cam_name]],
                    )

                print(f'Triangulating frame {frame_number} using cams {all_used_cam}')
                # triangulate                
                triangulator = Triangulator(
                    frame_number,
                    all_used_cam,
                    aria_exo_cameras,
                    multi_view_pose2d,
                    keypoint_thres=tri_threshold,
                    num_keypoints=42,
                    inlier_reproj_error_check=False,
                    #inlier_reproj_error_check=True,
                )
                pose3d_new, inlier_views, reprojection_error_vector = triangulator.run(debug=True, keypoints_list=hand_keypoints_list)  ## N x 4 (x, y, z, confidence)                                                
                if "annotation3D" not in frame_data:
                    pose3d = {}
                else:           
                    pose3d = frame_data["annotation3D"]
                                
                inlier_kp = get_inlier_by_camera_hand(all_used_cam, inlier_views)                                
                proj_error, projected_2d = calculate_reprojection_errors_hand(all_used_cam, aria_exo_cameras, pose3d_new, multi_view_pose2d, inlier_kp)                            
                
                projected_2d_annotation[frame_number].append(projected_2d)
                print(f"Average Projection Error: {config.inputs.take_name}-{frame_number}: {proj_error}\n")                 
                                
                if len(pose3d)==0:
                    pose3d_new = transform_basic(pose3d_new, hand_keypoints_list)
                else:
                    pose3d_new = transform(pose3d_new, pose3d, hand_keypoints_list)
                                
            annotation[frame_number][annotation_index]["annotation3D"] = pose3d_new
                                            
        print('-'*80)

        count+=1
        if count==3:
            break

    
    annotation_output_dir = os.path.join(annotation_output_base_dir, skel_type, annot_type)    
    os.makedirs(annotation_output_dir, exist_ok=True)

    annotation_output_json_path=os.path.join(annotation_output_dir, ctx.take["take_uid"]+".json")    
    print(f"Saving retriangulated annotations to {annotation_output_json_path}")
    json.dump(annotation, open(annotation_output_json_path, "w"))            

    annotation_output_json_path=os.path.join(annotation_output_dir, ctx.take["take_uid"]+"_projected.json")    
    print(f"Saving retriangulated annotations to {annotation_output_json_path}")
    json.dump(projected_2d_annotation, open(annotation_output_json_path, "w"))  
    
def mode_hand_pose3d_v2(config: Config, annot_type='annotation'):
    """
    Hand pose3d estimation with exo+ego cameras
    """    
    skel_type = "hand"
    

    # Load dataset info
    ctx = get_context(config)
    with open(ctx.dataset_json_path, 'r') as f:
        dset = json.load(f)
    
    ego_cam_names = [f"{cam}_rgb" for cam in ctx.ego_cam_names]    
    exo_cam_names = (ctx.exo_cam_names) 
    all_used_cam = exo_cam_names + ego_cam_names
    #all_used_cam = exo_cam_names 
    tri_threshold = 0.3
    
    # Load exo cameras
    aria_exo_cameras = {
        exo_camera_name: create_camera_simple(
            dset["frames"][f"{exo_camera_name}_0"]["camera_data"], None
        )
        for exo_camera_name in ctx.exo_cam_names
    }    

    '''
    capture_dir = os.path.join(
        cvpr_data_dir, "captures", ctx.take["capture"]["root_dir"]
    )
    print(capture_dir)
    aria_dir = os.path.join(capture_dir, "videos")
    aria_path = os.path.join(aria_dir, f"{ctx.ego_cam_names[0]}.vrs")
    assert os.path.exists(aria_path), f"Cannot find {aria_path}"    
    print(aria_path)
    aria_camera_models = get_aria_camera_models(aria_path)
    stream_name_to_id = {
        f"{ctx.ego_cam_names[0]}_rgb": "214-1",
        f"{ctx.ego_cam_names[0]}_slam-left": "1201-1",
        f"{ctx.ego_cam_names[0]}_slam-right": "1201-2",
    }
    '''
    
    annotation_dir = os.path.join(annotation_base_dir, skel_type, annot_type)    
    annotation_json_path=os.path.join(annotation_dir, ctx.take["take_uid"]+".json")    
    print(f"Loading annotation from {annotation_json_path}")
    with open(annotation_json_path, 'r') as f:
        annotation = json.load(f)
    
    projected_2d_annotation = dict()
    print(f"Num annotations found:", len(annotation))   
    count = 0 
    for frame_number in annotation:        
        num_annotators = len(annotation[frame_number])        
        print(frame_number, num_annotators)
        projected_2d_annotation[frame_number] = list()
        
        for annotation_index in range(num_annotators):
            frame_data = annotation[frame_number][annotation_index]
            pose2d = frame_data["annotation2D"]
            print(f"Loading annotation for cams {all_used_cam}\n")          
            multi_view_pose2d = convert_to_array_hand(pose2d, hand_keypoints_list, all_used_cam)                   
            if len(multi_view_pose2d.keys())==0:
                print("No camera annotations found")                
                pose3d_new = {}                
            else:                                                      
                # triangulate
                # Add ego camera for the frame in the camera dict
                
                for ego_cam_name in ego_cam_names:
                    print(f'Loading {frame_number} | {ego_cam_name} without ARIA camera model')
                    ego_cam_data = dset['frames'][ego_cam_name][frame_number]["camera_data"]
                    aria_exo_cameras[ego_cam_name] = create_camera_simple(ego_cam_data, None)                         
                   
                print(f'Triangulating frame {frame_number} using cams {all_used_cam}')
                # triangulate                
                triangulator = Triangulator(
                    frame_number,
                    all_used_cam,
                    aria_exo_cameras,
                    multi_view_pose2d,
                    keypoint_thres=tri_threshold,
                    num_keypoints=42,
                    inlier_reproj_error_check=False,
                    #inlier_reproj_error_check=True,
                )
                pose3d_new, inlier_views, reprojection_error_vector = triangulator.run(debug=True, keypoints_list=hand_keypoints_list)  ## N x 4 (x, y, z, confidence)                                                
                if "annotation3D" not in frame_data:
                    pose3d = {}
                else:           
                    pose3d = frame_data["annotation3D"]
                                
                inlier_kp = get_inlier_by_camera_hand(all_used_cam, inlier_views)                                
                proj_error, projected_2d = calculate_reprojection_errors_hand(all_used_cam, aria_exo_cameras, pose3d_new, multi_view_pose2d, inlier_kp)                            
                
                projected_2d_annotation[frame_number].append(projected_2d)
                print(f"Average Projection Error: {config.inputs.take_name}-{frame_number}: {proj_error}\n")                 
                                
                if len(pose3d)==0:
                    pose3d_new = transform_basic(pose3d_new, hand_keypoints_list)
                else:
                    pose3d_new = transform(pose3d_new, pose3d, hand_keypoints_list)
                                
            annotation[frame_number][annotation_index]["annotation3D"] = pose3d_new
                                            
        print('-'*80)

        count+=1
        if count==3:
            break

    
    annotation_output_dir = os.path.join(annotation_output_base_dir, skel_type, annot_type)    
    os.makedirs(annotation_output_dir, exist_ok=True)

    annotation_output_json_path=os.path.join(annotation_output_dir, ctx.take["take_uid"]+".json")    
    print(f"Saving retriangulated annotations to {annotation_output_json_path}")
    json.dump(annotation, open(annotation_output_json_path, "w"))            

    annotation_output_json_path=os.path.join(annotation_output_dir, ctx.take["take_uid"]+"_projected.json")    
    print(f"Saving retriangulated annotations to {annotation_output_json_path}")
    json.dump(projected_2d_annotation, open(annotation_output_json_path, "w"))  


def add_arguments(parser):
    parser.add_argument("--config-name", default="georgiatech_covid_02_2")
    parser.add_argument(
        "--config_path", default="configs", help="Path to the config folder"
    )
    parser.add_argument(
        "--take_name",
        default="georgiatech_covid_02_2",
        type=str,
        help="take names to run, concatenated by '+', "
        + "e.g., uniandes_dance_007_3+iiith_cooking_23+nus_covidtest_01",
    )
    parser.add_argument(
        "--steps",
        default="hand_pose3d_egoexo",
        type=str,
        help="steps to run concatenated by '+', e.g., preprocess+bbox+pose2d+pose3d",
    )


def config_single_job(args, job_id):
    args.job_id = job_id
    args.name = args.name_list[job_id]
    args.work_dir = args.work_dir_list[job_id]
    args.output_dir = args.work_dir

    args.take_name = args.take_name_list[job_id]


def create_job_list(args):
    args.take_name_list = args.take_name.split("+")

    args.job_list = []
    args.name_list = []

    for take_name in args.take_name_list:
        name = take_name
        args.name_list.append(name)
        args.job_list.append(name)

    args.job_num = len(args.job_list)


def parse_args():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    print(args)

    return args


def get_hydra_config(args):
    # https://stackoverflow.com/questions/60674012/how-to-get-a-hydra-config-without-using-hydra-main
    hydra.initialize(config_path=args.config_path)
    cfg = hydra.compose(
        config_name=args.config_name,
        # args.opts contains config overrides, e.g., ["inputs.from_frame_number=7000",]
        overrides=args.opts + [f"inputs.take_name={args.take_name}"],
    )
    print("Final config:", cfg)
    return cfg

@hydra.main(config_path="configs", config_name=None, version_base=None)
def new_run(config: Config):
    if config.mode == "preprocess":
        mode_preprocess(config)   
    else:
        raise AssertionError(f"unknown mode: {config.mode}")

def main(args):
    # Note: this function is called from launch_main.py
    config = get_hydra_config(args)

    steps = args.steps.split("+")
    print(f"steps: {steps}")

    for step in steps:
        print(f"[Info] Running step: {step}")
        start_time = time.time()
        if step == "extract_camera_info":
            extract_camera_data(config)
        elif step == "preprocess":
            mode_preprocess(config)
        elif step == "update_body_annotations":            
            mode_body_pose3d(config)
        elif step == "update_body_automatic":            
            mode_body_pose3d(config, 'automatic') 
        elif step == "update_hand_annotations":                        
            mode_hand_pose3d_v2(config)
        elif step == "update_hand_automatic":            
            mode_hand_pose3d_v2(config, 'automatic')              
        else:
            raise Exception(f"Unknown step: {step}")
        print(f"[Info] Time for {step}: {time.time() - start_time} s")


if __name__ == "__main__":
    new_run()
