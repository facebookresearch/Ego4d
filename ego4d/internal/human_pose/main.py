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
from ego4d.internal.colmap.preprocess import download_andor_generate_streams
from projectaria_tools.core import data_provider
from ego4d.internal.human_pose.camera import (
    batch_xworld_to_yimage,
    batch_xworld_to_yimage_check_camera_z,
    create_camera,
    create_camera_data,
    get_aria_camera_models,
)
from ego4d.internal.human_pose.config import Config
from ego4d.internal.human_pose.dataset import (
    get_synced_timesync_df,
    SyncedEgoExoCaptureDset,
)
#from ego4d.internal.human_pose.pose_estimator import PoseModel
from ego4d.internal.human_pose.pose_refiner import get_refined_pose3d
from ego4d.internal.human_pose.postprocess_pose3d import detect_outliers_and_interpolate
from ego4d.internal.human_pose.readers import read_frame_idx_set
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
    check_and_convert_bbox,
    draw_bbox_xyxy,
    draw_points_2d,
    get_bbox_from_kpts,
    get_exo_camera_plane,
    get_region_proposal,
    normalize_reprojection_error,
    wholebody_hand_selector,
)
from ego4d.research.readers import PyAvReader

from iopath.common.file_io import PathManager
from iopath.common.s3 import S3PathHandler
from projectaria_tools.core import data_provider
from tqdm.auto import tqdm

pathmgr = PathManager()
pathmgr.register_handler(S3PathHandler(profile="default"))

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
    take = [t for t in takes if t["root_dir"] == config.inputs.take_name]
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
        take["root_dir"],
    )
    cache_dir = os.path.join(
        config.cache_root_dir,
        cache_rel_dir,
    )
    # Initialize exo cameras from calibration file since sometimes some exo camera is missing
    traj_dir = os.path.join(
        data_dir, "captures", take["capture"]["root_dir"], "trajectory"
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
    # dataset_dir = os.path.join(cache_dir, config.mode_preprocess.dataset_name)
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
 
    cvpr_data_dir="/large_experiments/egoexo/cvpr"
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

    cvpr_data_dir="/large_experiments/egoexo/cvpr"
    capture_dir = os.path.join(
        cvpr_data_dir, "captures", ctx.take["capture"]["root_dir"]
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

def calculate_reprojection_errors_buggy(ctx, exo_cameras, pose3d_new, multi_view_pose2d, inlier_kp):
    error_vector = list()
    projected_2d = dict()
    print("2D Reprojection Info")
    for exo_camera_name in ctx.exo_cam_names:            
        exo_camera = exo_cameras[exo_camera_name]            
        projected_pose3d = batch_xworld_to_yimage(pose3d_new[:, :3], exo_camera)        
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
    print("3D pose updates | Advanced")
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
    print("3D pose updates | Basic")
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
                pose_array.append([ann['x'], ann['y'], confidence])
            else:
                pose_array.append([0, 0, 0])

        pose_array = np.array(pose_array)
        
        if cam_name.find('aria')!=-1: 
            #print(pose_array)
            pose_array = aria_extracted_to_original(pose_array)
            #print(pose_array)
        
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
        
        #if "aria" in egoexo_camera_name:
        #        projected_pose3d = aria_original_to_extracted(projected_pose3d)
        
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

    annotation_base_dir = "/large_experiments/egoexo/egopose/suyogjain/project_retriangulation/ego_pose_latest/"
    annotation_dir = os.path.join(annotation_base_dir, skel_type, annot_type)    
    annotation_json_path=os.path.join(annotation_dir, ctx.take["take_uid"]+".json")    
    print(f"Loading annotation from {annotation_json_path}")
    with open(annotation_json_path, 'r') as f:
        annotation = json.load(f)
    
    projected_2d_annotation = dict()
    print(f"Num annotations found:", len(annotation))    
    for frame_number in annotation:        
        num_annotators = len(annotation[frame_number])
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

    annotation_output_base_dir = "/large_experiments/egoexo/egopose/suyogjain/project_retriangulation/ego_pose_post_triangulation/"
    annotation_output_dir = os.path.join(annotation_output_base_dir, skel_type, annot_type)    
    os.makedirs(annotation_output_dir, exist_ok=True)

    annotation_output_json_path=os.path.join(annotation_output_dir, ctx.take["take_uid"]+".json")    
    print(f"Saving retriangulated annotations to {annotation_output_json_path}")
    json.dump(annotation, open(annotation_output_json_path, "w"))            

    annotation_output_json_path=os.path.join(annotation_output_dir, ctx.take["take_uid"]+"_projected.json")    
    print(f"Saving retriangulated annotations to {annotation_output_json_path}")
    json.dump(projected_2d_annotation, open(annotation_output_json_path, "w"))            

def mode_egoexo_hand_pose3d(config: Config):
    """
    Hand pose3d estimation with both ego and exo cameras
    """
    ctx = get_context(config)
    # TODO: Integrate those hardcoded values into args
    ########### Modify as needed #############
    exo_cam_names = (
        ctx.exo_cam_names
    )  # Select all default cameras: ctx.exo_cam_names or manual seelction: ['cam01','cam02']
    ego_cam_names = [f"{cam}_rgb" for cam in ctx.ego_cam_names]
    all_used_cam = exo_cam_names + ego_cam_names
    tri_threshold = 0.3
    visualization = True
    wholebody_hand_tri_threshold = 0.5
    use_wholebody_hand_selector = True
    ##########################################

    # Load dataset info
    dset = SyncedEgoExoCaptureDset(
        data_dir=config.cache_root_dir,
        dataset_json_path=ctx.dataset_json_path,
        read_frames=False,
    )
    
    # Create exo cameras
    aria_exo_cameras = {
        exo_camera_name: create_camera(
            dset[0][f"{exo_camera_name}_0"]["camera_data"], None
        )
        for exo_camera_name in ctx.exo_cam_names
    }

    # Load hand pose2d keypoints from both cam and aria
    cam_pose2d_file = os.path.join(ctx.dataset_dir, "hand/pose2d", "exo_pose2d.pkl")    
    with open(cam_pose2d_file, "rb") as f:
        exo_poses2d = pickle.load(f)
    aria_pose2d_file = os.path.join(ctx.dataset_dir, "hand/pose2d", "ego_pose2d.pkl")
    with open(aria_pose2d_file, "rb") as f:
        aria_poses2d = pickle.load(f)
    # Load wholebody-Hand kpts as selector candidate
    wholebody_hand_pose3d_file = os.path.join(
        ctx.dataset_dir,
        f"body/pose3d/wholebodyHand_pose3d_triThresh={wholebody_hand_tri_threshold}.pkl",
    )
    assert os.path.exists(
        wholebody_hand_pose3d_file
    ), f"{wholebody_hand_pose3d_file} does not exist"
    with open(wholebody_hand_pose3d_file, "rb") as f:
        wholebody_hand_poses3d = pickle.load(f)

    # Create aria calibration model
    capture_dir = os.path.join(
        ctx.data_dir, "captures", ctx.take["capture"]["root_dir"]
    )
    aria_path = os.path.join(capture_dir, f"videos/{ctx.ego_cam_names[0]}.vrs")
    assert os.path.exists(
        aria_path
    ), f"{aria_path} doesn't exit. Need aria video downloaded"
    aria_camera_models = get_aria_camera_models(aria_path)
    stream_name_to_id = {
        f"{ctx.ego_cam_names[0]}_rgb": "214-1",
        f"{ctx.ego_cam_names[0]}_slam-left": "1201-1",
        f"{ctx.ego_cam_names[0]}_slam-right": "1201-2",
    }

    # Iterate through all images and inference
    poses3d = {}
    reprojection_errors = {}
    for time_stamp in tqdm(range(len(dset)), total=len(dset)):
        info = dset[time_stamp]
        reprojection_errors[time_stamp] = {}

        # Load hand pose2d results for exo cameras
        multi_view_pose2d = {}
        original_pose2d_dict = {}
        # 1. Add exo camera keypoints
        for exo_camera_name in exo_cam_names:
            # For each hand, assign zero as hand keypoints
            # if it is None to perform triangulation
            right_hand_pose2d, left_hand_pose2d = exo_poses2d[time_stamp][
                exo_camera_name
            ].copy()
            right_hand_pose2d = (
                np.zeros((21, 3)) if right_hand_pose2d is None else right_hand_pose2d
            )
            left_hand_pose2d = (
                np.zeros((21, 3)) if left_hand_pose2d is None else left_hand_pose2d
            )
            # Concatenate right and left hand keypoints
            curr_exo_hand_pose2d_kpts = np.concatenate(
                (right_hand_pose2d, left_hand_pose2d), axis=0
            )  # (42,3)
            original_pose2d_dict[exo_camera_name] = curr_exo_hand_pose2d_kpts.copy()
            ### Heuristics: Hardcode hand wrist kpt conf to be 1 if average conf > 0.3 ###
            if np.mean(curr_exo_hand_pose2d_kpts[:21, -1]) > 0.3:
                curr_exo_hand_pose2d_kpts[0, -1] = 1
            if np.mean(curr_exo_hand_pose2d_kpts[21:, -1]) > 0.3:
                curr_exo_hand_pose2d_kpts[21, -1] = 1
            # Append kpts result
            multi_view_pose2d[exo_camera_name] = curr_exo_hand_pose2d_kpts
        # 2. Add ego camera keypoints (Rotate from extracted view
        # to original view since extrinsic/intrinsic is with original view)
        for ego_cam_name in ego_cam_names:
            # For each hand, assign zero as hand keypoints
            # if it is None to perform triangulation
            right_hand_pose2d, left_hand_pose2d = aria_poses2d[time_stamp][
                ego_cam_name
            ].copy()
            right_hand_pose2d = (
                np.zeros((21, 3)) if right_hand_pose2d is None else right_hand_pose2d
            )
            left_hand_pose2d = (
                np.zeros((21, 3)) if left_hand_pose2d is None else left_hand_pose2d
            )
            # Concatenate right and left hand keypoints
            ego_hand_pose2d_kpts = np.concatenate(
                (right_hand_pose2d, left_hand_pose2d), axis=0
            )  # (42,3)
            original_pose2d_dict[ego_cam_name] = ego_hand_pose2d_kpts.copy()
            # Rotate to aria frames
            ego_hand_pose2d_kpts = aria_extracted_to_original(ego_hand_pose2d_kpts)
            ### Heuristics: Hardcode hand wrist kpt conf to be 1 if average conf > 0.3 ###
            if np.mean(ego_hand_pose2d_kpts[:21, -1]) > 0.3:
                ego_hand_pose2d_kpts[0, -1] = 1
            if np.mean(ego_hand_pose2d_kpts[21:, -1]) > 0.3:
                ego_hand_pose2d_kpts[21, -1] = 1
            # Append kpts result
            multi_view_pose2d[ego_cam_name] = ego_hand_pose2d_kpts
        ###############################################################################

        # Add ego camera
        for ego_cam_name in ego_cam_names:
            aria_exo_cameras[ego_cam_name] = create_camera(
                info[ego_cam_name]["camera_data"],
                aria_camera_models[stream_name_to_id[ego_cam_name]],
            )

        ###### Heuristic Check: If two hands are too close, then drop the one with lower confidence ######
        for curr_cam_name in all_used_cam:
            right_hand_pos2d_kpts, left_hand_pos2d_kpts = (
                multi_view_pose2d[curr_cam_name][:21, :],
                multi_view_pose2d[curr_cam_name][21:, :],
            )
            pairwise_conf_dis = (
                np.linalg.norm(
                    left_hand_pos2d_kpts[:, :2] - right_hand_pos2d_kpts[:, :2], axis=1
                )
                * right_hand_pos2d_kpts[:, 2]
                * left_hand_pos2d_kpts[:, 2]
            )
            # Drop lower kpts result if pairwise_conf_dis is too low
            if np.mean(pairwise_conf_dis) < 10:
                right_conf_mean = np.mean(right_hand_pos2d_kpts[:, 2])
                left_conf_mean = np.mean(left_hand_pos2d_kpts[:, 2])
                if right_conf_mean < left_conf_mean:
                    right_hand_pos2d_kpts[:, :] = 0
                else:
                    left_hand_pos2d_kpts[:, :] = 0
            multi_view_pose2d[curr_cam_name][:21] = right_hand_pos2d_kpts
            multi_view_pose2d[curr_cam_name][21:] = left_hand_pos2d_kpts
        ###################################################################################################

        # triangulate
        triangulator = Triangulator(
            time_stamp,
            all_used_cam,
            aria_exo_cameras,
            multi_view_pose2d,
            keypoint_thres=tri_threshold,
            num_keypoints=42,
            inlier_reproj_error_check=True,
        )
        pose3d = triangulator.run(debug=False)  ## N x 4 (x, y, z, confidence)
        # Selector
        if use_wholebody_hand_selector:
            pose3d = wholebody_hand_selector(pose3d, wholebody_hand_poses3d[time_stamp])
        poses3d[time_stamp] = pose3d

        # visualize pose3d
        if visualization:
            # Visualization of images from all camera views
            for camera_name in (
                ctx.exo_cam_names + ego_cam_names
            ):  # Visualize all exo cameras + selected Aria camera
                
                curr_camera = aria_exo_cameras[camera_name]
                projected_pose3d = batch_xworld_to_yimage(pose3d[:, :3], curr_camera)
                projected_pose3d = np.concatenate(
                    [projected_pose3d, pose3d[:, 3].reshape(-1, 1)], axis=1
                )  ## N x 3 (17 for body,; 42 for hand)

                if "aria" in camera_name:
                    projected_pose3d = aria_original_to_extracted(projected_pose3d)

                
        # Compute reprojection error
        invalid_index = pose3d[:, 2] == 0
        for camera_name in ctx.exo_cam_names + ego_cam_names:
            # Extract projected pose3d results onto current camera plane
            curr_camera = aria_exo_cameras[camera_name]
            projected_pose3d = batch_xworld_to_yimage(pose3d[:, :3], curr_camera)
            # Rotate projected kpts if is aria camera
            if "aria" in camera_name:
                projected_pose3d = aria_original_to_extracted(projected_pose3d)
            # Compute L1-norm between projected 2D kpts and hand_pose2d
            original_pose2d = original_pose2d_dict[camera_name][:, :2]
            reprojection_error = np.linalg.norm(
                (original_pose2d - projected_pose3d), ord=1, axis=1
            )
            # Assign invalid index's reprojection error to be -1
            reprojection_error[invalid_index] = -1
            # Append result
            reprojection_errors[time_stamp][camera_name] = reprojection_error.reshape(
                -1, 1
            )

def mode_hand_pose3d(config: Config, annot_type='annotation'):
    """
    Body pose3d estimation with exo cameras, only uses first 17 body kpts for faster speed
    """    
    skel_type = "hand"
    

    # Load dataset info
    ctx = get_context(config)
    with open(ctx.dataset_json_path, 'r') as f:
        dset = json.load(f)
    
    ego_cam_names = [f"{cam}_rgb" for cam in ctx.ego_cam_names]    
    exo_cam_names = (ctx.exo_cam_names) 
    all_used_cam = exo_cam_names + ego_cam_names
    all_used_cam = exo_cam_names 
    tri_threshold = 0.3
    print(ego_cam_names, exo_cam_names)
    
    # Load exo cameras
    aria_exo_cameras = {
        exo_camera_name: create_camera(
            dset["frames"][f"{exo_camera_name}_0"]["camera_data"], None
        )
        for exo_camera_name in ctx.exo_cam_names
    }    

    cvpr_data_dir="/large_experiments/egoexo/cvpr"
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

    annotation_base_dir = "/large_experiments/egoexo/egopose/suyogjain/project_retriangulation/ego_pose_latest/"
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

                
                # triangulate
                print(frame_number, all_used_cam, aria_exo_cameras.keys(), multi_view_pose2d.keys())
                triangulator = Triangulator(
                    frame_number,
                    all_used_cam,
                    aria_exo_cameras,
                    multi_view_pose2d,
                    keypoint_thres=tri_threshold,
                    num_keypoints=42,
                    inlier_reproj_error_check=True,
                )
                pose3d_new, inlier_views, reprojection_error_vector = triangulator.run(debug=False)  ## N x 4 (x, y, z, confidence)                                
                if "annotation3D" not in frame_data:
                    pose3d = {}
                else:           
                    pose3d = frame_data["annotation3D"]
                                
                inlier_kp = get_inlier_by_camera_hand(all_used_cam, inlier_views)                
                
                #print(pose3d_new)
                #print(pose3d)                                
                print(inlier_kp)            
                
                proj_error, projected_2d = calculate_reprojection_errors_hand(all_used_cam, aria_exo_cameras, pose3d_new, multi_view_pose2d, inlier_kp)                            
                
                projected_2d_annotation[frame_number].append(projected_2d)
                print(f"Average Projection Error: {config.inputs.take_name}-{frame_number}: {proj_error}\n")                 
                
                continue
                if len(pose3d)==0:
                    pose3d_new = transform_basic(pose3d_new, body_keypoints_list)
                else:
                    pose3d_new = transform(pose3d_new, pose3d, body_keypoints_list)
                                
            #annotation[frame_number][annotation_index]["annotation3D"] = pose3d_new
                                            
        print('-'*80)

        count+=1
        if count==3:
            break

    #annotation_output_base_dir = "/large_experiments/egoexo/egopose/suyogjain/project_retriangulation/ego_pose_post_triangulation/"
    #annotation_output_dir = os.path.join(annotation_output_base_dir, skel_type, annot_type)    
    #os.makedirs(annotation_output_dir, exist_ok=True)

    #annotation_output_json_path=os.path.join(annotation_output_dir, ctx.take["take_uid"]+".json")    
    #print(f"Saving retriangulated annotations to {annotation_output_json_path}")
    #json.dump(annotation, open(annotation_output_json_path, "w"))            

    #annotation_output_json_path=os.path.join(annotation_output_dir, ctx.take["take_uid"]+"_projected.json")    
    #print(f"Saving retriangulated annotations to {annotation_output_json_path}")
    #json.dump(projected_2d_annotation, open(annotation_output_json_path, "w"))            


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
            mode_hand_pose3d(config)
        elif step == "update_hand_automatic":            
            mode_hand_pose3d(config, 'automatic')              
        else:
            raise Exception(f"Unknown step: {step}")
        print(f"[Info] Time for {step}: {time.time() - start_time} s")


if __name__ == "__main__":
    new_run()
