import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional

import hydra
import numpy as np
import pandas as pd
from projectaria_tools.core import data_provider
from ego4d.internal.human_pose.camera import (    
    create_camera_data,
    get_aria_camera_models,
)
from ego4d.internal.human_pose.config import Config

from ego4d.internal.human_pose.camera_utils import (
    process_exocam_data,
    get_aria_extrinsics,
    get_aria_intrinsics,    
)


from iopath.common.file_io import PathManager
from iopath.common.s3 import S3PathHandler
from tqdm.auto import tqdm

pathmgr = PathManager()
pathmgr.register_handler(S3PathHandler(profile="default"))

cvpr_data_dir="/large_experiments/egoexo/cvpr"
project_root_dir = "/large_experiments/egoexo/egopose/suyogjain/project_retriangulation_production_v2/"


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


def run_pipeline(args, take_name):
    print(take_name)
    args.take_name = take_name
    config = get_hydra_config(args)        
    extract_camera_data(config)
    