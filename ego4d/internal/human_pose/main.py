import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import cv2
import pickle

import hydra
import pandas as pd
from ego4d.internal.colmap.preprocess import download_andor_generate_streams
from ego4d.internal.human_pose.camera import (
    create_camera,
    create_camera_data,
    get_aria_camera_models,
    batch_xworld_to_yimage,
)

from ego4d.internal.human_pose.utils import (
    check_and_convert_bbox, draw_points_2d, get_region_proposal, get_exo_camera_plane,
    draw_bbox_xyxy
)

from ego4d.internal.human_pose.config import Config
from ego4d.internal.human_pose.dataset import (
    get_synced_timesync_df,
    SyncedEgoExoCaptureDset,
)
from ego4d.internal.human_pose.readers import read_frame_idx_set

from iopath.common.file_io import PathManager
from iopath.common.s3 import S3PathHandler
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor

pathmgr = PathManager()
pathmgr.register_handler(S3PathHandler(profile="default"))


@dataclass
class Context:
    root_dir: str
    cache_dir: str
    cache_rel_dir: str
    metadata_json: str
    dataset_dir: str
    dataset_json_path: str
    dataset_rel_dir: str
    frame_dir: str
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
    human_height: float = 1.5
    human_radius: float = 0.3
    min_bbox_score: float = 0.7
    pose3d_start_frame: int = 0
    pose3d_end_frame: int = -1
    refine_pose3d_dir: str = None
    vis_refine_pose3d_dir: str = None

def _create_json_from_capture_dir(capture_dir: Optional[str]) -> Dict[str, Any]:
    if capture_dir.endswith("/"):
        capture_dir = capture_dir[0:-1]

    if capture_dir.startswith("s3://"):
        bucket_name = capture_dir.split("s3://")[1].split("/")[0]
        prefix_path = "s3://{bucket_name}"
    else:
        prefix_path = capture_dir

    dirs = capture_dir.split("/")
    take_id = dirs[-1]
    video_source = dirs[-2]
    files_in_capture = pathmgr.ls(capture_dir + "/")
    video_files = pathmgr.ls(os.path.join(capture_dir, "videos/"))

    def _create_video(f):
        device_id = os.path.splitext(os.path.basename(f))[0]
        device_type = "aria" if "aria" in device_id else "gopro"
        is_ego = device_type == "aria"
        has_walkaround = "mobile" in device_id or "aria" in device_id
        s3_path = os.path.join(prefix_path, f)
        return {
            "device_id": device_id,
            "device_type": device_type,
            "is_ego": is_ego,
            "has_walkaround": has_walkaround,
            "s3_path": s3_path,
        }

    return {
        "take_id": take_id,
        "video_source": video_source,
        "ego_id": "aria01",
        "timesync_csv_path": os.path.join(capture_dir, "timesync.csv"),
        "preview_path": os.path.join(capture_dir, "preview.mp4"),
        "videos": [_create_video(f) for f in video_files],
    }


def get_context(config: Config) -> Context:
    if config.inputs.metadata_json_path is not None:
        metadata_json = (
            json.load(pathmgr.open(config.inputs.metadata_json_path))
            if config.inputs.metadata_json_path is not None
            else _create_json_from_capture_dir(config.inputs.input_capture_dir)
        )
    else:
        metadata_json = _create_json_from_capture_dir(config.inputs.capture_root_dir)

    cache_rel_dir = os.path.join(
        "cache",
        f"{metadata_json['video_source']}_{metadata_json['take_id']}",
    )
    cache_dir = os.path.join(
        config.root_dir,
        cache_rel_dir,
    )
    exo_cam_names = [
        x["device_id"]
        for x in metadata_json["videos"]
        if not x["is_ego"] and not x["has_walkaround"]
    ]
    dataset_dir = os.path.join(cache_dir, config.mode_preprocess.dataset_name)

    return Context(
        root_dir=config.root_dir,
        cache_dir=cache_dir,
        cache_rel_dir=cache_rel_dir,
        metadata_json=metadata_json,
        dataset_dir=dataset_dir,
        dataset_json_path=os.path.join(dataset_dir, "data.json"),
        dataset_rel_dir=os.path.join(
            cache_rel_dir, config.mode_preprocess.dataset_name
        ),
        frame_dir=os.path.join(dataset_dir, "frames"),
        exo_cam_names=exo_cam_names,
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
    )

##-------------------------------------------------------------------------------
##-------------------------------------------------------------------------------
def mode_pose3d(config: Config):
    ctx = get_context(config)

    dset = SyncedEgoExoCaptureDset(
        root_dir=config.root_dir,
        dataset_json_path=ctx.dataset_json_path,
        read_frames=False,
    )
    # by_dev_id = download_andor_generate_streams(
    #     metadata=ctx.metadata_json,
    #     download_video_files=config.mode_preprocess.download_video_files,
    #     force_download=config.mode_preprocess.force_download,
    #     output_dir=ctx.cache_dir,
    # )
    all_cam_ids = set(dset.all_cam_ids())

    # aria_path = by_dev_id["aria01"]["local_path"]
    aria_path = "/home/rawalk/Desktop/datasets/ego4d_data/cache/unc_T1/aria01/aria01.vrs"
    assert not aria_path.startswith("https:") or not aria_path.startswith("s3:")
    # aria_camera_models = get_aria_camera_models(aria_path)

    #---------------import triangulator-------------------
    from ego4d.internal.human_pose.triangulator import Triangulator
    from ego4d.internal.human_pose.pose_estimator import PoseModel
    pose_model = PoseModel(pose_config=ctx.dummy_pose_config, pose_checkpoint=ctx.dummy_pose_checkpoint) ## lightweight for visualization only!

    exo_cameras = {exo_camera_name: create_camera(dset[0][exo_camera_name]["camera_data"], None) for exo_camera_name in ctx.exo_cam_names}
    if not os.path.exists(ctx.pose3d_dir):
        os.makedirs(ctx.pose3d_dir)
    
    ## if ctx.vis_pose3d_dir does not exist make it
    if not os.path.exists(ctx.vis_pose3d_dir):
        os.makedirs(ctx.vis_pose3d_dir)

    start_time_stamp = ctx.pose3d_start_frame
    end_time_stamp = ctx.pose3d_end_frame if ctx.pose3d_end_frame > 0 else len(dset)
    time_stamps = range(start_time_stamp, end_time_stamp)

    ## check if ctx.pose2d_dir, 
    camera_pose2d_files = [os.path.join(ctx.pose2d_dir, f"pose2d_{exo_camera_name}.pkl") for exo_camera_name in ctx.exo_cam_names]

    ## check if all camera pose2d files exist
    is_parallel = True
    for camera_pose2d_file in camera_pose2d_files:
        if not os.path.exists(camera_pose2d_file):
            is_parallel = False
            break
    
    if is_parallel:
        poses2d = {time_stamp: {camera_name: None for camera_name in ctx.exo_cam_names} for time_stamp in time_stamps} 
        for exo_camera_name in ctx.exo_cam_names:
            pose2d_file = os.path.join(ctx.pose2d_dir, f"pose2d_{exo_camera_name}.pkl")
            with open(pose2d_file, "rb") as f:
                poses2d_camera = pickle.load(f)

            for time_stamp in time_stamps:
                poses2d[time_stamp][exo_camera_name] = poses2d_camera[time_stamp][exo_camera_name]

    else:
        ## load pose2d.pkl
        pose2d_file = os.path.join(ctx.pose2d_dir, "pose2d.pkl")
        with open(pose2d_file, "rb") as f:
            poses2d = pickle.load(f)
    
    for time_stamp in tqdm(time_stamps):
        info = dset[time_stamp]

        multi_view_pose2d = {exo_camera_name: poses2d[time_stamp][exo_camera_name] for exo_camera_name in ctx.exo_cam_names}

        ## triangulate
        triangulator = Triangulator(time_stamp, ctx.exo_cam_names, exo_cameras, multi_view_pose2d)
        pose3d = triangulator.run(debug=True) ## 17 x 4 (x, y, z, confidence)

        ## visualize pose3d
        for exo_camera_name in ctx.exo_cam_names:
            image_path = info[exo_camera_name]['abs_frame_path']
            image = cv2.imread(image_path)
            exo_camera = exo_cameras[exo_camera_name]

            vis_pose3d_cam_dir = os.path.join(ctx.vis_pose3d_dir, exo_camera_name)
            if not os.path.exists(vis_pose3d_cam_dir):
                os.makedirs(vis_pose3d_cam_dir)

            projected_pose3d =  batch_xworld_to_yimage(pose3d[:, :3], exo_camera)
            projected_pose3d = np.concatenate([projected_pose3d, pose3d[:, 3].reshape(-1, 1)], axis=1) ## 17 x 3

            save_path = os.path.join(vis_pose3d_cam_dir, f"{time_stamp:05d}.jpg")
            pose_model.draw_projected_poses3d([projected_pose3d], image, save_path)
        
        ## save pose3d as timestamp.npy
        np.save(os.path.join(ctx.pose3d_dir, f"{time_stamp:05d}.npy"), pose3d)

    return

##-------------------------------------------------------------------------------
def mode_refine_pose3d(config: Config):
    ctx = get_context(config)

    dset = SyncedEgoExoCaptureDset(
        root_dir=config.root_dir,
        dataset_json_path=ctx.dataset_json_path,
        read_frames=False,
    )
    # by_dev_id = download_andor_generate_streams(
    #     metadata=ctx.metadata_json,
    #     download_video_files=config.mode_preprocess.download_video_files,
    #     force_download=config.mode_preprocess.force_download,
    #     output_dir=ctx.cache_dir,
    # )
    all_cam_ids = set(dset.all_cam_ids())

    # aria_path = by_dev_id["aria01"]["local_path"]
    aria_path = "/home/rawalk/Desktop/datasets/ego4d_data/cache/unc_T1/aria01/aria01.vrs"
    assert not aria_path.startswith("https:") or not aria_path.startswith("s3:")
    # aria_camera_models = get_aria_camera_models(aria_path)

    #---------------import triangulator-------------------
    from ego4d.internal.human_pose.pose_refiner import get_refined_pose3d
    from ego4d.internal.human_pose.pose_estimator import PoseModel
    from ego4d.internal.human_pose.postprocess_pose3d import detect_outliers_and_interpolate

    pose_model = PoseModel(pose_config=ctx.dummy_pose_config, pose_checkpoint=ctx.dummy_pose_checkpoint) ## lightweight for visualization only!

    exo_cameras = {exo_camera_name: create_camera(dset[0][exo_camera_name]["camera_data"], None) for exo_camera_name in ctx.exo_cam_names}
    if not os.path.exists(ctx.refine_pose3d_dir):
        os.makedirs(ctx.refine_pose3d_dir)
    
    ## if ctx.vis_pose3d_dir does not exist make it
    if not os.path.exists(ctx.vis_refine_pose3d_dir):
        os.makedirs(ctx.vis_refine_pose3d_dir)

    ## load all pose3d from ctx.pose3d_dir, they are 00000.npy, 00001.npy, using os.listdir ending in .npy and is 05d, do not use dset
    time_stamps = sorted([int(f.split(".")[0]) for f in os.listdir(ctx.pose3d_dir) if f.endswith(".npy")])
    pose3d_files = [os.path.join(ctx.pose3d_dir, f"{time_stamp:05d}.npy") for time_stamp in time_stamps]

    poses3d = []
    for time_stamp, pose3d_file in enumerate(pose3d_files):
        poses3d.append(np.load(pose3d_file))
    
    poses3d = np.stack(poses3d, axis=0) ## T x 17 x 4 (x, y, z, confidence)

    ## check if ctx.pose2d_dir, 
    camera_pose2d_files = [os.path.join(ctx.pose2d_dir, f"pose2d_{exo_camera_name}.pkl") for exo_camera_name in ctx.exo_cam_names]

    ## check if all camera pose2d files exist
    is_parallel = True
    for camera_pose2d_file in camera_pose2d_files:
        if not os.path.exists(camera_pose2d_file):
            is_parallel = False
            break
    
    if is_parallel:
        poses2d = {time_stamp: {camera_name: None for camera_name in ctx.exo_cam_names} for time_stamp in time_stamps} 
        for exo_camera_name in ctx.exo_cam_names:
            pose2d_file = os.path.join(ctx.pose2d_dir, f"pose2d_{exo_camera_name}.pkl")
            with open(pose2d_file, "rb") as f:
                poses2d_camera = pickle.load(f)

            for time_stamp in time_stamps:
                poses2d[time_stamp][exo_camera_name] = poses2d_camera[time_stamp][exo_camera_name]

    else:
        ## load pose2d.pkl
        pose2d_file = os.path.join(ctx.pose2d_dir, "pose2d.pkl")
        with open(pose2d_file, "rb") as f:
            poses2d = pickle.load(f)

    ## detect outliers and replace with interpolated values, basic smoothing
    poses3d = detect_outliers_and_interpolate(poses3d)

    ## refine pose3d
    poses3d = get_refined_pose3d(poses3d)

    for time_stamp in tqdm(range(len(time_stamps)), total=len(time_stamps)):
        info = dset[time_stamp]
        pose3d = poses3d[time_stamp]

        ## visualize pose3d
        for exo_camera_name in ctx.exo_cam_names:
            image_path = info[exo_camera_name]['abs_frame_path']
            image = cv2.imread(image_path)
            exo_camera = exo_cameras[exo_camera_name]

            vis_refine_pose3d_cam_dir = os.path.join(ctx.vis_refine_pose3d_dir, exo_camera_name)
            if not os.path.exists(vis_refine_pose3d_cam_dir):
                os.makedirs(vis_refine_pose3d_cam_dir)

            projected_pose3d =  batch_xworld_to_yimage(pose3d[:, :3], exo_camera)
            projected_pose3d = np.concatenate([projected_pose3d, pose3d[:, 3].reshape(-1, 1)], axis=1) ## 17 x 3

            save_path = os.path.join(vis_refine_pose3d_cam_dir, f"{time_stamp:05d}.jpg")
            pose_model.draw_projected_poses3d([projected_pose3d], image, save_path)

    ## save poses3d.pkl
    with open(os.path.join(ctx.refine_pose3d_dir, "pose3d.pkl"), "wb") as f:
        pickle.dump(poses3d, f)

    return


##-------------------------------------------------------------------------------
def mode_pose2d(config: Config):
    ctx = get_context(config)

    dset = SyncedEgoExoCaptureDset(
        root_dir=config.root_dir,
        dataset_json_path=ctx.dataset_json_path,
        read_frames=False,
    )
    # by_dev_id = download_andor_generate_streams(
    #     metadata=ctx.metadata_json,
    #     download_video_files=config.mode_preprocess.download_video_files,
    #     force_download=config.mode_preprocess.force_download,
    #     output_dir=ctx.cache_dir,
    # )
    all_cam_ids = set(dset.all_cam_ids())

    # aria_path = by_dev_id["aria01"]["local_path"]
    aria_path = "/home/rawalk/Desktop/datasets/ego4d_data/cache/unc_T1/aria01/aria01.vrs"
    assert not aria_path.startswith("https:") or not aria_path.startswith("s3:")
    # aria_camera_models = get_aria_camera_models(aria_path)

    #---------------construct pose model-------------------
    from ego4d.internal.human_pose.pose_estimator import PoseModel
    pose_model = PoseModel(pose_config=ctx.pose_config, pose_checkpoint=ctx.pose_checkpoint)

    ##--------construct ground plane, it is parallel to the plane with all gopro camera centers----------------
    exo_cameras = {exo_camera_name: create_camera(dset[0][exo_camera_name]["camera_data"], None) for exo_camera_name in ctx.exo_cam_names}
    
    if not os.path.exists(ctx.pose2d_dir):
        os.makedirs(ctx.pose2d_dir)
    
    ## if ctx.vis_pose_dir does not exist make it
    if not os.path.exists(ctx.vis_pose2d_dir):
        os.makedirs(ctx.vis_pose2d_dir)

    poses2d = {}

    ## load bboxes from bbox_dir/bbox.pkl
    bbox_file = os.path.join(ctx.bbox_dir, "bbox.pkl")

    with open(bbox_file, "rb") as f:
        bboxes = pickle.load(f)
    
    for time_stamp in tqdm(range(len(dset)), total=len(dset)):
        info = dset[time_stamp]

        poses2d[time_stamp] = {}

        for exo_camera_name in ctx.exo_cam_names:
            image_path = info[exo_camera_name]['abs_frame_path']
            image = cv2.imread(image_path)

            vis_pose2d_cam_dir = os.path.join(ctx.vis_pose2d_dir, exo_camera_name)
            if not os.path.exists(vis_pose2d_cam_dir):
                os.makedirs(vis_pose2d_cam_dir)

            exo_camera = create_camera(info[exo_camera_name]["camera_data"], None)

            bbox_xyxy = bboxes[time_stamp][exo_camera_name] ## x1, y1, x2, y2
           
            if bbox_xyxy is not None:
                 ## add confidence score to the bbox
                bbox_xyxy = np.append(bbox_xyxy, 1.0)
                
                pose_results = pose_model.get_poses2d(bboxes=[{'bbox': bbox_xyxy}], \
                                        image_name=image_path, \
                                    )

                assert(len(pose_results) == 1)

                save_path = os.path.join(vis_pose2d_cam_dir, f"{time_stamp:05d}.jpg")
                pose_model.draw_poses2d(pose_results, image, save_path)

                pose_result = pose_results[0]
                pose2d = pose_result['keypoints']
            
            else:
                pose2d = None
                save_path = os.path.join(vis_pose2d_cam_dir, f"{time_stamp:05d}.jpg")
                cv2.imwrite(save_path, image)

            poses2d[time_stamp][exo_camera_name] = pose2d

    ## save poses2d to pose2d_dir/pose2d.pkl
    with open(os.path.join(ctx.pose2d_dir, "pose2d.pkl"), "wb") as f:
        pickle.dump(poses2d, f)

    return


# ###-------------------------------------------------------------------------------
def mode_bbox(config: Config):
    ctx = get_context(config)

    dset = SyncedEgoExoCaptureDset(
        root_dir=config.root_dir,
        dataset_json_path=ctx.dataset_json_path,
        read_frames=False,
    )
    # by_dev_id = download_andor_generate_streams(
    #     metadata=ctx.metadata_json,
    #     download_video_files=config.mode_preprocess.download_video_files,
    #     force_download=config.mode_preprocess.force_download,
    #     output_dir=ctx.cache_dir,
    # )
    all_cam_ids = set(dset.all_cam_ids())

    # aria_path = by_dev_id["aria01"]["local_path"]
    # aria_path = "/home/rawalk/Desktop/datasets/ego4d_data/cache/unc_T1/aria01/aria01.vrs"
    # assert not aria_path.startswith("https:") or not aria_path.startswith("s3:")
    # aria_camera_models = get_aria_camera_models(aria_path)

    #---------------construct bbox detector----------------
    from ego4d.internal.human_pose.bbox_detector import DetectorModel
    detector_model = DetectorModel(detector_config=ctx.detector_config, detector_checkpoint=ctx.detector_checkpoint, min_bbox_score=ctx.min_bbox_score)

    ##--------construct ground plane, it is parallel to the plane with all gopro camera centers----------------
    exo_cameras = {exo_camera_name: create_camera(dset[0][exo_camera_name]["camera_data"], None) for exo_camera_name in ctx.exo_cam_names}
    exo_camera_centers = np.array([exo_camera.center for exo_camera_name, exo_camera in exo_cameras.items()])
    camera_plane, camera_plane_unit_normal  = get_exo_camera_plane(exo_camera_centers)
    
    ## if ctx.bbox_dir does not exist make it
    if not os.path.exists(ctx.bbox_dir):
        os.makedirs(ctx.bbox_dir)
    
    ## if ctx.vis_bbox_dir does not exist make it
    if not os.path.exists(ctx.vis_bbox_dir):
        os.makedirs(ctx.vis_bbox_dir)

    bboxes = {}
    
    for time_stamp in tqdm(range(len(dset)), total=len(dset)):
        info = dset[time_stamp]
        bboxes[time_stamp] = {}

        for exo_camera_name in ctx.exo_cam_names:
            image_path = info[exo_camera_name]['abs_frame_path']
            image = cv2.imread(image_path)

            vis_bbox_cam_dir = os.path.join(ctx.vis_bbox_dir, exo_camera_name)
            if not os.path.exists(vis_bbox_cam_dir):
                os.makedirs(vis_bbox_cam_dir)

            exo_camera = create_camera(info[exo_camera_name]["camera_data"], None)
            left_camera = create_camera(info["aria_slam_left"]["camera_data"], None) ## TODO: use the camera model of the aria camera
            right_camera = create_camera(info["aria_slam_right"]["camera_data"], None) ## TODO: use the camera model of the aria camera
            human_center_3d = (left_camera.center + right_camera.center) / 2

            proposal_points_3d = get_region_proposal(human_center_3d, radius=ctx.human_radius, unit_normal=camera_plane_unit_normal, human_height=ctx.human_height)
            proposal_points_2d = batch_xworld_to_yimage(proposal_points_3d, exo_camera)
            proposal_bbox = check_and_convert_bbox(proposal_points_2d, exo_camera.camera_model.width, exo_camera.camera_model.height)

            bbox_xyxy = None
            offshelf_bboxes = None

            if proposal_bbox is not None:
                proposal_bbox = np.array([proposal_bbox[0], proposal_bbox[1], proposal_bbox[2], proposal_bbox[3], 1]) ## add confidnece
                proposal_bboxes = [{'bbox': proposal_bbox}]
                offshelf_bboxes = detector_model.get_bboxes(image_name=image_path, bboxes=proposal_bboxes)

            if offshelf_bboxes is not None:
                assert len(offshelf_bboxes) == 1 ## single human per scene
                bbox_xyxy = offshelf_bboxes[0]['bbox'][:4]

                ## uncomment to visualize the bounding box
                # bbox_image = draw_points_2d(image, proposal_points_2d, radius=5, color=(0, 255, 0))
                bbox_image = draw_bbox_xyxy(image, bbox_xyxy, color=(0, 255, 0))
            # 
            else:
                bbox_image = image
            
            # bbox_image = draw_points_2d(image, proposal_points_2d, radius=5, color=(0, 255, 0))

            cv2.imwrite(os.path.join(vis_bbox_cam_dir, f"{time_stamp:05d}.jpg"), bbox_image)
            bboxes[time_stamp][exo_camera_name] = bbox_xyxy
        
    ## save the bboxes as a pickle file
    with open(os.path.join(ctx.bbox_dir, 'bbox.pkl'), 'wb') as f:
        pickle.dump(bboxes, f)

    return

def mode_preprocess(config: Config):
    """
    Does the following:
    - extract all frames for all cameras
    """
    ctx = get_context(config)
    assert config.mode_preprocess.download_video_files, "must download files"
    by_dev_id = download_andor_generate_streams(
        metadata=ctx.metadata_json,
        download_video_files=config.mode_preprocess.download_video_files,
        force_download=config.mode_preprocess.force_download,
        output_dir=ctx.cache_dir,
    )

    shutil.rmtree(ctx.frame_dir, ignore_errors=True)
    os.makedirs(ctx.frame_dir, exist_ok=True)
    i1 = config.inputs.from_frame_number
    i2 = config.inputs.to_frame_number

    synced_df = get_synced_timesync_df(ctx.metadata_json)
    aria_stream_ks = [
        f"aria01_{stream_id}_capture_timestamp_ns"
        for stream_id in config.inputs.aria_streams
    ]
    aria_t1 = min(synced_df[aria_stream_ks].iloc[i1]) / 1e9 - 1 / 30
    aria_t2 = max(synced_df[aria_stream_ks].iloc[i2]) / 1e9 + 1 / 30

    # # aria
    aria_frame_dir = os.path.join(ctx.frame_dir, "aria01")
    os.makedirs(aria_frame_dir, exist_ok=True)
    print("Extracting aria")
    cmd = [
        config.mode_preprocess.vrs_bin_path,
        "extract-all",
        by_dev_id["aria01"]["local_path"],
    ]
    if not config.mode_preprocess.extract_all_aria_frames:
        cmd += [
            "--after",
            str(aria_t1),
            "--before",
            str(aria_t2),
        ]
    cmd += [
        "--to",
        aria_frame_dir,
    ]
    print("Running:")
    print(" ".join(cmd))
    subprocess.run(cmd)

    # gopro
    for cam in ctx.exo_cam_names:
        cam_frame_dir = os.path.join(ctx.frame_dir, cam)
        os.makedirs(cam_frame_dir, exist_ok=True)
        frame_indices = [
            int(x) for x in synced_df[f"{cam}_frame_number"].iloc[i1 : i2 + 1].tolist()
        ]
        frames = read_frame_idx_set(
            path=by_dev_id[cam]["local_path"],
            frame_indices=frame_indices,
            stream_id=0,
        )
        for idx, frame in tqdm(frames, total=len(frame_indices)):
            out_path = os.path.join(cam_frame_dir, f"{idx:06d}.jpg")
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            assert cv2.imwrite(out_path, frame), out_path

    # TODO: do as separate processing step
    print("Normalizing pose data")
    aria_frames_by_stream = {}
    for stream_id in config.inputs.aria_streams:
        aria_rel_frames = [
            f
            for f in os.listdir(os.path.join(aria_frame_dir, stream_id))
            if f.endswith("jpg")
        ]

        aria_frame_to_info = {
            ".".join(f.split("-")[-1].split(".")[0:2]): {
                "t": float(".".join(f.split("-")[-1].split(".")[0:2])),
                "path": os.path.join("aria01", stream_id, f),
            }
            for f in aria_rel_frames
        }
        aria_frames_by_stream[stream_id] = aria_frame_to_info

    aria_traj_df = pd.read_csv(pathmgr.open(config.inputs.aria_trajectory_path))
    exo_traj_df = pd.read_csv(pathmgr.open(config.inputs.exo_trajectory_path))
    output = []

    stream_id_to_name = {
        "211-1": "aria_et",
        "214-1": "aria_rgb",
        "1201-1": "aria_slam_left",
        "1201-2": "aria_slam_right",
    }

    aria_path = by_dev_id["aria01"]["local_path"]
    assert not aria_path.startswith("https:") or not aria_path.startswith("s3:")
    aria_camera_models = get_aria_camera_models(aria_path)
    for idx in tqdm(range(i1, i2 + 1), total=i2 - i1 + 1):
        row = {}
        row_df = synced_df.iloc[idx]
        for stream_id in config.inputs.aria_streams:
            frame_num = int(row_df[f"aria01_{stream_id}_frame_number"])
            frame_t = row_df[f"aria01_{stream_id}_capture_timestamp_ns"] / 1e9
            frame_t = f"{frame_t:.3f}"
            assert frame_t in aria_frames_by_stream[stream_id]
            a_info = aria_frames_by_stream[stream_id][frame_t]
            aria_t = a_info["t"]
            aria_pose = aria_traj_df.iloc[
                (aria_traj_df.tracking_timestamp_us - aria_t * 1e6)
                .abs()
                .argsort()
                .iloc[0]
            ].to_dict()
            row[stream_id_to_name[stream_id]] = {
                "frame_path": a_info["path"],
                "frame_number": idx,
                "t": aria_t,
                "camera_data": create_camera_data(
                    device_row=aria_pose,
                    name=stream_id,
                    camera_model=aria_camera_models[stream_id],
                    device_row_key="device",
                ),
                "_raw_camera": aria_pose,
            }

        for cam_id in ctx.exo_cam_names:
            frame_num = int(row_df[f"{cam_id}_frame_number"])
            frame_path = os.path.join(cam_id, f"{frame_num:06d}.jpg")
            exo_name = (
                config.inputs.exo_timesync_name_to_calib_name[cam_id]
                if config.inputs.exo_timesync_name_to_calib_name
                else cam_id
            )
            # TODO: matching on tracking_timestamp_us when it is supported
            cam_data = exo_traj_df[exo_traj_df.gopro_uid == exo_name].iloc[0].to_dict()

            row[cam_id] = {
                "frame_path": frame_path,
                "frame_number": frame_num,
                "t": None,
                "camera_data": create_camera_data(
                    device_row=cam_data,
                    name=cam_id,
                    camera_model=None,
                    device_row_key="gopro",
                ),
                "_raw_camera": cam_data,
            }
        output.append(row)

    dataset_json = {
        "cache_dir": ctx.cache_rel_dir,
        "dataset_dir": ctx.dataset_rel_dir,
        "frames": output,
    }
    json.dump(dataset_json, open(ctx.dataset_json_path, "w"))


def mode_multi_view_vis(config: Config, flag="pose3d"):
    ctx = get_context(config)
    camera_names = ctx.exo_cam_names

    if flag == "pose3d":
        read_dir = ctx.vis_refine_pose3d_dir
        write_dir = os.path.join(ctx.vis_refine_pose3d_dir, "multi_view")
    elif flag == "bbox":
        read_dir = ctx.vis_bbox_dir
        write_dir = os.path.join(ctx.vis_bbox_dir, "multi_view")
    elif flag == "pose2d":
        read_dir = ctx.vis_pose2d_dir
        write_dir = os.path.join(ctx.vis_pose2d_dir, "multi_view")

    multi_view_vis(camera_names, read_dir, write_dir)

    return

def multi_view_vis(camera_names, read_dir, write_dir):
    os.makedirs(write_dir, exist_ok=True)

    factor = 1
    write_image_width = 3840 // factor
    write_image_height = 2160 // factor

    read_image_width = 3840 // factor
    read_image_height = 2160 // factor

    fps = 30
    padding = 5

    total_width_with_padding = 2*read_image_width + padding
    total_height_with_padding = 2*read_image_height + padding

    total_width = 2*read_image_width
    total_height = 2*read_image_height
    divide_val = 2

    image_names = [image_name for image_name in sorted(os.listdir(os.path.join(read_dir, camera_names[0]))) if image_name.endswith('.jpg')]

    for t, image_name in enumerate(tqdm(image_names)):
        canvas = 255*np.ones((total_height_with_padding, total_width_with_padding, 3))

        for idx, camera_name in enumerate(camera_names):
            camera_image = cv2.imread(os.path.join(read_dir, camera_name, image_name))
            camera_image = cv2.resize(camera_image, (read_image_width, read_image_height))
            
            ##------------paste-----------------
            col_idx = idx % divide_val
            row_idx = idx // divide_val

            origin_x = read_image_width*col_idx + col_idx*padding; 
            origin_y = read_image_height*row_idx + row_idx*padding
            image = camera_image

            canvas[origin_y:origin_y + image.shape[0], origin_x:origin_x + image.shape[1], :] = image[:, :, :]

        ##---------resize to target size, ffmpeg does not work with offset image sizes---------
        canvas = cv2.resize(canvas, (total_width, total_height))
        canvas = cv2.resize(canvas, (write_image_width, write_image_height))

        cv2.imwrite(os.path.join(write_dir, image_name), canvas)

    ##----------make video--------------
    command = 'rm -rf {}/exo.mp4'.format(write_dir)
    os.system(command)

    command = 'ffmpeg -r {} -f image2 -i {}/%05d.jpg -pix_fmt yuv420p {}/exo.mp4'.format(fps, write_dir, read_dir)
    os.system(command)

    return

@hydra.main(config_path="configs", config_name=None)
def run(config: Config):
    if config.mode == "preprocess":
        mode_preprocess(config)
    elif config.mode == "bbox":
        mode_bbox(config)
    elif config.mode == "pose2d":
        mode_pose2d(config)
    elif config.mode == "pose3d":
        mode_pose3d(config)
    elif config.mode == "refine_pose3d":
        mode_refine_pose3d(config)
    elif config.mode == "multi_view_vis_bbox":
        mode_multi_view_vis(config, "bbox")
    elif config.mode == "multi_view_vis_pose2d":
        mode_multi_view_vis(config, "pose2d")
    elif config.mode == "multi_view_vis_pose3d":
        mode_multi_view_vis(config, "pose3d")
    else:
        raise AssertionError(f"unknown mode: {config.mode}")


if __name__ == "__main__":
    run()  # pyre-ignore
