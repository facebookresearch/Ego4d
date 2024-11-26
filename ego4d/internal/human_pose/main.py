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

import cv2

import hydra
import numpy as np
import pandas as pd
from ego4d.internal.colmap.preprocess import download_andor_generate_streams

from ego4d.internal.human_pose.bbox_detector import DetectorModel
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
from ego4d.internal.human_pose.pose_estimator import PoseModel
from ego4d.internal.human_pose.pose_refiner import get_refined_pose3d
from ego4d.internal.human_pose.postprocess_pose3d import detect_outliers_and_interpolate
from ego4d.internal.human_pose.readers import read_frame_idx_set
from ego4d.internal.human_pose.triangulator import Triangulator
from ego4d.internal.human_pose.undistort_to_halo import (
    body_keypoints_list,
    get_default_attachment,
    hand_keypoints_list,
    process_aria_data,
    process_exocam_data,
)
from ego4d.internal.human_pose.utils import (
    aria_extracted_to_original,
    aria_original_to_extracted,
    check_and_convert_bbox,
    draw_bbox_xyxy,
    draw_points_2d,
    get_bbox_from_kpts,
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
    refine_pose3d_dir: Optional[str] = None
    vis_refine_pose3d_dir: Optional[str] = None
    take: Optional[Dict[str, Any]] = None
    all_cams: Optional[List[str]] = None
    frame_rel_dir: Optional[str] = None
    storage_level: int = 30


def _create_json_from_capture_dir(capture_dir: Optional[str]) -> Dict[str, Any]:
    assert capture_dir is not None

    if capture_dir.endswith("/"):
        capture_dir = capture_dir[0:-1]

    video_dir = os.path.join(capture_dir, "videos")

    dirs = capture_dir.split("/")
    take_id = dirs[-1]
    video_source = dirs[-2]
    video_files = pathmgr.ls(video_dir)

    def _create_video(f):
        device_id = os.path.splitext(os.path.basename(f))[0]
        device_type = "aria" if "aria" in device_id else "gopro"
        is_ego = device_type == "aria"
        has_walkaround = "mobile" in device_id or "aria" in device_id
        s3_path = os.path.join(video_dir, f)
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
    take_json_path = os.path.join(config.data_dir, "takes.json")
    takes = json.load(open(take_json_path))
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
        data_dir, "captures", take["capture"]["capture_name"], "trajectory"
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

    # pose2d config
    for rel_path_key in ["pose_config", "dummy_pose_config"]:
        rel_path = config.mode_pose2d[rel_path_key]
        abs_path = os.path.join(config.repo_root_dir, rel_path)
        assert os.path.exists(abs_path), f"{abs_path} does not exist"
        config.mode_pose2d[rel_path_key] = abs_path

    # bbox config
    for rel_path_key in ["detector_config"]:
        rel_path = config.mode_bbox[rel_path_key]
        abs_path = os.path.join(config.repo_root_dir, rel_path)
        assert os.path.exists(abs_path), f"{abs_path} does not exist"
        config.mode_bbox[rel_path_key] = abs_path

    # Hand pose2d config
    config.mode_pose2d.hand_pose_config = os.path.join(
        config.repo_root_dir, config.mode_pose2d.hand_pose_config
    )

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
        refine_pose3d_dir=os.path.join(dataset_dir, "refine_pose3d"),
        vis_refine_pose3d_dir=os.path.join(dataset_dir, "vis_refine_pose3d"),
        take=take,
        hand_pose_config=config.mode_pose2d.hand_pose_config,
        hand_pose_ckpt=config.mode_pose2d.hand_pose_ckpt,
        storage_level=config.outputs.storage_level,
    )


def mode_body_refine_pose3d(config: Config):
    skel_type = "body"

    ctx = get_context(config)

    dset = SyncedEgoExoCaptureDset(
        data_dir=config.cache_root_dir,
        dataset_json_path=ctx.dataset_json_path,
        read_frames=False,
    )

    # ---------------import triangulator-------------------
    pose_model = PoseModel(
        pose_config=ctx.dummy_pose_config, pose_checkpoint=ctx.dummy_pose_checkpoint
    )  ## lightweight for visualization only!

    exo_cameras = {
        exo_camera_name: create_camera(
            dset[0][f"{exo_camera_name}_0"]["camera_data"], None
        )
        for exo_camera_name in ctx.exo_cam_names
    }

    pose2d_dir = os.path.join(ctx.dataset_dir, skel_type, "pose2d")
    pose3d_dir = os.path.join(ctx.dataset_dir, skel_type, "pose3d")
    refine_pose3d_dir = os.path.join(ctx.dataset_dir, skel_type, "refine_pose3d")
    vis_refine_pose3d_dir = os.path.join(
        ctx.dataset_dir, skel_type, "vis_refine_pose3d"
    )

    os.makedirs(refine_pose3d_dir, exist_ok=True)
    os.makedirs(vis_refine_pose3d_dir, exist_ok=True)

    # load all pose3d from pose3d_dir, they are 00000.npy, 00001.npy,
    # using os.listdir ending in .npy and is 05d, do not use dset
    time_stamps = sorted(
        [int(f.split(".")[0]) for f in os.listdir(pose3d_dir) if f.endswith(".npy")]
    )
    pose3d_files = [
        os.path.join(pose3d_dir, f"{time_stamp:05d}.npy") for time_stamp in time_stamps
    ]

    poses3d = []
    for time_stamp, pose3d_file in enumerate(pose3d_files):
        poses3d.append(np.load(pose3d_file))

    poses3d = np.stack(poses3d, axis=0)  ## T x 17 x 4 (x, y, z, confidence)

    ## check if ctx.pose2d_dir,
    camera_pose2d_files = [
        os.path.join(pose2d_dir, f"pose2d_{exo_camera_name}.pkl")
        for exo_camera_name in ctx.exo_cam_names
    ]

    ## check if all camera pose2d files exist
    is_parallel = True
    for camera_pose2d_file in camera_pose2d_files:
        if not os.path.exists(camera_pose2d_file):
            is_parallel = False
            break

    if is_parallel:
        poses2d = {
            time_stamp: {camera_name: None for camera_name in ctx.exo_cam_names}
            for time_stamp in time_stamps
        }
        for exo_camera_name in ctx.exo_cam_names:
            pose2d_file = os.path.join(pose2d_dir, f"pose2d_{exo_camera_name}.pkl")
            with open(pose2d_file, "rb") as f:
                poses2d_camera = pickle.load(f)

            for time_stamp in time_stamps:
                poses2d[time_stamp][exo_camera_name] = poses2d_camera[time_stamp][
                    exo_camera_name
                ]

    else:
        ## load pose2d.pkl
        pose2d_file = os.path.join(pose2d_dir, "pose2d.pkl")
        with open(pose2d_file, "rb") as f:
            poses2d = pickle.load(f)

    ## detect outliers and replace with interpolated values, basic smoothing
    poses3d = detect_outliers_and_interpolate(poses3d)

    ## refine pose3d
    poses3d = get_refined_pose3d(poses3d)

    for time_stamp in tqdm(range(len(time_stamps)), total=len(time_stamps)):
        info = dset[time_stamp]
        pose3d = poses3d[time_stamp]

        # save pose3d as timestamp.npy
        np.save(os.path.join(refine_pose3d_dir, f"{time_stamp:05d}.npy"), pose3d)

        ## visualize pose3d
        for exo_camera_name in ctx.exo_cam_names:
            image_path = info[f"{exo_camera_name}_0"]["abs_frame_path"]
            image = cv2.imread(image_path)
            exo_camera = exo_cameras[exo_camera_name]

            # pyre-ignore
            vis_refine_pose3d_cam_dir = os.path.join(
                vis_refine_pose3d_dir, exo_camera_name
            )
            os.makedirs(vis_refine_pose3d_cam_dir, exist_ok=True)

            projected_pose3d = batch_xworld_to_yimage(pose3d[:, :3], exo_camera)
            projected_pose3d = np.concatenate(
                [projected_pose3d, pose3d[:, 3].reshape(-1, 1)], axis=1
            )  ## 17 x 3

            save_path = os.path.join(vis_refine_pose3d_cam_dir, f"{time_stamp:05d}.jpg")
            pose_model.draw_projected_poses3d([projected_pose3d], image, save_path)

    ## save poses3d.pkl
    # pyre-ignore
    with open(os.path.join(refine_pose3d_dir, "pose3d.pkl"), "wb") as f:
        pickle.dump(poses3d, f)


# -------------------------------------------- New pipeline's code start -------------------------------------------- #
def mode_body_bbox(config: Config):
    """
    Detect human body bbox with aria position as heuristics
    """
    #################################### << Hard coded to show visualization. Can be integrated into args
    visualization = True
    ####################################
    skel_type = "body"

    # Load dataset info
    ctx = get_context(config)
    dset = SyncedEgoExoCaptureDset(
        data_dir=config.cache_root_dir,
        dataset_json_path=ctx.dataset_json_path,
        read_frames=False,
    )
    # Load pretrained body detector
    detector_model = DetectorModel(
        detector_config=ctx.detector_config,
        detector_checkpoint=ctx.detector_checkpoint,
    )

    # construct ground plane, it is parallel to the plane with all gopro camera centers
    # exo_cameras = {
    #     exo_camera_name: create_camera(
    #         dset[0][f"{exo_camera_name}_0"]["camera_data"], None
    #     )
    #     for exo_camera_name in ctx.exo_cam_names
    # }

    # sometimes the exo cameras are not all on the ground,
    # so using get_exo_camera_plane is problematic
    # _, camera_plane_unit_normal = get_exo_camera_plane(exo_camera_centers)
    camera_plane_unit_normal = np.array([0, 0, 1])

    # Directory to store bbox result and visualization
    bbox_dir = os.path.join(ctx.dataset_dir, skel_type, "bbox")
    os.makedirs(bbox_dir, exist_ok=True)
    vis_bbox_dir = os.path.join(ctx.dataset_dir, skel_type, "vis_bbox")
    if visualization:
        os.makedirs(vis_bbox_dir, exist_ok=True)

    bboxes = {}
    for time_stamp in tqdm(range(len(dset)), total=len(dset)):
        info = dset[time_stamp]
        bboxes[time_stamp] = {}

        for exo_camera_name in ctx.exo_cam_names:
            image_path = info[f"{exo_camera_name}_0"]["abs_frame_path"]
            image = cv2.imread(image_path)

            # Directory to store body bbox visualization
            vis_bbox_cam_dir = os.path.join(vis_bbox_dir, exo_camera_name)
            if visualization:
                if not os.path.exists(vis_bbox_cam_dir):
                    os.makedirs(vis_bbox_cam_dir)

            exo_camera = create_camera(
                info[f"{exo_camera_name}_0"]["camera_data"], None
            )
            left_camera = create_camera(
                info[f"{ctx.ego_cam_names[0]}_slam-left"]["camera_data"], None
            )  # TODO: use the camera model of the aria camera
            right_camera = create_camera(
                info[f"{ctx.ego_cam_names[0]}_slam-right"]["camera_data"], None
            )  # TODO: use the camera model of the aria camera
            human_center_3d = (left_camera.center + right_camera.center) / 2

            proposal_points_3d = get_region_proposal(
                human_center_3d,
                unit_normal=camera_plane_unit_normal,
                human_height=ctx.human_height,
            )
            proposal_points_2d = batch_xworld_to_yimage(proposal_points_3d, exo_camera)
            proposal_bbox = check_and_convert_bbox(
                proposal_points_2d,
                exo_camera.camera_model.width,
                exo_camera.camera_model.height,
                min_area_ratio=config.mode_bbox.min_area_ratio,
            )

            bbox_xyxy = None
            offshelf_bboxes = None

            if proposal_bbox is not None:
                proposal_bbox = np.array(
                    [
                        proposal_bbox[0],
                        proposal_bbox[1],
                        proposal_bbox[2],
                        proposal_bbox[3],
                        1,
                    ]
                )  ## add confidnece
                proposal_bboxes = [{"bbox": proposal_bbox}]
                offshelf_bboxes = detector_model.get_bboxes(
                    image_name=image_path, bboxes=proposal_bboxes
                )

            if offshelf_bboxes is not None:
                assert len(offshelf_bboxes) == 1  ## single human per scene
                bbox_xyxy = offshelf_bboxes[0]["bbox"][:4]

                # uncomment to visualize the bounding box
                bbox_image = draw_points_2d(
                    image, proposal_points_2d, radius=5, color=(0, 255, 0)
                )
                bbox_image = draw_bbox_xyxy(image, bbox_xyxy, color=(0, 255, 0))
            else:
                bbox_image = image

            # bbox_image = draw_points_2d(image, proposal_points_2d, radius=5, color=(0, 255, 0))
            if visualization:
                cv2.imwrite(
                    os.path.join(vis_bbox_cam_dir, f"{time_stamp:05d}.jpg"), bbox_image
                )
            bboxes[time_stamp][exo_camera_name] = bbox_xyxy

    # save the bboxes as a pickle file
    with open(os.path.join(bbox_dir, "bbox.pkl"), "wb") as f:
        pickle.dump(bboxes, f)


def mode_body_pose2d(config: Config):
    """
    Human body pose2d estimation with all exo cameras
    """
    ################################# << Hard coded to show visualization. Can be integrated into args
    visualization = True
    #################################
    skel_type = "body"
    step = "pose2d"

    # Load dataset info
    ctx = get_context(config)
    dset = SyncedEgoExoCaptureDset(
        data_dir=config.cache_root_dir,
        dataset_json_path=ctx.dataset_json_path,
        read_frames=False,
    )
    # Load body keypoints estimation model
    pose_model = PoseModel(
        pose_config=ctx.pose_config, pose_checkpoint=ctx.pose_checkpoint
    )

    # Create directory to store body pose2d results and visualization
    pose2d_dir = os.path.join(ctx.dataset_dir, skel_type, step)
    if not os.path.exists(pose2d_dir):
        os.makedirs(pose2d_dir)

    vis_pose2d_dir = os.path.join(ctx.dataset_dir, skel_type, "vis_pose2d")
    if visualization:
        if not os.path.exists(vis_pose2d_dir):
            os.makedirs(vis_pose2d_dir)

    # load bboxes from bbox_dir/bbox.pkl
    bbox_dir = os.path.join(ctx.dataset_dir, skel_type, "bbox")
    bbox_file = os.path.join(bbox_dir, "bbox.pkl")
    if not os.path.exists(bbox_file):
        print(f"bbox path does not exist: {bbox_file}")
        print("NOTE: run mode_bbox")
        sys.exit(1)
    with open(bbox_file, "rb") as f:
        bboxes = pickle.load(f)

    # Iterate through every frame
    poses2d = {}
    for time_stamp in tqdm(range(len(dset)), total=len(dset)):
        info = dset[time_stamp]
        poses2d[time_stamp] = {}
        # Iterate through every cameras
        for exo_camera_name in ctx.exo_cam_names:
            image_path = info[f"{exo_camera_name}_0"]["abs_frame_path"]
            image = cv2.imread(image_path)

            # Directory to store body kpts visualization for current camera
            vis_pose2d_cam_dir = os.path.join(vis_pose2d_dir, exo_camera_name)
            if visualization:
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
                if visualization:
                    save_path = os.path.join(
                        vis_pose2d_cam_dir, f"{time_stamp:05d}.jpg"
                    )
                    pose_model.draw_poses2d(pose_results, image, save_path)
                pose_result = pose_results[0]
                pose2d = pose_result["keypoints"]
            else:
                pose2d = None
                if visualization:
                    save_path = os.path.join(
                        vis_pose2d_cam_dir, f"{time_stamp:05d}.jpg"
                    )
                    cv2.imwrite(save_path, image)

            poses2d[time_stamp][exo_camera_name] = pose2d

    # save poses2d to pose2d_dir/pose2d.pkl
    with open(os.path.join(pose2d_dir, "pose2d.pkl"), "wb") as f:
        pickle.dump(poses2d, f)


def mode_body_pose3d(config: Config):
    """
    Body pose3d estimation with exo cameras, only uses first 17 body kpts for faster speed
    """
    ############################ << Hard coded to show visualization. Can be integrated into args
    visualization = True
    ############################
    skel_type = "body"

    # Load dataset info
    ctx = get_context(config)
    dset = SyncedEgoExoCaptureDset(
        data_dir=config.cache_root_dir,
        dataset_json_path=ctx.dataset_json_path,
        read_frames=False,
    )
    # Load body keypoints estimation model (dummy model for faster visualization)
    pose_model = PoseModel(
        pose_config=ctx.dummy_pose_config, pose_checkpoint=ctx.dummy_pose_checkpoint
    )  # lightweight for visualization only!

    # Load exo cameras
    exo_cameras = {
        exo_camera_name: create_camera(
            dset[0][f"{exo_camera_name}_0"]["camera_data"], None
        )
        for exo_camera_name in ctx.exo_cam_names
    }

    # Directory to store pose3d result and visualization
    pose3d_dir = os.path.join(ctx.dataset_dir, skel_type, "pose3d")
    if not os.path.exists(pose3d_dir):
        os.makedirs(pose3d_dir)
    vis_pose3d_dir = os.path.join(ctx.dataset_dir, skel_type, "vis_pose3d")
    if visualization:
        if not os.path.exists(vis_pose3d_dir):
            os.makedirs(vis_pose3d_dir)

    # Load body pose2d estimation result
    pose2d_file = os.path.join(ctx.dataset_dir, skel_type, "pose2d", "pose2d.pkl")
    assert os.path.exists(pose2d_file), f"{pose2d_file} does not exist"
    with open(pose2d_file, "rb") as f:
        poses2d = pickle.load(f)

    # Body pose3d estimation starts
    poses3d = {}
    reprojection_errors = {}
    for time_stamp in tqdm(range(len(dset)), total=len(dset)):
        info = dset[time_stamp]
        reprojection_errors[time_stamp] = {}

        multi_view_pose2d = {
            exo_camera_name: poses2d[time_stamp][exo_camera_name]
            for exo_camera_name in ctx.exo_cam_names
        }

        # triangulate
        triangulator = Triangulator(
            time_stamp,
            ctx.exo_cam_names,
            exo_cameras,
            multi_view_pose2d,
            keypoint_thres=config.mode_pose3d.min_body_kpt2d_conf,
        )

        pose3d = triangulator.run(debug=False)  ## 17 x 4 (x, y, z, confidence)
        poses3d[time_stamp] = pose3d

        # save pose3d as timestamp.npy
        np.save(os.path.join(pose3d_dir, f"{time_stamp:05d}.npy"), pose3d)

        # visualize pose3d
        if visualization:
            for exo_camera_name in ctx.exo_cam_names:
                image_path = info[f"{exo_camera_name}_0"]["abs_frame_path"]
                image = cv2.imread(image_path)
                exo_camera = exo_cameras[exo_camera_name]

                vis_pose3d_cam_dir = os.path.join(vis_pose3d_dir, exo_camera_name)
                if not os.path.exists(vis_pose3d_cam_dir):
                    os.makedirs(vis_pose3d_cam_dir)

                projected_pose3d = batch_xworld_to_yimage(pose3d[:, :3], exo_camera)
                projected_pose3d = np.concatenate(
                    [projected_pose3d, pose3d[:, 3].reshape(-1, 1)], axis=1
                )  ## 17 x 3

                save_path = os.path.join(vis_pose3d_cam_dir, f"{time_stamp:05d}.jpg")
                pose_model.draw_projected_poses3d([projected_pose3d], image, save_path)

        # Compute reprojection error
        invalid_index = pose3d[:, 2] == 0
        for camera_name in ctx.exo_cam_names:
            # Extract projected pose3d results onto current camera plane
            curr_camera = exo_cameras[camera_name]
            projected_pose3d = batch_xworld_to_yimage(pose3d[:, :3], curr_camera)

            if poses2d[time_stamp][camera_name] is None:
                # Assign reprojection error to be -1
                # for all indices since they are all invalid
                reprojection_error = -np.ones(17)
            else:
                # Compute L1-norm between projected 2D kpts and hand_pose2d
                original_pose2d = poses2d[time_stamp][camera_name][:17, :2]
                reprojection_error = np.linalg.norm(
                    (original_pose2d - projected_pose3d), ord=1, axis=1
                )
                # Assign invalid index's reprojection error to be -1
                reprojection_error[invalid_index] = -1

            # Append result
            reprojection_errors[time_stamp][camera_name] = reprojection_error.reshape(
                -1, 1
            )

    # Save pose3d kpts result
    with open(os.path.join(pose3d_dir, "body_pose3d.pkl"), "wb") as f:
        pickle.dump(poses3d, f)
    # Save reprojection errors
    with open(
        os.path.join(pose3d_dir, "body_pose3d_reprojection_error.pkl"), "wb"
    ) as f:
        pickle.dump(reprojection_errors, f)


def mode_wholebodyHand_pose3d(config: Config):
    """
    Body pose3d estimation with exo cameras, but with only Wholebody-hand kpts (42 points)
    """
    ctx = get_context(config)
    # TODO: Integrate those hardcoded values into args
    ##################################
    # Select all default cameras: ctx.exo_cam_names or manual seelction: ['cam01','cam02']
    exo_cam_names = ctx.exo_cam_names
    # wholebody-Hand kpts confidence threshold to perform triangulation
    tri_threshold = 0.5
    # Whether show visualization
    visualization = True
    ##################################

    # Load dataset info
    dset = SyncedEgoExoCaptureDset(
        data_dir=config.cache_root_dir,
        dataset_json_path=ctx.dataset_json_path,
        read_frames=False,
    )
    # Load hand keypoints estimation model (dummy model for faster visualization)
    pose_model = PoseModel(
        pose_config=ctx.hand_pose_config,
        pose_checkpoint=ctx.hand_pose_ckpt,
        rgb_keypoint_thres=tri_threshold,
        rgb_keypoint_vis_thres=tri_threshold,
    )

    # Load exo cameras
    exo_cameras = {
        exo_camera_name: create_camera(
            dset[0][f"{exo_camera_name}_0"]["camera_data"], None
        )
        for exo_camera_name in ctx.exo_cam_names
    }

    # Directory to store pose3d result and visualization
    pose3d_dir = os.path.join(ctx.dataset_dir, "body/pose3d")
    if not os.path.exists(pose3d_dir):
        os.makedirs(pose3d_dir)

    vis_pose3d_dir = os.path.join(
        ctx.dataset_dir, f"body/vis_pose3d/wholebodyHand_triThresh={tri_threshold}"
    )
    if visualization:
        if not os.path.exists(vis_pose3d_dir):
            os.makedirs(vis_pose3d_dir)

    # Load body pose2d estimation result
    pose2d_file = os.path.join(ctx.dataset_dir, "body/pose2d", "pose2d.pkl")
    assert os.path.exists(pose2d_file), f"{pose2d_file} does not exist"
    with open(pose2d_file, "rb") as f:
        poses2d = pickle.load(f)

    # Body pose3d estimation starts
    poses3d = {}
    for time_stamp in tqdm(range(len(dset)), total=len(dset)):
        info = dset[time_stamp]

        ########### Heuristic Check: Hardcode hand wrist kpt conf to be 1 ################################
        multi_view_pose2d = {}
        for exo_camera_name in exo_cam_names:
            if poses2d[time_stamp][exo_camera_name] is None:
                multi_view_pose2d[exo_camera_name] = None
            else:
                curr_exo_hand_pose2d_kpts = poses2d[time_stamp][exo_camera_name][-42:]
                if np.mean(curr_exo_hand_pose2d_kpts[:, -1]) > 0.3:
                    curr_exo_hand_pose2d_kpts[[0, 21], 2] = 1
                multi_view_pose2d[exo_camera_name] = curr_exo_hand_pose2d_kpts
        ##################################################################################################

        ###### Heuristic Check: If two hands are too close, then drop the one with lower confidence ######
        for exo_camera_name in exo_cam_names:
            if multi_view_pose2d[exo_camera_name] is not None:
                right_hand_pos2d_kpts, left_hand_pos2d_kpts = (
                    multi_view_pose2d[exo_camera_name][:21, :],
                    multi_view_pose2d[exo_camera_name][21:, :],
                )
                pairwise_conf_dis = (
                    np.linalg.norm(
                        left_hand_pos2d_kpts[:, :2] - right_hand_pos2d_kpts[:, :2],
                        axis=1,
                    )
                    * right_hand_pos2d_kpts[:, 2]
                    * left_hand_pos2d_kpts[:, 2]
                )
                # Drop lower kpts result if pairwise_conf_dis is too low
                if np.mean(pairwise_conf_dis) < 5:
                    right_conf_mean = np.mean(right_hand_pos2d_kpts[:, 2])
                    left_conf_mean = np.mean(left_hand_pos2d_kpts[:, 2])
                    if right_conf_mean < left_conf_mean:
                        right_hand_pos2d_kpts[:, :] = 0
                    else:
                        left_hand_pos2d_kpts[:, :] = 0
                multi_view_pose2d[exo_camera_name][:21] = right_hand_pos2d_kpts
                multi_view_pose2d[exo_camera_name][21:] = left_hand_pos2d_kpts
        ###################################################################################################

        # triangulate
        triangulator = Triangulator(
            time_stamp,
            exo_cam_names,
            exo_cameras,
            multi_view_pose2d,
            keypoint_thres=tri_threshold,
            num_keypoints=42,
        )
        pose3d = triangulator.run(debug=False)  ## 17 x 4 (x, y, z, confidence)
        poses3d[time_stamp] = pose3d

        # visualize pose3d
        if visualization:
            for exo_camera_name in ctx.exo_cam_names:
                image_path = info[f"{exo_camera_name}_0"]["abs_frame_path"]
                image = cv2.imread(image_path)
                exo_camera = exo_cameras[exo_camera_name]
                # visualization directory for current camera
                vis_pose3d_cam_dir = os.path.join(vis_pose3d_dir, exo_camera_name)
                if not os.path.exists(vis_pose3d_cam_dir):
                    os.makedirs(vis_pose3d_cam_dir)
                # Project onto current camera
                projected_pose3d = batch_xworld_to_yimage(pose3d[:, :3], exo_camera)
                projected_pose3d = np.concatenate(
                    [projected_pose3d, pose3d[:, 3].reshape(-1, 1)], axis=1
                )  ## N x 3
                # Save visualization
                save_path = os.path.join(vis_pose3d_cam_dir, f"{time_stamp:05d}.jpg")
                pose_model.draw_projected_poses3d(
                    [projected_pose3d[:21], projected_pose3d[21:]], image, save_path
                )

    with open(
        os.path.join(pose3d_dir, f"wholebodyHand_pose3d_triThresh={tri_threshold}.pkl"),
        "wb",
    ) as f:
        pickle.dump(poses3d, f)


def mode_exo_hand_pose2d(config: Config):
    """
    Hand pose2d estimation for all exo cameras, using hand bbox proposed from wholebody-hand kpts
    """
    ctx = get_context(config)
    # TODO: Integrate those hardcoded values into args
    ##################################
    exo_cam_names = ctx.exo_cam_names  # Select all default cameras: ctx.exo_cam_names or manual selection: ['cam01','cam02']
    kpts_vis_threshold = 0.3  # hand pose2d kpts confidence threshold for visualization
    visualization = True  # Whether show visualization
    vis_hand_bbox = ctx.storage_level > 50
    ##################################

    # Load dataset info
    dset = SyncedEgoExoCaptureDset(
        data_dir=config.cache_root_dir,
        dataset_json_path=ctx.dataset_json_path,
        read_frames=False,
    )
    # Hand pose estimation model
    hand_pose_model = PoseModel(
        pose_config=ctx.hand_pose_config,
        pose_checkpoint=ctx.hand_pose_ckpt,
        rgb_keypoint_thres=kpts_vis_threshold,
        rgb_keypoint_vis_thres=kpts_vis_threshold,
        refine_bbox=False,
    )

    # Directory to store bbox and pose2d kpts
    bbox_dir = os.path.join(ctx.dataset_dir, f"hand/bbox")
    os.makedirs(bbox_dir, exist_ok=True)
    pose2d_dir = os.path.join(ctx.dataset_dir, f"hand/pose2d")
    if not os.path.exists(pose2d_dir):
        os.makedirs(pose2d_dir)

    # Directory to store pose2d estimation
    vis_pose2d_dir = os.path.join(
        ctx.dataset_dir, f"hand/vis_pose2d/visThresh={kpts_vis_threshold}"
    )
    # Directory to store hand bbox
    vis_bbox_dir = os.path.join(ctx.dataset_dir, f"hand/vis_bbox")
    if visualization:
        if not os.path.exists(vis_pose2d_dir):
            os.makedirs(vis_pose2d_dir)
    if vis_hand_bbox:
        os.makedirs(vis_bbox_dir, exist_ok=True)

    # Load human body keypoints result from mode_pose2d
    body_pose2d_path = os.path.join(ctx.dataset_dir, "body/pose2d", "pose2d.pkl")
    assert os.path.exists(body_pose2d_path), f"{body_pose2d_path} does not exist"
    with open(body_pose2d_path, "rb") as f:
        body_poses2d = pickle.load(f)

    # Iterate through every frame
    poses2d = {}
    bboxes = {}
    for time_stamp in tqdm(range(len(dset)), total=len(dset)):
        info = dset[time_stamp]
        poses2d[time_stamp] = {}
        bboxes[time_stamp] = {}
        # Iterate through every cameras
        for exo_camera_name in exo_cam_names:
            image_path = info[f"{exo_camera_name}_0"]["abs_frame_path"]
            image = cv2.imread(image_path)

            # Directory to store hand pose2d results
            vis_pose2d_cam_dir = os.path.join(vis_pose2d_dir, exo_camera_name)
            if visualization:
                if not os.path.exists(vis_pose2d_cam_dir):
                    os.makedirs(vis_pose2d_cam_dir)

            # Directory to store hand bbox results
            vis_bbox_cam_dir = os.path.join(vis_bbox_dir, exo_camera_name)
            if vis_hand_bbox:
                if not os.path.exists(vis_bbox_cam_dir):
                    os.makedirs(vis_bbox_cam_dir)

            # Extract left and right hand hpts from wholebody kpts estimation
            body_pose_kpts = body_poses2d[time_stamp][exo_camera_name]

            # If there is no wholebody-Hand results, then assign None as hand bbox
            if body_pose_kpts is None:
                right_hand_bbox, left_hand_bbox = None, None
            else:
                # Right hand kpts
                right_hand_kpts_index = list(range(112, 132))
                right_hand_kpts = body_pose_kpts[right_hand_kpts_index, :]
                # Left hand kpts
                left_hand_kpts_index = list(range(91, 111))
                left_hand_kpts = body_pose_kpts[left_hand_kpts_index, :]

                ############## Hand bbox ##############
                img_H, img_W = image.shape[:2]
                right_hand_bbox = get_bbox_from_kpts(
                    right_hand_kpts, img_W, img_H, padding=50
                )
                left_hand_bbox = get_bbox_from_kpts(
                    left_hand_kpts, img_W, img_H, padding=50
                )
                ################# Heuristic Check: If wholeBody-Hand kpts confidence is too low, then assign zero bbox #################
                right_kpts_avgConf, left_kpts_avgConf = (
                    np.mean(right_hand_kpts[:, 2]),
                    np.mean(left_hand_kpts[:, 2]),
                )
                if right_kpts_avgConf < 0.5:
                    right_hand_bbox = None
                if left_kpts_avgConf < 0.5:
                    left_hand_bbox = None
                ########################################################################################################################

            # Append result
            bboxes[time_stamp][exo_camera_name] = [right_hand_bbox, left_hand_bbox]

            # Visualization
            if vis_hand_bbox:
                vis_bbox_img = image.copy()
                vis_bbox_img = (
                    draw_bbox_xyxy(vis_bbox_img, right_hand_bbox, color=(255, 0, 0))
                    if right_hand_bbox is not None
                    else vis_bbox_img
                )
                vis_bbox_img = (
                    draw_bbox_xyxy(vis_bbox_img, left_hand_bbox, color=(0, 0, 255))
                    if left_hand_bbox is not None
                    else vis_bbox_img
                )
                cv2.imwrite(
                    os.path.join(vis_bbox_cam_dir, f"{time_stamp:05d}.jpg"),
                    vis_bbox_img,
                )

            ############## Hand pose 2d ##############
            # Append confience score to bbox
            bbox_xyxy_right = (
                np.append(right_hand_bbox, 1)
                if right_hand_bbox is not None
                else np.array([0, 0, 0, 0, 1])
            )
            bbox_xyxy_left = (
                np.append(left_hand_bbox, 1)
                if left_hand_bbox is not None
                else np.array([0, 0, 0, 0, 1])
            )
            two_hand_bboxes = [{"bbox": bbox_xyxy_right}, {"bbox": bbox_xyxy_left}]
            # Hand pose estimation
            pose_results = hand_pose_model.get_poses2d(
                bboxes=two_hand_bboxes,
                image_name=image_path,
            )
            # Save 2d hand pose estimation result ~ (2,21,3)
            curr_pose2d_kpts = [res["keypoints"] for res in pose_results]
            # Assign None if hand bbox is None
            if right_hand_bbox is None:
                curr_pose2d_kpts[0] = None
            if left_hand_bbox is None:
                curr_pose2d_kpts[1] = None

            # Append pose2d result
            poses2d[time_stamp][exo_camera_name] = curr_pose2d_kpts

            # Visualization
            if visualization:
                save_path = os.path.join(vis_pose2d_cam_dir, f"{time_stamp:05d}.jpg")
                vis_pose2d_img = image.copy()
                hand_pose_model.draw_poses2d(
                    [pose_results[0]], vis_pose2d_img, save_path
                )
                vis_pose2d_img = cv2.imread(save_path)
                hand_pose_model.draw_poses2d(
                    [pose_results[1]], vis_pose2d_img, save_path
                )

    # save poses2d key points result
    with open(os.path.join(pose2d_dir, "exo_pose2d.pkl"), "wb") as f:
        pickle.dump(poses2d, f)

    # save the bboxes as a pickle file
    with open(os.path.join(bbox_dir, "exo_bbox.pkl"), "wb") as f:
        pickle.dump(bboxes, f)


def mode_ego_hand_pose2d(config: Config):
    """
    Hand bbox detection & pose2d estimation for aria images.
    """
    ctx = get_context(config)
    # TODO: Integrate those hardcoded values into args
    ################# Modified as needed #####################
    ego_cam_names = [f"{cam}_rgb" for cam in ctx.ego_cam_names]
    kpts_vis_threshold = 0.3  # This value determines the threshold to visualize hand pose2d estimated kpts
    tri_threshold = 0.5  # This value determines which wholebody-Hand pose3d kpts to use
    visualization = True
    vis_hand_bbox = ctx.storage_level > 50
    ##########################################################

    # Load dataset info
    dset = SyncedEgoExoCaptureDset(
        data_dir=config.cache_root_dir,
        dataset_json_path=ctx.dataset_json_path,
        read_frames=False,
    )

    # Hand pose estimation model
    hand_pose_model = PoseModel(
        pose_config=ctx.hand_pose_config,
        pose_checkpoint=ctx.hand_pose_ckpt,
        rgb_keypoint_thres=kpts_vis_threshold,
        rgb_keypoint_vis_thres=kpts_vis_threshold,
        refine_bbox=False,
    )

    # Directory to store bbox result and visualization
    bbox_dir = os.path.join(ctx.dataset_dir, f"hand/bbox")
    os.makedirs(bbox_dir, exist_ok=True)
    # Directory to store pose2d result and visualization
    pose2d_dir = os.path.join(ctx.dataset_dir, f"hand/pose2d")
    if not os.path.exists(pose2d_dir):
        os.makedirs(pose2d_dir)

    # Directory to store bbox and pose2d estimation
    vis_pose2d_dir = os.path.join(
        ctx.dataset_dir,
        f"hand/vis_pose2d/visThresh={kpts_vis_threshold}",
    )
    vis_bbox_dir = os.path.join(ctx.dataset_dir, "hand/vis_bbox")
    if visualization:
        if not os.path.exists(vis_pose2d_dir):
            os.makedirs(vis_pose2d_dir)

    if vis_hand_bbox:
        os.makedirs(vis_bbox_dir, exist_ok=True)

    # Load wholebody-Hand pose3d estimation result
    pose3d_dir = os.path.join(
        ctx.dataset_dir,
        "body/pose3d",
        f"wholebodyHand_pose3d_triThresh={tri_threshold}.pkl",
    )
    assert os.path.exists(
        pose3d_dir
    ), f"{pose3d_dir} doesn't exist. Please make sure you have run mode=body_pose3d_wholebodyHand"
    with open(pose3d_dir, "rb") as f:
        wholebodyHand_pose3d = pickle.load(f)

    # Create aria camera model
    capture_dir = os.path.join(
        ctx.data_dir, "captures", ctx.take["capture"]["capture_name"]
    )
    take_dir = os.path.join(ctx.data_dir, "takes", ctx.take["take_name"])
    aria_path = os.path.join(take_dir, f"{ctx.ego_cam_names[0]}.vrs")
    assert os.path.exists(
        aria_path
    ), f"{aria_path} doesn't exit. Need aria video downloaded"
    aria_camera_models = get_aria_camera_models(aria_path)
    stream_name_to_id = {
        f"{ctx.ego_cam_names[0]}_rgb": "214-1",
        f"{ctx.ego_cam_names[0]}_slam-left": "1201-1",
        f"{ctx.ego_cam_names[0]}_slam-right": "1201-2",
    }

    # Iterate through every frame
    poses2d = {}
    bboxes = {}
    for time_stamp in tqdm(range(len(dset)), total=len(dset)):
        info = dset[time_stamp]
        poses2d[time_stamp] = {}
        bboxes[time_stamp] = {}

        # Iterate through every cameras
        for ego_cam_name in ego_cam_names:
            # Load in original image at first
            image_path = info[ego_cam_name]["abs_frame_path"]
            image = cv2.imread(image_path)

            # Create aria camera at this timestamp
            aria_camera = create_camera(
                info[ego_cam_name]["camera_data"],
                aria_camera_models[stream_name_to_id[ego_cam_name]],
            )

            # Directory to save visualization
            vis_pose2d_cam_dir = os.path.join(vis_pose2d_dir, ego_cam_name)
            if visualization:
                # Directory to store hand pose2d results
                if not os.path.exists(vis_pose2d_cam_dir):
                    os.makedirs(vis_pose2d_cam_dir)

            # Directory to store hand bbox results
            vis_bbox_cam_dir = os.path.join(vis_bbox_dir, ego_cam_name)
            if vis_hand_bbox:
                if not os.path.exists(vis_bbox_cam_dir):
                    os.makedirs(vis_bbox_cam_dir)

            ########## Hand bbox from re-projected wholebody-Hand kpts ##########
            # Project wholebody-Hand pose3d kpts onto current aria image plane
            pose3d = wholebodyHand_pose3d[time_stamp]
            projected_pose3d = batch_xworld_to_yimage_check_camera_z(
                pose3d[:, :3], aria_camera
            )
            projected_pose3d = np.concatenate(
                [projected_pose3d, pose3d[:, 3].reshape(-1, 1)], axis=1
            )
            # Reproject hand pose2d kpts onto original aria image plane
            img_H, img_W = image.shape[:2]  # extracted image shape
            orig_H, orig_W = img_W, img_H
            # Get out-of-bound kpts index
            x_valid = np.logical_and(
                projected_pose3d[:, 0] > 0, projected_pose3d[:, 0] < orig_W - 1
            )
            y_valid = np.logical_and(
                projected_pose3d[:, 1] > 0, projected_pose3d[:, 1] < orig_H - 1
            )
            # Get invalid pose3d keypoints (with zero confidence)
            zero_conf_kpts_index = pose3d[:, -1] == 0
            valid_index = x_valid * y_valid * ~zero_conf_kpts_index
            # Rotate from original to extracted view
            extracted_kpts = aria_original_to_extracted(
                projected_pose3d[:, :2], (orig_H - 1, orig_W)
            )
            rot_right_hand_kpts, rot_left_hand_kpts = (
                extracted_kpts[21:],
                extracted_kpts[:21],
            )
            # Filter out zero conf kpts
            rot_right_hand_kpts, rot_left_hand_kpts = (
                rot_right_hand_kpts[valid_index[21:]],
                rot_left_hand_kpts[valid_index[:21]],
            )
            # Propose both hand's bbox based on projected kpts
            right_hand_bbox = (
                get_bbox_from_kpts(rot_right_hand_kpts, img_W, img_H, padding=50)
                if rot_right_hand_kpts.shape[0] > 10
                else None
            )
            left_hand_bbox = (
                get_bbox_from_kpts(rot_left_hand_kpts, img_W, img_H, padding=50)
                if rot_left_hand_kpts.shape[0] > 10
                else None
            )

            # Append bbox result
            bboxes[time_stamp][ego_cam_name] = [right_hand_bbox, left_hand_bbox]

            # Hand bbox visualization
            if vis_hand_bbox:
                vis_bbox_img = image.copy()
                vis_bbox_img = (
                    draw_bbox_xyxy(vis_bbox_img, right_hand_bbox, color=(255, 0, 0))
                    if right_hand_bbox is not None
                    else vis_bbox_img
                )
                vis_bbox_img = (
                    draw_bbox_xyxy(vis_bbox_img, left_hand_bbox, color=(0, 0, 255))
                    if left_hand_bbox is not None
                    else vis_bbox_img
                )
                cv2.imwrite(
                    os.path.join(vis_bbox_cam_dir, f"{time_stamp:05d}.jpg"),
                    vis_bbox_img,
                )

            ########### Hand pose2d estimation on ego camera (aria) ###########
            # Format hand bbox
            bbox_xyxy_right = (
                np.append(right_hand_bbox, 1)
                if right_hand_bbox is not None
                else np.array([0, 0, 0, 0, 1])
            )
            bbox_xyxy_left = (
                np.append(left_hand_bbox, 1)
                if left_hand_bbox is not None
                else np.array([0, 0, 0, 0, 1])
            )
            two_hand_bboxes = [{"bbox": bbox_xyxy_right}, {"bbox": bbox_xyxy_left}]
            # Hand pose estimation
            pose_results = hand_pose_model.get_poses2d(
                bboxes=two_hand_bboxes,
                image_name=image_path,
            )
            # Save result
            curr_pose2d_kpts = [res["keypoints"] for res in pose_results]
            # Assign None if hand bbox is None
            if right_hand_bbox is None:
                curr_pose2d_kpts[0] = None
            if left_hand_bbox is None:
                curr_pose2d_kpts[1] = None

            # Append pose2d result
            poses2d[time_stamp][ego_cam_name] = curr_pose2d_kpts

            # Visualization
            if visualization:
                save_path = os.path.join(vis_pose2d_cam_dir, f"{time_stamp:05d}.jpg")
                vis_twoHand = cv2.imread(image_path)
                hand_pose_model.draw_poses2d([pose_results[0]], vis_twoHand, save_path)
                vis_twoHand = cv2.imread(save_path)
                hand_pose_model.draw_poses2d([pose_results[1]], vis_twoHand, save_path)

    # save poses2d key points result
    with open(os.path.join(pose2d_dir, "ego_pose2d.pkl"), "wb") as f:
        pickle.dump(poses2d, f)
    # save the bboxes as a pickle file
    with open(os.path.join(bbox_dir, "ego_bbox.pkl"), "wb") as f:
        pickle.dump(bboxes, f)


def mode_exo_hand_pose3d(config: Config):
    """
    Hand pose3d estimation with only exo cameras
    """
    ctx = get_context(config)
    # TODO: Integrate those hardcoded values into args
    ################### Modify as needed #################################
    exo_cam_names = ctx.exo_cam_names  # ctx.exo_cam_names  ['cam01','cam02']
    tri_threshold = 0.3
    visualization = True
    wholebody_hand_tri_threshold = 0.5
    use_wholebody_hand_selector = True
    ######################################################################

    # Load dataset info
    dset = SyncedEgoExoCaptureDset(
        data_dir=config.cache_root_dir,
        dataset_json_path=ctx.dataset_json_path,
        read_frames=False,
    )

    # Hand pose estimation model (same as hand_pose2d)
    hand_pose_model = PoseModel(
        pose_config=ctx.hand_pose_config,
        pose_checkpoint=ctx.hand_pose_ckpt,
        rgb_keypoint_thres=tri_threshold,
        rgb_keypoint_vis_thres=tri_threshold,
        refine_bbox=False,
    )

    # Create both aria and exo camera
    exo_cameras = {
        exo_camera_name: create_camera(
            dset[0][f"{exo_camera_name}_0"]["camera_data"], None
        )
        for exo_camera_name in ctx.exo_cam_names
    }

    # Directory to store pose3d result and visualization
    pose3d_dir = os.path.join(ctx.dataset_dir, f"hand/pose3d")
    if not os.path.exists(pose3d_dir):
        os.makedirs(pose3d_dir)
    # Directory to store pose3d visualization
    vis_pose3d_dir = os.path.join(
        ctx.dataset_dir,
        f"hand/vis_pose3d",
        "exo_camera",
        f"triThresh={tri_threshold}",
    )
    if visualization:
        if not os.path.exists(vis_pose3d_dir):
            os.makedirs(vis_pose3d_dir)

    # Load hand pose2d keypoints from exo cameras
    exo_pose2d_file = os.path.join(ctx.dataset_dir, f"hand/pose2d", "exo_pose2d.pkl")
    assert os.path.exists(exo_pose2d_file), f"{exo_pose2d_file} does not exist"
    with open(exo_pose2d_file, "rb") as f:
        exo_poses2d = pickle.load(f)
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

    poses3d = {}
    reprojection_errors = {}
    for time_stamp in tqdm(range(len(dset)), total=len(dset)):
        info = dset[time_stamp]
        reprojection_errors[time_stamp] = {}

        # Pose2d estimation from exo camera
        ###################################
        # Heuristic Check: Hardcode hand wrist kpt conf to be 1

        multi_view_pose2d = {}
        curr_pose2d_dict = {}
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
            curr_pose2d_dict[exo_camera_name] = (
                curr_exo_hand_pose2d_kpts.copy()
            )  # (42,3)
            # Heuristics
            if np.mean(curr_exo_hand_pose2d_kpts[:21, -1]) > 0.3:
                curr_exo_hand_pose2d_kpts[0, -1] = 1
            if np.mean(curr_exo_hand_pose2d_kpts[21:, -1]) > 0.3:
                curr_exo_hand_pose2d_kpts[21, -1] = 1
            # Append kpts result
            multi_view_pose2d[exo_camera_name] = curr_exo_hand_pose2d_kpts
        #
        ###################################

        ###################################
        # Heuristic Check:
        # If two hands are too close, then drop the one with lower confidence

        for exo_camera_name in exo_cam_names:
            right_hand_pos2d_kpts, left_hand_pos2d_kpts = (
                multi_view_pose2d[exo_camera_name][:21, :],
                multi_view_pose2d[exo_camera_name][21:, :],
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
            multi_view_pose2d[exo_camera_name][:21] = right_hand_pos2d_kpts
            multi_view_pose2d[exo_camera_name][21:] = left_hand_pos2d_kpts
        #
        ###################################

        # triangulate
        triangulator = Triangulator(
            time_stamp,
            exo_cam_names,
            exo_cameras,
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
            for camera_name in ctx.exo_cam_names:
                image_path = info[f"{camera_name}_0"]["abs_frame_path"]
                image = cv2.imread(image_path)
                curr_camera = exo_cameras[camera_name]

                vis_pose3d_cam_dir = os.path.join(vis_pose3d_dir, camera_name)
                if not os.path.exists(vis_pose3d_cam_dir):
                    os.makedirs(vis_pose3d_cam_dir)

                projected_pose3d = batch_xworld_to_yimage(pose3d[:, :3], curr_camera)
                projected_pose3d = np.concatenate(
                    [projected_pose3d, pose3d[:, 3].reshape(-1, 1)], axis=1
                )  ## N x 3 (17 for body; 42 for hand)

                save_path = os.path.join(vis_pose3d_cam_dir, f"{time_stamp:05d}.jpg")
                hand_pose_model.draw_projected_poses3d(
                    [projected_pose3d[:21], projected_pose3d[21:]], image, save_path
                )

        # Compute reprojection error
        invalid_index = pose3d[:, 2] == 0
        for camera_name in ctx.exo_cam_names:
            # Extract projected pose3d results onto current camera plane
            curr_camera = exo_cameras[camera_name]
            projected_pose3d = batch_xworld_to_yimage(pose3d[:, :3], curr_camera)
            # Compute L1-norm between projected 2D kpts and hand_pose2d
            original_pose2d = curr_pose2d_dict[camera_name][:, :2]
            reprojection_error = np.linalg.norm(
                (original_pose2d - projected_pose3d), ord=1, axis=1
            )
            # Assign invalid index's reprojection error to be -1
            reprojection_error[invalid_index] = -1
            # Append result
            reprojection_errors[time_stamp][camera_name] = reprojection_error.reshape(
                -1, 1
            )

    # Save pose3d kpts result
    with open(
        os.path.join(pose3d_dir, f"exo_pose3d_triThresh={tri_threshold}.pkl"), "wb"
    ) as f:
        pickle.dump(poses3d, f)
    # Save reprojection errors
    with open(
        os.path.join(
            pose3d_dir, f"exo_pose3d_triThresh={tri_threshold}_reprojection_error.pkl"
        ),
        "wb",
    ) as f:
        pickle.dump(reprojection_errors, f)


def mode_egoexo_hand_pose3d(config: Config):
    """
    Hand pose3d estimation with both ego and exo cameras
    """
    ctx = get_context(config)
    # TODO: Integrate those hardcoded values into args
    ########### Modify as needed #############
    exo_cam_names = ctx.exo_cam_names  # Select all default cameras: ctx.exo_cam_names or manual seelction: ['cam01','cam02']
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
    # Hand pose estimation model (same as hand_pose2d)
    hand_pose_model = PoseModel(
        pose_config=ctx.hand_pose_config,
        pose_checkpoint=ctx.hand_pose_ckpt,
        rgb_keypoint_thres=tri_threshold,
        rgb_keypoint_vis_thres=tri_threshold,
        refine_bbox=False,
    )

    # Create exo cameras
    aria_exo_cameras = {
        exo_camera_name: create_camera(
            dset[0][f"{exo_camera_name}_0"]["camera_data"], None
        )
        for exo_camera_name in ctx.exo_cam_names
    }

    # Directory to store pose3d result and visualization
    pose3d_dir = os.path.join(ctx.dataset_dir, "hand/pose3d")
    if not os.path.exists(pose3d_dir):
        os.makedirs(pose3d_dir)
    # Directory to store pose3d visualization
    vis_pose3d_dir = os.path.join(
        ctx.dataset_dir,
        "hand/vis_pose3d",
        "ego_exo_camera",
        f"triThresh={tri_threshold}",
    )
    if visualization:
        if not os.path.exists(vis_pose3d_dir):
            os.makedirs(vis_pose3d_dir)

    # Load hand pose2d keypoints from both cam and aria
    cam_pose2d_file = os.path.join(ctx.dataset_dir, "hand/pose2d", "exo_pose2d.pkl")
    assert os.path.exists(cam_pose2d_file), f"{cam_pose2d_file} does not exist"
    with open(cam_pose2d_file, "rb") as f:
        exo_poses2d = pickle.load(f)
    aria_pose2d_file = os.path.join(ctx.dataset_dir, "hand/pose2d", "ego_pose2d.pkl")
    assert os.path.exists(aria_pose2d_file), f"{aria_pose2d_file} does not exist"
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
        ctx.data_dir, "captures", ctx.take["capture"]["capture_name"]
    )
    take_dir = os.path.join(ctx.data_dir, "takes", ctx.take["take_name"])
    aria_path = os.path.join(take_dir, f"{ctx.ego_cam_names[0]}.vrs")
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
                img_name_path = (
                    camera_name if "aria" in camera_name else f"{camera_name}_0"
                )
                image_path = info[img_name_path]["abs_frame_path"]
                image = cv2.imread(image_path)
                curr_camera = aria_exo_cameras[camera_name]

                vis_pose3d_cam_dir = os.path.join(vis_pose3d_dir, camera_name)
                if not os.path.exists(vis_pose3d_cam_dir):
                    os.makedirs(vis_pose3d_cam_dir)

                projected_pose3d = batch_xworld_to_yimage(pose3d[:, :3], curr_camera)
                projected_pose3d = np.concatenate(
                    [projected_pose3d, pose3d[:, 3].reshape(-1, 1)], axis=1
                )  ## N x 3 (17 for body,; 42 for hand)

                if "aria" in camera_name:
                    projected_pose3d = aria_original_to_extracted(projected_pose3d)

                save_path = os.path.join(vis_pose3d_cam_dir, f"{time_stamp:05d}.jpg")
                hand_pose_model.draw_projected_poses3d(
                    [projected_pose3d[:21], projected_pose3d[21:]], image, save_path
                )

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

    # Save pose3d kpts result
    with open(
        os.path.join(pose3d_dir, f"egoexo_pose3d_triThresh={tri_threshold}.pkl"), "wb"
    ) as f:
        pickle.dump(poses3d, f)
        # Save reprojection errors
    with open(
        os.path.join(
            pose3d_dir,
            f"egoexo_pose3d_triThresh={tri_threshold}_reprojection_error.pkl",
        ),
        "wb",
    ) as f:
        pickle.dump(reprojection_errors, f)


# ----------- New pipeline's code end ----------- #


def mode_preprocess_legacy(config: Config):
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

    synced_df = get_synced_timesync_df(ctx.metadata_json)
    aria_stream_ks = [
        f"aria01_{stream_id}_capture_timestamp_ns"
        for stream_id in config.inputs.aria_streams
    ]

    num_frames = len(synced_df)
    i1 = max(0, min(config.inputs.from_frame_number, num_frames - 1))
    i2 = max(0, min(config.inputs.to_frame_number, num_frames - 1))
    print(f"[Info] Final frame range: {i1} ~ {i2}")

    aria_t1 = min(synced_df[aria_stream_ks].iloc[i1]) / 1e9 - 1 / 30
    aria_t2 = max(synced_df[aria_stream_ks].iloc[i2]) / 1e9 + 1 / 30

    aria_frame_dir = os.path.join(ctx.frame_dir, "aria01")
    os.makedirs(aria_frame_dir, exist_ok=True)
    print("Extracting aria")

    vrs_deps_in_bin_path = False
    vrs_deps = ["libxxhash.so.0", "libturbojpeg.so.0"]
    for vrs_dep in vrs_deps:
        if not os.path.exists(
            os.path.join(os.path.dirname(config.mode_preprocess.vrs_bin_path), vrs_dep)
        ):
            print(
                " ".join(
                    [
                        f"[Info] If you cannot find {vrs_dep}",
                        "when running vrs binary in the next step,",
                        f"consider copying {vrs_dep} to the same folder of",
                        "your vrs_bin_path so we could pick it up.",
                    ]
                )
            )
        else:
            vrs_deps_in_bin_path = True

    if vrs_deps_in_bin_path:
        os.environ["LD_LIBRARY_PATH"] = ":".join(
            [
                os.environ["LD_LIBRARY_PATH"],
                os.path.dirname(config.mode_preprocess.vrs_bin_path),
            ]
        )

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
    subprocess.run(cmd, check=True)

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
        "frame_dir": ctx.frame_rel_dir,
        "dataset_dir": ctx.dataset_rel_dir,
        "frames": output,
    }
    json.dump(dataset_json, open(ctx.dataset_json_path, "w"))


def calculate_frame_selection(
    subclip_json_path, start_frame, end_frame, min_subclip_length
):
    frame_selection = [1 for k in range(start_frame, end_frame)]

    if subclip_json_path is None:
        return frame_selection
    if not os.path.exists(subclip_json_path):
        print(
            f"[Warning] Cannot find sub-clip json: {subclip_json_path}, use all frames"
        )
        return frame_selection

    print(f"[Info] Using {subclip_json_path} to subsample frames")
    with open(subclip_json_path, "r") as f:
        raw_subclips = json.load(f)

    frame_selection = [0 for k in range(start_frame, end_frame)]

    for i in range(len(raw_subclips)):
        left = max(start_frame, raw_subclips[i][0])
        right = min(
            end_frame,
            max(raw_subclips[i][0] + min_subclip_length + 1, raw_subclips[i][1] + 1),
        )
        for k in range(left, right):
            frame_selection[k - start_frame] = 1

    return frame_selection


def mode_preprocess(config: Config):
    if config.legacy:
        mode_preprocess_legacy(config)
    ctx = get_context(config)
    assert config.mode_preprocess.download_video_files, "must download files"
    # Note: sometimes the preprocess takes >72 hours,
    # so we need to resume without deleting saved frames.
    # if there's a change in meta data (e.g., different set of cameras),
    # please manually remove the frames first
    # shutil.rmtree(ctx.frame_dir, ignore_errors=True)
    os.makedirs(ctx.frame_dir, exist_ok=True)

    capture_dir = os.path.join(
        ctx.data_dir, "captures", ctx.take["capture"]["capture_name"]
    )
    take_dir = os.path.join(ctx.data_dir, "takes", ctx.take["take_name"])
    traj_dir = os.path.join(capture_dir, "trajectory")
    aria_traj_path = os.path.join(traj_dir, "closed_loop_trajectory.csv")
    exo_traj_path = os.path.join(traj_dir, "gopro_calibs.csv")
    aria_traj_df = pd.read_csv(aria_traj_path)
    exo_traj_df = pd.read_csv(exo_traj_path)
    all_timesync_df = pd.read_csv(os.path.join(capture_dir, "timesync.csv"))

    # TODO: confirm that this is correct?
    i1, i2 = ctx.take["timesync_start_idx"], ctx.take["timesync_end_idx"] - 1
    synced_df = all_timesync_df.iloc[i1:i2]

    # Use predefined subclip info if it exists,
    # otherwise use start/end to define the frame selection

    # Note: start_frame and end_frame are relative to i1 (i.e., synced_df)
    start_frame = config.inputs.from_frame_number
    end_frame = config.inputs.to_frame_number

    frame_window_size = 1
    if end_frame is None or end_frame > len(synced_df) - frame_window_size:
        end_frame = len(synced_df) - frame_window_size

    if config.inputs.subclip_json_dir is not None:
        subclip_json_path = os.path.join(
            config.inputs.subclip_json_dir, f"{config.inputs.take_name}.json"
        )
    else:
        subclip_json_path = None

    frame_selection = calculate_frame_selection(
        subclip_json_path, start_frame, end_frame, config.inputs.min_subclip_length
    )

    frame_paths = {}
    for cam in ctx.all_cams:
        if cam in ctx.ego_cam_names:
            # TODO: for hands, do we want to use slam left/right?
            streams = config.inputs.aria_streams
        else:
            streams = ["0"]
        for stream_name in streams:
            rel_path = ctx.take["frame_aligned_videos"][cam][stream_name][
                "relative_path"
            ]
            rel_frame_dir = f"{cam}_{stream_name}"
            cam_frame_dir = os.path.join(ctx.frame_dir, f"{cam}_{stream_name}")
            os.makedirs(cam_frame_dir, exist_ok=True)
            path = os.path.join(ctx.data_dir, "takes", ctx.take["take_name"], rel_path)
            reader = PyAvReader(
                path=path,
                resize=None,
                crop=None,
                mean=None,
                std=None,
                frame_window_size=frame_window_size,
                stride=1,
                gpu_idx=-1,
            )

            # n_frames = len(reader)

            key = (cam, stream_name)
            frame_paths[key] = {}

            count = 0
            count_skipped = 0

            for idx in range(start_frame, end_frame, config.inputs.sample_interval):
                if frame_selection[idx - start_frame] == 1:
                    rel_out_path = os.path.join(rel_frame_dir, f"{idx:06d}.jpg")
                    out_path = os.path.join(cam_frame_dir, f"{idx:06d}.jpg")
                    frame_paths[key][idx] = rel_out_path

                    if not config.mode_preprocess.extract_frames:
                        count_skipped += 1
                        if count_skipped % 500 == 0:
                            print(
                                f"[Info] Skipped saving {count_skipped} frames for {cam}_{stream_name}"
                            )
                    elif os.path.exists(out_path):
                        count_skipped += 1
                        if count_skipped % 100 == 0:
                            print(
                                " ".join(
                                    [
                                        "[Info] Found and skipped",
                                        f"{count_skipped} frames for {cam}_{stream_name}",
                                    ]
                                )
                            )
                    else:
                        frame = reader[idx]["video"][0].cpu().numpy()
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        assert cv2.imwrite(out_path, frame), out_path
                        count += 1
                        if count % 100 == 0:
                            print(
                                f"[Info] Saved {count} frames for {cam}_{stream_name}"
                            )

    stream_name_to_id = {
        "et": "211-1",
        "rgb": "214-1",
        "slam-left": "1201-1",
        "slam-right": "1201-2",
    }

    aria_dir = take_dir
    aria_path = os.path.join(aria_dir, f"{ctx.ego_cam_names[0]}.vrs")
    assert os.path.exists(aria_path), f"Cannot find {aria_path}"
    aria_camera_models = get_aria_camera_models(aria_path)

    assert config.inputs.exo_timesync_name_to_calib_name is None

    print("[Info] Preparing metadata for data.json ..")

    output = []
    for idx in range(start_frame, end_frame, config.inputs.sample_interval):
        if frame_selection[idx - start_frame] == 1:
            if (idx - start_frame) % 100 == 0:
                print(f"Processed {idx - start_frame}")
            row = {}
            row_df = synced_df.iloc[idx]
            skip_frame = False
            for stream_name in config.inputs.aria_streams:
                # TODO: support multiple aria cameras?
                key = (ctx.ego_cam_names[0], stream_name)
                key_str = "_".join(key)
                frame_path = frame_paths[key][idx]

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
                row[key_str] = {
                    "frame_path": frame_path,
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

            if skip_frame:
                continue

            for cam_id in ctx.exo_cam_names:
                key = (cam_id, "0")
                key_str = "_".join(key)
                frame_path = frame_paths[key][idx]
                frame_num = int(row_df[f"{cam_id}_frame_number"])
                cam_data = exo_traj_df[exo_traj_df.cam_uid == cam_id].iloc[0].to_dict()

                row[key_str] = {
                    "frame_path": frame_path,
                    "frame_number": idx,
                    "capture_frame_number": frame_num,
                    "t": None,
                    "camera_data": create_camera_data(
                        device_row=cam_data,
                        name=cam_id,
                        camera_model=None,
                        device_row_key="cam",
                    ),
                    "_raw_camera": cam_data,
                }
            output.append(row)

    dataset_json = {
        "cache_dir": ctx.cache_rel_dir,
        "frame_dir": ctx.frame_rel_dir,
        "dataset_dir": ctx.dataset_rel_dir,
        "frames": output,
    }
    json.dump(dataset_json, open(ctx.dataset_json_path, "w"))


def mode_multi_view_vis(config: Config, step="pose3d", skel_type="body"):
    ctx = get_context(config)
    camera_names = ctx.exo_cam_names
    os.makedirs(ctx.vis_pose3d_dir, exist_ok=True)

    if skel_type == "body":
        if step in ["bbox", "pose2d", "pose3d", "refine_pose3d"]:
            read_dir = os.path.join(ctx.dataset_dir, skel_type, f"vis_{step}")
    elif skel_type == "hand":
        # TODO: unify the paths with body
        if step in ["wholebodyhand"]:
            tri_threshold = 0.5
            read_dir = os.path.join(
                ctx.dataset_dir,
                "body",
                "vis_pose3d",
                f"wholebodyHand_triThresh={tri_threshold}",
            )
        elif step in ["pose2d"]:
            kpts_vis_threshold = 0.3
            camera_names.append(f"{ctx.ego_cam_names[0]}_rgb")
            read_dir = os.path.join(
                ctx.dataset_dir,
                skel_type,
                f"vis_{step}",
                f"visThresh={kpts_vis_threshold}",
            )
        elif step in ["pose3d"]:
            tri_threshold = 0.3
            camera_names.append(f"{ctx.ego_cam_names[0]}_rgb")
            read_dir = os.path.join(
                ctx.dataset_dir,
                skel_type,
                f"vis_{step}",
                "ego_exo_camera",
                f"triThresh={tri_threshold}",
            )

    write_dir = os.path.join(ctx.dataset_dir, skel_type, "vis_multiview", step)
    write_video = os.path.join(ctx.vis_pose3d_dir, f"{skel_type}_{step}.mp4")

    multi_view_vis(ctx, camera_names, read_dir, write_dir, write_video)


def multi_view_vis(ctx, camera_names, read_dir, write_dir, write_video):
    os.makedirs(write_dir, exist_ok=True)

    factor = 1
    write_image_width = 3840 // factor
    write_image_height = 2160 // factor

    read_image_width = 3840 // factor
    read_image_height = 2160 // factor

    fps = 30
    padding = 5

    if len(camera_names) <= 4:
        num_cols = 2
        num_rows = 2
    elif len(camera_names) <= 6:
        num_cols = 3
        num_rows = 2
    else:
        raise Exception(
            f"Large number ({len(camera_names)}) of cameras found: {camera_names}. "
            + "Please add visualization layout."
        )

    total_width_with_padding = num_cols * read_image_width + (num_cols - 1) * padding
    total_height_with_padding = num_rows * read_image_height + (num_rows - 1) * padding

    total_width = num_cols * read_image_width
    total_height = num_rows * read_image_height
    divide_val = num_cols

    image_names = [
        image_name
        for image_name in sorted(os.listdir(os.path.join(read_dir, camera_names[0])))
        if image_name.endswith(".jpg")
    ]

    for _t, image_name in enumerate(tqdm(image_names)):
        canvas = 255 * np.ones((total_height_with_padding, total_width_with_padding, 3))

        for idx, camera_name in enumerate(camera_names):
            camera_image = cv2.imread(os.path.join(read_dir, camera_name, image_name))
            camera_image = cv2.resize(
                camera_image, (read_image_width, read_image_height)
            )

            ##------------paste-----------------
            col_idx = idx % divide_val
            row_idx = idx // divide_val

            origin_x = read_image_width * col_idx + col_idx * padding
            origin_y = read_image_height * row_idx + row_idx * padding
            image = camera_image

            canvas[
                origin_y : origin_y + image.shape[0],
                origin_x : origin_x + image.shape[1],
                :,
            ] = image[:, :, :]

        # ---------resize to target size, ffmpeg does not work with offset image sizes---------
        canvas = cv2.resize(canvas, (total_width, total_height))
        canvas = cv2.resize(canvas, (write_image_width, write_image_height))

        cv2.imwrite(os.path.join(write_dir, image_name), canvas)

    # ----------make video--------------
    command = f"rm -rf {write_video}"
    os.system(command)

    command = "ffmpeg -r {} -f image2 -i {}/%05d.jpg -pix_fmt yuv420p {}".format(
        fps, write_dir, write_video
    )
    os.system(command)

    if os.path.exists(write_video):
        if ctx.storage_level <= 50:
            # remove the multiview frames after the video is generated
            print(f"[Info] Removing {write_dir} since {write_video} is generated")
            shutil.rmtree(write_dir)
        if ctx.storage_level <= 40:
            # remove the single-view frames after the video is generated
            print(f"[Info] Removing {read_dir} since {write_video} is generated")
            shutil.rmtree(read_dir)


def mode_undistort_to_halo(config: Config, skel_type="body"):
    ctx = get_context(config)
    use_reproj_error_sample = True

    id_to_halo_map = {}

    if skel_type == "body":
        for id, keypoint in enumerate(body_keypoints_list):
            id_to_halo_map[id] = keypoint["id"]
    elif skel_type == "hand":
        for id, keypoint in enumerate(hand_keypoints_list):
            id_to_halo_map[id] = keypoint["id"]

    capture_id = ctx.cache_dir.strip("/").split("/")[-1]

    if skel_type == "hand":
        use_ego = True
    else:
        use_ego = False

    if use_ego:
        ###############################
        # TODO: extract this to a separate function
        capture_dir = os.path.join(
            ctx.data_dir, "captures", ctx.take["capture"]["capture_name"]
        )
        take_dir = os.path.join(ctx.data_dir, "takes", ctx.take["take_name"])
        aria_path = os.path.join(take_dir, f"{ctx.ego_cam_names[0]}.vrs")
        assert os.path.exists(aria_path), f"Cannot find {aria_path}"
        print(f"Creating data provider from {aria_path}")
        provider = data_provider.create_vrs_data_provider(aria_path)
        assert provider is not None
        #
        ###############################

    ## The naming of aria and gopro might be different; modifiy values below if needed.
    cam_name_map = {}

    for cam in ctx.ego_cam_names:
        cam_name_map[cam] = f"{cam}_rgb"

    for cam in ctx.exo_cam_names:
        cam_name_map[cam] = f"{cam}_0"

    output_images_dir = os.path.join(ctx.cache_dir, skel_type, "halo", "images")
    os.makedirs(output_images_dir, exist_ok=True)
    output_attachments_dir = os.path.join(
        ctx.cache_dir, skel_type, "halo", "attachments"
    )
    os.makedirs(output_attachments_dir, exist_ok=True)

    if skel_type == "hand":
        # Hand pose3d results file path
        pose3d_file = os.path.join(
            ctx.cache_dir, skel_type, "pose3d", "egoexo_pose3d_triThresh=0.3.pkl"
        )
        # Reprojection error file path
        reproj_error_file = os.path.join(
            ctx.cache_dir,
            skel_type,
            "pose3d",
            "egoexo_pose3d_triThresh=0.3_reprojection_error.pkl",
        )
        # Load in ego and exo hand bbox
        exo_hand_bbox_dir = os.path.join(
            ctx.cache_dir, skel_type, "bbox", "exo_bbox.pkl"
        )
        with open(exo_hand_bbox_dir, "rb") as f:
            exo_hand_bboxes = pickle.load(f)
        ego_hand_bbox_dir = os.path.join(
            ctx.cache_dir, skel_type, "bbox", "ego_bbox.pkl"
        )
        with open(ego_hand_bbox_dir, "rb") as f:
            ego_hand_bboxes = pickle.load(f)
        # Concatenated ego and exo hand bboxes as hand bboxes
        all_bboxes = {
            curr_ts: {**exo_hand_bboxes[curr_ts], **ego_hand_bboxes[curr_ts]}
            for curr_ts in exo_hand_bboxes.keys()
        }
    elif skel_type == "body":
        # Body pose3d results file path;
        # if refine_pose3d succeeded, use it, otherwise fall back to pose3d
        pose3d_file = os.path.join(
            ctx.cache_dir, skel_type, "refine_pose3d", "body_pose3d.pkl"
        )
        if not os.path.exists(pose3d_file):
            pose3d_file = os.path.join(
                ctx.cache_dir, skel_type, "pose3d", "body_pose3d.pkl"
            )
        # Body reprojection error file path
        reproj_error_file = os.path.join(
            ctx.cache_dir, skel_type, "pose3d", "body_pose3d_reprojection_error.pkl"
        )
        assert os.path.exists(
            reproj_error_file
        ), "Please first run mode=body_pose3d to get body reprojection error .pkl file"
        # Load body bbox
        body_bbox_file = os.path.join(ctx.cache_dir, skel_type, "bbox", "bbox.pkl")
        with open(body_bbox_file, "rb") as f:
            all_bboxes = pickle.load(f)
    else:
        raise Exception(f"Unknown skeleton type: {skel_type}")

    # Load pose3d results and reprojection error file
    with open(pose3d_file, "rb") as f:
        poses3d = pickle.load(f)
    with open(reproj_error_file, "rb") as f:
        reprojection_error = pickle.load(f)

    # Normalize reprojection error (to account for scale effect)
    reprojection_error = normalize_reprojection_error(
        reprojection_error, all_bboxes, skel_type
    )

    # Run through the data.json file and process each timestamp
    # for corresponding aria-rgb and gopros
    #
    # 1. create a temp attachment json
    # 2. process aria and write K, M
    # 3. process gopros and write ks and Ms
    #
    # naming:
    # 1. image and json share the same name except the extension (required by halo)
    # 2. file name = <CAMERA_ID>_<FRAME_NUMBER>.<EXT>
    # 3. inside the json, frame_number = <FRAME_NUMBER>
    #

    with open(ctx.dataset_json_path, "r") as f:
        json_data = json.load(f)
    frames = json_data["frames"]
    print(f"Total frame number: {len(frames)}")
    print(f"images will be saved to {output_images_dir}")
    print(f"attachments will be saved to {output_attachments_dir}")

    assert len(frames) == len(poses3d)

    id = 0

    for _idx, frame in tqdm(enumerate(frames), total=len(frames)):
        frame_names = []
        curr_reproj_error = reprojection_error[_idx]

        # Process exocams (gopros)
        exocam_list = []
        for cam in ctx.exo_cam_names:
            exocam = cam_name_map[cam]
            if exocam not in frame:
                continue

            exocam_list.append(exocam)

            exocam_data = frame[exocam]
            frame_path = exocam_data["frame_path"]
            frame_path = os.path.join(ctx.frame_dir, frame_path)
            save_name = ("_").join(frame_path.split("/")[-2:])
            frame_name = save_name.split(".")[0]
            frame_names.append(frame_name)

        # Process aria
        if use_ego:
            aria = cam_name_map[ctx.ego_cam_names[0]]
            aria_data = frame[aria]
            frame_path = aria_data["frame_path"]
            frame_path = os.path.join(ctx.frame_dir, frame_path)
            save_name = ("_").join(frame_path.split("/")[-2:])
            frame_name = save_name.split(".")[0]
            frame_names.append(frame_name)

        high_conf_frame_list = []
        for _kp_id in range(len(id_to_halo_map)):
            if use_reproj_error_sample:
                # Extract reprojection error for currrent joint across all views
                all_view_reproj_error = np.array(
                    [
                        curr_reproj_error[curr_view][_kp_id]
                        for curr_view in curr_reproj_error.keys()
                    ]
                ).flatten()
                # Select two views with minimum reprojection error
                best_two_view_index = np.argsort(
                    np.where(all_view_reproj_error == -1, np.inf, all_view_reproj_error)
                )[:2]
                high_conf_frame_list.append(
                    [frame_names[view_idx] for view_idx in best_two_view_index]
                )
            else:
                # randomly sample 2 for each keypoint for now
                high_conf_frame_list.append(random.sample(frame_names, 2))

        # Process aria
        if use_ego:
            default_attachment_json = get_default_attachment()

            default_attachment_json = process_aria_data(
                aria,
                aria_data,
                default_attachment_json,
                ctx.frame_dir,
                provider,
                capture_id,
                poses3d[id],
                id_to_halo_map,
                high_conf_frame_list,
                output_images_dir,
                output_attachments_dir,
            )

        # Process exocams (gopros)
        for exocam in exocam_list:
            exocam_data = frame[exocam]

            default_attachment_json = get_default_attachment()

            default_attachment_json = process_exocam_data(
                exocam,
                exocam_data,
                default_attachment_json,
                ctx.frame_dir,
                capture_id,
                poses3d[id],
                id_to_halo_map,
                high_conf_frame_list,
                output_images_dir,
                output_attachments_dir,
            )

        id += 1


def mode_upload_to_s3(config: Config):
    today = date.today().strftime("%Y%m%d")
    take_name = config.inputs.take_name

    for skel_type in ["body", "hand"]:
        command = " ".join(
            [
                "aws s3 sync",
                f"'{config.cache_root_dir}/cache/{take_name}/{skel_type}/halo'",
                f"'s3://ego4d-fair/egopose/production/{today}/{take_name}/{skel_type}'",
            ]
        )
        print(f"Running command: {command}")
        # os.system(command)
        subprocess.check_call(command, stderr=subprocess.STDOUT, shell=True)

    command = " ".join(
        [
            "aws s3 sync",
            f"'{config.cache_root_dir}/cache/{take_name}/vis_pose3d'",
            f"'s3://ego4d-fair/egopose/production/{today}/{take_name}'",
        ]
    )
    print(f"Running command: {command}")
    subprocess.check_call(command, stderr=subprocess.STDOUT, shell=True)
    # os.system(command)


"""
Newly added main function to run body & hand pose estimation inference
"""


@hydra.main(config_path="configs", config_name=None, version_base=None)
def new_run(config: Config):
    if config.mode == "preprocess":
        mode_preprocess(config)
    elif config.mode == "body_bbox":
        mode_body_bbox(config)
    elif config.mode == "body_pose2d":
        mode_body_pose2d(config)
    elif config.mode == "body_pose3d":
        mode_body_pose3d(config)
    elif config.mode == "wholebodyHand_pose3d":
        mode_wholebodyHand_pose3d(config)
    elif config.mode == "hand_pose2d_exo":
        mode_exo_hand_pose2d(config)
    elif config.mode == "hand_pose2d_ego":
        mode_ego_hand_pose2d(config)
    elif config.mode == "hand_pose3d_exo":
        mode_exo_hand_pose3d(config)
    elif config.mode == "hand_pose3d_egoexo":
        mode_egoexo_hand_pose3d(config)
    elif config.mode == "show_all_config":
        ctx = get_context(config)
        print(ctx)
    else:
        raise AssertionError(f"unknown mode: {config.mode}")


def add_arguments(parser):
    parser.add_argument("--config-name", default="georgiatech_covid_02_2")
    parser.add_argument(
        "--config_path", default="configs", help="Path to the config folder"
    )
    parser.add_argument(
        "--min_resume_date",
        default="20231130",
        type=str,
        help="start from scratch if the date in finished_step.log is earlier",
    )
    parser.add_argument(
        "--take_name",
        default="georgiatech_covid_02_2",
        type=str,
        help="take names to run, concatenated by '+', "
        + "e.g., uniandes_dance_007_3+iiith_cooking_23+nus_covidtest_01",
    )
    parser.add_argument(
        "--resume",
        default="on",
        type=str,
        choices=["on", "off"],
        help="whether to resume from the step in finished_step.log",
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


def main(args):
    # Note: this function is called from launch_main.py
    config = get_hydra_config(args)
    ctx = get_context(config)

    steps = args.steps.split("+")

    resume_log = os.path.join(ctx.dataset_dir, "finished_step.log")
    if args.resume == "on":
        if os.path.exists(resume_log):
            with open(resume_log, "r") as f:
                lines = [line.split("\n")[0] for line in f.readlines()]
            previous_step = lines[0]
            checkpoint_date = lines[1]
            if int(checkpoint_date) >= int(args.min_resume_date):
                if (
                    int(checkpoint_date) >= 20231203
                    and int(checkpoint_date) <= 20240101
                ):
                    if previous_step == "upload_to_s3":
                        previous_step = "hand_undistort_to_halo"
                        print(
                            " ".join(
                                [
                                    "[Warning] Redo upload_to_s3 step",
                                    "between 20231203 and 20240101 due to aws upgrade",
                                ]
                            )
                        )
                print(
                    f"[Info] Resume from previous step {previous_step} on {checkpoint_date}"
                )
                new_steps = []
                reached_previous_skip = False
                for step in steps:
                    if reached_previous_skip:
                        new_steps.append(step)
                    else:
                        print(f"[Info] Skipping {step}")
                        if step == previous_step:
                            reached_previous_skip = True
                steps = new_steps
            else:
                print(
                    f"[Info] Previous checkpoint date {checkpoint_date}"
                    + f"was earlier than {args.min_resume_date}, start from scratch"
                )

    print(f"steps: {steps}")

    skip_body_refine_pose3d = False

    for step in steps:
        print(f"[Info] Running step: {step}")
        start_time = time.time()
        if step == "preprocess":
            mode_preprocess(config)
        elif step == "body_bbox":
            mode_body_bbox(config)
        elif step == "body_pose2d":
            mode_body_pose2d(config)
        elif step == "body_pose3d":
            mode_body_pose3d(config)
        elif step == "body_refine_pose3d":
            try:
                mode_body_refine_pose3d(config)
            except Exception as e:
                # failure is likely due to consistently missing keypoint
                print(f"[Warning] Skipping body_refine_pose3d due to exception:\n{e}")
                traceback = e.__traceback__
                while traceback:
                    print(
                        f"{traceback.tb_frame.f_code.co_filename}: line {traceback.tb_lineno}"
                    )
                    traceback = traceback.tb_next
                skip_body_refine_pose3d = True
        elif step == "body_undistort_to_halo":
            mode_undistort_to_halo(config, skel_type="body")
        elif step == "wholebodyHand_pose3d":
            mode_wholebodyHand_pose3d(config)
        elif step == "hand_pose2d_exo":
            mode_exo_hand_pose2d(config)
        elif step == "hand_pose2d_ego":
            mode_ego_hand_pose2d(config)
        elif step == "hand_pose3d_exo":
            mode_exo_hand_pose3d(config)
        elif step == "hand_pose3d_egoexo":
            mode_egoexo_hand_pose3d(config)
        elif step == "hand_undistort_to_halo":
            mode_undistort_to_halo(config, skel_type="hand")
        elif step == "vis_body_bbox":
            mode_multi_view_vis(config, step="bbox", skel_type="body")
        elif step == "vis_body_pose2d":
            mode_multi_view_vis(config, step="pose2d", skel_type="body")
        elif step == "vis_body_pose3d":
            mode_multi_view_vis(config, step="pose3d", skel_type="body")
        elif step == "vis_hand_wholebodyhand_pose3d":
            mode_multi_view_vis(config, step="wholebodyhand", skel_type="hand")
        elif step == "vis_hand_pose2d":
            mode_multi_view_vis(config, step="pose2d", skel_type="hand")
        elif step == "vis_hand_pose3d":
            mode_multi_view_vis(config, step="pose3d", skel_type="hand")
        elif step == "upload_to_s3":
            mode_upload_to_s3(config)
        elif step == "vis_body_refine_pose3d":
            if not skip_body_refine_pose3d:
                mode_multi_view_vis(config, step="refine_pose3d", skel_type="body")
        else:
            raise Exception(f"Unknown step: {step}")
        print(f"[Info] Time for {step}: {time.time() - start_time} s")

        today = date.today().strftime("%Y%m%d")
        with open(resume_log, "w") as f:
            f.write(f"{step}\n{today}\n")
        print(f"[Info] Updated {resume_log} with {step} on {today}")


if __name__ == "__main__":
    new_run()
    # # Not using hydra:
    # args = parse_args()
    # main(args)
