import argparse
import json
import os
import ast
import pickle
import shutil
import subprocess
import matplotlib.pyplot as plt
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2

import hydra
import time
import numpy as np
import pandas as pd
from ego4d.internal.colmap.preprocess import (
    download_andor_generate_streams
)

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
# from ego4d.internal.human_pose.triangulator_v2 import Triangulator
from ego4d.internal.human_pose.utils import (
    check_and_convert_bbox,
    draw_bbox_xyxy,
    # draw_points_2d,
    get_exo_camera_plane,
    get_region_proposal,
    get_bbox_fromKpts,
    aria_extracted_to_original,
    aria_original_to_extracted
)

from iopath.common.file_io import PathManager
from iopath.common.s3 import S3PathHandler
from tqdm.auto import tqdm

from mmdet.apis import inference_detector, init_detector

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
    exo_cam_names: List[str]
    # bbox_dir: str
    # vis_bbox_dir: str
    # pose2d_dir: str
    # vis_pose2d_dir: str
    # pose3d_dir: str
    # vis_pose3d_dir: str
    detector_config: str
    detector_checkpoint: str
    pose_config: str
    pose_checkpoint: str
    dummy_pose_config: str
    dummy_pose_checkpoint: str
    human_height: float = 1.5



def _create_json_from_capture_dir(capture_dir: str) -> Dict[str, Any]:
    if capture_dir.endswith("/"):
        capture_dir = capture_dir[0:-1]

    if capture_dir.startswith("s3://"):
        bucket_name = capture_dir.split("s3://")[1].split("/")[0]
        prefix_path = f"s3://{bucket_name}"
    else:
        prefix_path = capture_dir

    dirs = capture_dir.split("/")
    take_id = dirs[-1]
    video_source = dirs[-2]
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
        metadata_json = _create_json_from_capture_dir(config.inputs.capture_data_dir)

    if not config.data_dir.startswith("/"):
        config.data_dir = os.path.join(config.repo_root_dir, config.data_dir)
        print(f"Using data dir: {config.data_dir}")

    data_dir = config.data_dir
    cache_rel_dir = os.path.join(
        "cache",
        f"{metadata_json['video_source']}_{metadata_json['take_id']}",
    )
    cache_dir = os.path.join(
        config.data_dir,
        cache_rel_dir,
    )
    exo_cam_names = [
        x["device_id"]
        for x in metadata_json["videos"]
        if not x["is_ego"] and not x["has_walkaround"]
    ]
    dataset_dir = os.path.join(cache_dir, config.mode_preprocess.dataset_name)

    # pose2d config
    for rel_path_key in ["pose_config", "dummy_pose_config"]:
        rel_path = config.mode_pose2d[rel_path_key]
        abs_path = os.path.join(config.repo_root_dir, rel_path)
        assert os.path.exists(
            abs_path
        ), f"path for {rel_path_key} must be relative to root repo dir ({config.repo_root_dir})"

        config.mode_pose2d[rel_path_key] = abs_path

    # bbox config
    for rel_path_key in ["detector_config"]:
        rel_path = config.mode_bbox[rel_path_key]
        abs_path = os.path.join(config.repo_root_dir, rel_path)
        assert os.path.exists(
            abs_path
        ), f"path for {rel_path_key} must be relative to root repo dir ({config.repo_root_dir})"

        config.mode_bbox[rel_path_key] = abs_path

    return Context(
        data_dir=data_dir,
        repo_root_dir=config.repo_root_dir,
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
        # bbox_dir=os.path.join(dataset_dir, "bbox"),
        # vis_bbox_dir=os.path.join(dataset_dir, "vis_bbox"),
        # pose2d_dir=os.path.join(dataset_dir, "pose2d"),
        # vis_pose2d_dir=os.path.join(dataset_dir, "vis_pose2d"),
        # pose3d_dir=os.path.join(dataset_dir, "pose3d"),
        # vis_pose3d_dir=os.path.join(dataset_dir, "vis_pose3d"),
        detector_config=config.mode_bbox.detector_config,
        detector_checkpoint=config.mode_bbox.detector_checkpoint,
        pose_config=config.mode_pose2d.pose_config,
        pose_checkpoint=config.mode_pose2d.pose_checkpoint,
        dummy_pose_config=config.mode_pose2d.dummy_pose_config,
        dummy_pose_checkpoint=config.mode_pose2d.dummy_pose_checkpoint,
        human_height=config.mode_bbox.human_height,
    )


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

    ################ aria ###############
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

    ############### gopro ###############
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
            f for f in os.listdir(os.path.join(aria_frame_dir, stream_id)) if f.endswith("jpg")
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
        ############### aria ###############
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

        ############### exo camera ###############
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



def mode_bbox(config: Config):
    """
    Detect human body bbox with proposed aria
    """
    ctx = get_context(config)
    ####################################
    visualization = False
    ####################################

    dset = SyncedEgoExoCaptureDset(
        data_dir=config.data_dir,
        dataset_json_path=ctx.dataset_json_path,
        read_frames=False,
        legacy=config.legacy
    )

    detector_model = DetectorModel(
        detector_config=ctx.detector_config,
        detector_checkpoint=ctx.detector_checkpoint,
    )

    # construct ground plane, it is parallel to the plane with all gopro camera centers
    exo_cameras = {
        exo_camera_name: create_camera(dset[0][exo_camera_name]["camera_data"], None)
        for exo_camera_name in ctx.exo_cam_names
    }
    exo_camera_centers = np.array(
        [exo_camera.center for exo_camera_name, exo_camera in exo_cameras.items()]
    )
    _, camera_plane_unit_normal = get_exo_camera_plane(exo_camera_centers)


    # Directory to store bbox result and visualization
    bbox_dir = os.path.join(ctx.dataset_dir, 'body/bbox')
    os.makedirs(bbox_dir, exist_ok=True)
    if visualization:
        vis_bbox_dir = os.path.join(ctx.dataset_dir, 'body/vis_bbox')
        os.makedirs(vis_bbox_dir, exist_ok=True)

    bboxes = {}

    for time_stamp in tqdm(range(len(dset)), total=len(dset)):
        info = dset[time_stamp]
        bboxes[time_stamp] = {}

        for exo_camera_name in ctx.exo_cam_names:
            image_path = info[exo_camera_name]["abs_frame_path"]
            image = cv2.imread(image_path)

            if visualization:
                vis_bbox_cam_dir = os.path.join(vis_bbox_dir, exo_camera_name)
                if not os.path.exists(vis_bbox_cam_dir):
                    os.makedirs(vis_bbox_cam_dir)

            exo_camera = create_camera(info[exo_camera_name]["camera_data"], None)
            left_camera = create_camera(
                info["aria_slam_left"]["camera_data"], None
            )  # TODO: use the camera model of the aria camera
            right_camera = create_camera(
                info["aria_slam_right"]["camera_data"], None
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
                # bbox_image = draw_points_2d(image, proposal_points_2d, radius=5, color=(0, 255, 0))
                bbox_image = draw_bbox_xyxy(image, bbox_xyxy, color=(0, 255, 0))
            else:
                bbox_image = image

            # Save visualization results
            if visualization:
                # bbox_image = draw_points_2d(image, proposal_points_2d, radius=5, color=(0, 255, 0))
                cv2.imwrite(os.path.join(vis_bbox_cam_dir, f"{time_stamp:05d}.jpg"), bbox_image)
            # Append bbox result
            bboxes[time_stamp][exo_camera_name] = bbox_xyxy

    # save the bboxes as a pickle file
    with open(os.path.join(bbox_dir, "bbox.pkl"), "wb") as f:
        pickle.dump(bboxes, f)



def mode_body_bbox(config: Config):
    ctx = get_context(config)
    ####################################
    visualization = False
    ####################################
    # Load dataset info
    dset = SyncedEgoExoCaptureDset(
        data_dir=config.data_dir,
        dataset_json_path=ctx.dataset_json_path,
        read_frames=False,
        legacy=config.legacy
    )
    # Pre-trained human bounding box detector
    detector = init_detector(ctx.detector_config, ctx.detector_checkpoint, device='cuda:0')

    # Directory to store bbox result and visualization
    bbox_dir = os.path.join(ctx.dataset_dir, 'body/bbox')
    os.makedirs(bbox_dir, exist_ok=True)
    if visualization:
        vis_bbox_dir = os.path.join(ctx.dataset_dir, 'body/vis_bbox')
        os.makedirs(vis_bbox_dir, exist_ok=True)

    # Iterate through every frames and generate body bbox
    bboxes = {}
    for time_stamp in tqdm(range(len(dset)), total=len(dset)):
        info = dset[time_stamp]
        bboxes[time_stamp] = {}
        # Iterate through every cameras
        for exo_camera_name in ctx.exo_cam_names:
            # Load in image
            image_path = info[exo_camera_name]["abs_frame_path"]

            # bbox visualization save dir
            if visualization:
                vis_bbox_cam_dir = os.path.join(vis_bbox_dir, exo_camera_name)
                if not os.path.exists(vis_bbox_cam_dir):
                    os.makedirs(vis_bbox_cam_dir)

            # Inference
            det_results = inference_detector(detector, image_path)
            curr_bbox = det_results[0].flatten()[:4] ######### Assume single person per frame!!!

            # Append result
            bboxes[time_stamp][exo_camera_name] = curr_bbox

            # Save visualization result
            if visualization:
                original_img = cv2.imread(image_path)
                bbox_img = draw_bbox_xyxy(original_img, curr_bbox)
                cv2.imwrite(os.path.join(vis_bbox_cam_dir, f"{time_stamp:05d}.jpg"), bbox_img)

    # save the bboxes as a pickle file
    with open(os.path.join(bbox_dir, "bbox.pkl"), "wb") as f:
        pickle.dump(bboxes, f)


def mode_body_pose2d(config: Config):
    #################################
    visualization = False
    #################################

    # Load dataset info
    ctx = get_context(config)
    dset = SyncedEgoExoCaptureDset(
        data_dir=config.data_dir,
        dataset_json_path=ctx.dataset_json_path,
        read_frames=False,
        legacy=config.legacy
    )
    # Load body keypoints estimation model
    pose_model = PoseModel(
        pose_config=ctx.pose_config, pose_checkpoint=ctx.pose_checkpoint
    )

    # Create directory to store body pose2d results and visualization
    pose2d_dir = os.path.join(ctx.dataset_dir, 'body/pose2d')
    if not os.path.exists(pose2d_dir):
        os.makedirs(pose2d_dir)
    if visualization:
        vis_pose2d_dir = os.path.join(ctx.dataset_dir, 'body/vis_pose2d')
        if not os.path.exists(vis_pose2d_dir):
            os.makedirs(vis_pose2d_dir)

    # load bboxes from bbox_dir/bbox.pkl
    bbox_dir = os.path.join(ctx.dataset_dir, 'body/bbox')
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
            image_path = info[exo_camera_name]["abs_frame_path"]
            image = cv2.imread(image_path)
            
            # Directory to store body kpts visualization for current camera
            if visualization:
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
                if visualization:
                    save_path = os.path.join(vis_pose2d_cam_dir, f"{time_stamp:05d}.jpg")
                    pose_model.draw_poses2d(pose_results, image, save_path)
                pose_result = pose_results[0]
                pose2d = pose_result["keypoints"]
            else:
                pose2d = None
                if visualization:
                    save_path = os.path.join(vis_pose2d_cam_dir, f"{time_stamp:05d}.jpg")
                    cv2.imwrite(save_path, image)

            poses2d[time_stamp][exo_camera_name] = pose2d

    # save poses2d to pose2d_dir/pose2d.pkl
    with open(os.path.join(pose2d_dir, "pose2d.pkl"), "wb") as f:
        pickle.dump(poses2d, f)



def mode_body_pose3d(config: Config):
    """
    Body pose3d estimation with exo cameras, only uses first 17 body kpts for faster speed
    """
    ############################
    visualization = False
    ############################

    ctx = get_context(config)
    # Load dataset info
    dset = SyncedEgoExoCaptureDset(
        data_dir=config.data_dir,
        dataset_json_path=ctx.dataset_json_path,
        read_frames=False,
        legacy=config.legacy
    )
    # Load body keypoints estimation model (dummy model for faster visualization)
    pose_model = PoseModel(
        pose_config=ctx.dummy_pose_config, pose_checkpoint=ctx.dummy_pose_checkpoint
    )  # lightweight for visualization only!

    # Load exo cameras 
    exo_cameras = {
        exo_camera_name: create_camera(dset[0][exo_camera_name]["camera_data"], None)
        for exo_camera_name in ctx.exo_cam_names
    }

    # Directory to store pose3d result and visualization
    pose3d_dir = os.path.join(ctx.dataset_dir, 'body/pose3d')
    if not os.path.exists(pose3d_dir):
        os.makedirs(pose3d_dir)
    if visualization:
        vis_pose3d_dir = os.path.join(ctx.dataset_dir, 'body/vis_pose3d/body_dummy')
        if not os.path.exists(vis_pose3d_dir):
            os.makedirs(vis_pose3d_dir)

    # Load body pose2d estimation result
    pose2d_file = os.path.join(ctx.dataset_dir, 'body/pose2d', 'pose2d.pkl')
    assert os.path.exists(pose2d_file), f"{pose2d_file} does not exist"
    with open(pose2d_file, "rb") as f:
        poses2d = pickle.load(f)

    # Body pose3d estimation starts
    poses3d = {}
    for time_stamp in tqdm(range(len(dset)), total=len(dset)):
        info = dset[time_stamp]

        multi_view_pose2d = {
            exo_camera_name: poses2d[time_stamp][exo_camera_name]
            for exo_camera_name in ctx.exo_cam_names
        }

        # triangulate
        triangulator = Triangulator(
            time_stamp, 
            ctx.exo_cam_names, 
            exo_cameras, 
            multi_view_pose2d
        )

        pose3d = triangulator.run(debug=False)  ## 17 x 4 (x, y, z, confidence)
        poses3d[time_stamp] = pose3d

        # visualize pose3d
        if visualization:
            for exo_camera_name in ctx.exo_cam_names:
                image_path = info[exo_camera_name]["abs_frame_path"]
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
                # pose_model.draw_projected_poses3d([projected_pose3d[:21], projected_pose3d[21:]], image, save_path)

    with open(os.path.join(pose3d_dir, "body_dummy_pose3d.pkl"), "wb") as f:
        pickle.dump(poses3d, f)



def mode_wholebodyHand_pose3d(config: Config):
    """
    Body pose3d estimation with exo cameras, but with only Wholebody-hand kpts (42 points)
    NOTE:
        1. Hand wrist kpts (index=0) confidence is hardcoded to be 1 for better triangulation results
    """
    ctx = get_context(config)
    ##################################
    exo_cam_names = ctx.exo_cam_names # ctx.exo_cam_names ['cam01','cam02']
    tri_threshold = 0.5
    visualization = False
    ##################################

    # Load dataset info
    dset = SyncedEgoExoCaptureDset(
        data_dir=config.data_dir,
        dataset_json_path=ctx.dataset_json_path,
        read_frames=False,
        legacy=config.legacy
    )
    # Load body keypoints estimation model (dummy model for faster visualization)
    # pose_model = PoseModel(
    #     pose_config=ctx.dummy_pose_config, pose_checkpoint=ctx.dummy_pose_checkpoint
    # )  # lightweight for visualization only!
    hand_pose_config = 'ego4d/internal/human_pose/external/mmlab/mmpose/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/coco_wholebody_hand/hrnetv2_w18_coco_wholebody_hand_256x256_dark.py'
    hand_pose_ckpt = 'https://download.openmmlab.com/mmpose/hand/dark/hrnetv2_w18_coco_wholebody_hand_256x256_dark-a9228c9c_20210908.pth'
    pose_model = PoseModel(
        hand_pose_config, 
        hand_pose_ckpt, 
        rgb_keypoint_thres=tri_threshold, 
        rgb_keypoint_vis_thres=tri_threshold) # Since pose3d assign 1 as confidence for valid 2d kpts, we use the same theshold for visualization

    # Load exo cameras 
    exo_cameras = {
        exo_camera_name: create_camera(dset[0][exo_camera_name]["camera_data"], None)
        for exo_camera_name in exo_cam_names
    }

    # Directory to store pose3d result and visualization
    pose3d_dir = os.path.join(ctx.dataset_dir, 'body/pose3d')
    if not os.path.exists(pose3d_dir):
        os.makedirs(pose3d_dir)
    
    if visualization:
        vis_pose3d_dir = os.path.join(ctx.dataset_dir, f'body/vis_pose3d/wholebodyHand_triThresh={tri_threshold}')
        if not os.path.exists(vis_pose3d_dir):
            os.makedirs(vis_pose3d_dir)

    # Load body pose2d estimation result
    pose2d_file = os.path.join(ctx.dataset_dir, 'body/pose2d', 'pose2d.pkl')
    assert os.path.exists(pose2d_file), f"{pose2d_file} does not exist"
    with open(pose2d_file, "rb") as f:
        poses2d = pickle.load(f)

    # Body pose3d estimation starts
    poses3d = {}
    for time_stamp in tqdm(range(len(dset)), total=len(dset)):
        info = dset[time_stamp]

        # multi_view_pose2d = {
        #     exo_camera_name: poses2d[time_stamp][exo_camera_name][-42:] ###### Only get wholebody-Hand keypoints for triangulation
        #     for exo_camera_name in exo_cam_names
        # }
        ########### Heuristic Check: Hardcode hand wrist kpt conf to be 1 ################################
        multi_view_pose2d = {}
        for exo_camera_name in exo_cam_names:
            curr_exo_hand_pose2d_kpts = poses2d[time_stamp][exo_camera_name][-42:]
            if np.mean(curr_exo_hand_pose2d_kpts[:,-1]) > 0.3:
                curr_exo_hand_pose2d_kpts[[0,21],2] = 1
            multi_view_pose2d[exo_camera_name] = curr_exo_hand_pose2d_kpts
        ##################################################################################################

        # triangulate
        triangulator = Triangulator(
            time_stamp, 
            exo_cam_names, 
            exo_cameras, 
            multi_view_pose2d, 
            keypoint_thres=tri_threshold, 
            num_keypoints=42
        )
        pose3d = triangulator.run(debug=False)  ## 17 x 4 (x, y, z, confidence)
        poses3d[time_stamp] = pose3d

        # visualize pose3d
        if visualization:
            for exo_camera_name in exo_cam_names:
                image_path = info[exo_camera_name]["abs_frame_path"]
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
                pose_model.draw_projected_poses3d([projected_pose3d[:21], projected_pose3d[21:]], image, save_path)

    with open(os.path.join(pose3d_dir, f"wholebodyHand_pose3d_triThresh={tri_threshold}.pkl"), "wb") as f:
        pickle.dump(poses3d, f)



def mode_exo_hand_pose2d(config: Config):
    """
    Hand pose2d estimation for all exo cameras, using hand bbox proposed from wholebody-hand kpts
    """
    ctx = get_context(config)
    ##################################
    exo_cam_names = ctx.exo_cam_names # ctx.exo_cam_names ['cam01','cam02']
    kpts_vis_threshold = 0.3
    visualization = False
    ##################################

    # Load dataset info
    dset = SyncedEgoExoCaptureDset(
        data_dir=config.data_dir,
        dataset_json_path=ctx.dataset_json_path,
        read_frames=False,
        legacy=config.legacy
    )
    # Hand pose estimation model
    ### COCOWholebody hand ###
    hand_pose_config = 'ego4d/internal/human_pose/external/mmlab/mmpose/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/coco_wholebody_hand/hrnetv2_w18_coco_wholebody_hand_256x256_dark.py'
    hand_pose_ckpt = 'https://download.openmmlab.com/mmpose/hand/dark/hrnetv2_w18_coco_wholebody_hand_256x256_dark-a9228c9c_20210908.pth'
    hand_pose_model = PoseModel(
        hand_pose_config, 
        hand_pose_ckpt, 
        rgb_keypoint_thres=kpts_vis_threshold, 
        rgb_keypoint_vis_thres=kpts_vis_threshold,
        refine_bbox=False)

    # Directory to store bbox and pose2d kpts 
    bbox_dir = os.path.join(ctx.dataset_dir, f'hand/bbox')
    os.makedirs(bbox_dir, exist_ok=True)
    pose2d_dir = os.path.join(ctx.dataset_dir, f'hand/pose2d')
    if not os.path.exists(pose2d_dir):
        os.makedirs(pose2d_dir)

    if visualization:
        # Directory to store pose2d estimation
        vis_pose2d_dir = os.path.join(ctx.dataset_dir, f'hand/vis_pose2d/visThresh={kpts_vis_threshold}')
        if not os.path.exists(vis_pose2d_dir):
            os.makedirs(vis_pose2d_dir)
        # Directory to store hand bbox 
        vis_bbox_dir = os.path.join(ctx.dataset_dir, f'hand/vis_bbox')
        os.makedirs(vis_bbox_dir, exist_ok=True)

    # Load human body keypoints result from mode_pose2d
    body_pose2d_path = os.path.join(ctx.dataset_dir, 'body/pose2d', "pose2d.pkl")
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
            image_path = info[exo_camera_name]["abs_frame_path"]
            image = cv2.imread(image_path)
            
            if visualization:
                # Directory to store hand pose2d results
                vis_pose2d_cam_dir = os.path.join(vis_pose2d_dir, exo_camera_name)
                if not os.path.exists(vis_pose2d_cam_dir):
                    os.makedirs(vis_pose2d_cam_dir)
                # Directory to store hand bbox results
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
            right_hand_bbox = get_bbox_fromKpts(right_hand_kpts, img_W, img_H, padding=30)
            left_hand_bbox = get_bbox_fromKpts(left_hand_kpts, img_W, img_H, padding=30)
            ################# Heuristic Check: If wholeBody-Hand kpts confidence is too low, then assign zero bbox #################
            right_kpts_avgConf, left_kpts_avgConf = np.mean(right_hand_kpts[:,2]), np.mean(left_hand_kpts[:,2])
            if right_kpts_avgConf < 0.5:
                right_hand_bbox = np.zeros(4)
            if left_kpts_avgConf < 0.5:
                left_hand_bbox = np.zeros(4)
            ########################################################################################################################
            # Append result
            bboxes[time_stamp][exo_camera_name] = [right_hand_bbox, left_hand_bbox]
            # Visualization
            if visualization:
                vis_bbox_img = image.copy()
                vis_bbox_img = draw_bbox_xyxy(vis_bbox_img, right_hand_bbox, color=(255,0,0))
                vis_bbox_img = draw_bbox_xyxy(vis_bbox_img, left_hand_bbox, color=(0,0,255))
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
            
            # Save 2d hand pose estimation result ~ (2,21,3)
            curr_pose2d_kpts = np.array([res['keypoints'] for res in pose_results])
            poses2d[time_stamp][exo_camera_name] = curr_pose2d_kpts
            
            # Visualization
            if visualization:
                save_path = os.path.join(vis_pose2d_cam_dir, f"{time_stamp:05d}.jpg")
                vis_pose2d_img = image.copy()
                hand_pose_model.draw_poses2d([pose_results[0]], vis_pose2d_img, save_path)
                vis_pose2d_img = cv2.imread(save_path)
                hand_pose_model.draw_poses2d([pose_results[1]], vis_pose2d_img, save_path)

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
    ################# Modified as needed #####################
    ego_cam_name = 'aria_rgb'
    kpts_vis_threshold = 0.3    # This value determines the threshold to visualize hand pose2d estimated kpts
    tri_threshold = 0.5         # This value determines which wholebody-Hand pose3d kpts to use
    visualization = False
    ##########################################################
    
    # Load dataset info
    ctx = get_context(config)
    dset = SyncedEgoExoCaptureDset(
        data_dir=config.data_dir,
        dataset_json_path=ctx.dataset_json_path,
        read_frames=False,
        legacy=config.legacy
    )

    # Hand pose estimation model
    ### COCOWholebody hand ###
    hand_pose_config = 'ego4d/internal/human_pose/external/mmlab/mmpose/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/coco_wholebody_hand/hrnetv2_w18_coco_wholebody_hand_256x256_dark.py'
    hand_pose_ckpt = 'https://download.openmmlab.com/mmpose/hand/dark/hrnetv2_w18_coco_wholebody_hand_256x256_dark-a9228c9c_20210908.pth'
    hand_pose_model = PoseModel(
        hand_pose_config, 
        hand_pose_ckpt, 
        rgb_keypoint_thres=kpts_vis_threshold, 
        rgb_keypoint_vis_thres=kpts_vis_threshold,
        refine_bbox=False)

    # Directory to store bbox result and visualization
    bbox_dir = os.path.join(ctx.dataset_dir, f'hand/bbox')
    os.makedirs(bbox_dir, exist_ok=True)
    # Directory to store pose2d result and visualization
    pose2d_dir = os.path.join(ctx.dataset_dir, f'hand/pose2d')
    if not os.path.exists(pose2d_dir):
        os.makedirs(pose2d_dir)
    
    # Directory to store bbox and pose2d estimation
    if visualization:
        vis_bbox_dir = os.path.join(ctx.dataset_dir, f'hand/vis_bbox', ego_cam_name)
        os.makedirs(vis_bbox_dir, exist_ok=True)
        vis_pose2d_dir = os.path.join(ctx.dataset_dir, f'hand/vis_pose2d/visThresh={kpts_vis_threshold}', ego_cam_name)
        if not os.path.exists(vis_pose2d_dir):
            os.makedirs(vis_pose2d_dir)

    # Load wholebody-Hand pose3d estimation result
    pose3d_dir = os.path.join(ctx.dataset_dir, 'body/pose3d', f"wholebodyHand_pose3d_triThresh={tri_threshold}.pkl")
    assert os.path.exists(pose3d_dir), f'{pose3d_dir} doesn\'t exist. Please make sure you have run mode=body_pose3d_wholebodyHand'
    with open(pose3d_dir, "rb") as f:
        wholebodyHand_pose3d = pickle.load(f)

    # Iterate through every frame
    poses2d = {}
    bboxes = {}
    for time_stamp in tqdm(range(len(dset)), total=len(dset)):
        # Load in original image at first
        image_path = dset[time_stamp]['aria_rgb']['abs_frame_path']
        image = cv2.imread(image_path)
        
        # Create aria camera at this timestamp
        aria_camera = create_camera(dset[time_stamp][ego_cam_name]["camera_data"], None)

        ########################## Hand bbox from re-projected wholebody-Hand kpts ############################
        # Project wholebody-Hand pose3d kpts onto current aria image plane
        pose3d = wholebodyHand_pose3d[time_stamp]
        projected_pose3d = batch_xworld_to_yimage(pose3d[:, :3], aria_camera)
        projected_pose3d = np.concatenate(
                [projected_pose3d, pose3d[:, 3].reshape(-1, 1)], axis=1
            )
        # Propose hand bbox based on projected hand kpts 
        img_H, img_W = image.shape[:2] # image shape of original view aria images
        # Clip hand kpts into valid range
        projected_pose3d[:,0] = np.clip(projected_pose3d[:,0], 0, img_H)
        projected_pose3d[:,1] = np.clip(projected_pose3d[:,1], 0, img_W)
        right_hand_kpts, left_hand_kpts = projected_pose3d[21:], projected_pose3d[:21]
        # Select only nonzero confidence kpts
        right_hand_kpts, left_hand_kpts = right_hand_kpts[right_hand_kpts[:,2]!=0, :2], left_hand_kpts[left_hand_kpts[:,2]!=0, :2]
        right_hand_bbox = get_bbox_fromKpts(right_hand_kpts, img_W, img_H, padding=30)  # Adjust padding as needed
        left_hand_bbox = get_bbox_fromKpts(left_hand_kpts, img_W, img_H, padding=30)    # Adjust padding as needed
        # Save result
        bboxes[time_stamp] = [right_hand_bbox, left_hand_bbox]
        # Hand bbox visualization
        if visualization:
            vis_bbox_img = image.copy()
            vis_bbox_img = draw_bbox_xyxy(vis_bbox_img, right_hand_bbox, color=(255,0,0))
            vis_bbox_img = draw_bbox_xyxy(vis_bbox_img, left_hand_bbox, color=(0,0,255))
            cv2.imwrite(os.path.join(vis_bbox_dir, f"{time_stamp:05d}.jpg"), vis_bbox_img)

        ######################### Hand pose2d estimation on ego camera (aria) ##################################
        # Format hand bbox
        two_hand_bboxes = [{'bbox':np.append(curr_hand_bbox,1)} for curr_hand_bbox in [right_hand_bbox, left_hand_bbox]]
        # Hand pose estimation
        pose_results = hand_pose_model.get_poses2d(
                            bboxes=two_hand_bboxes,
                            image_name=image_path,
                        )
        # Save result
        curr_pose2d_kpts = np.array([res['keypoints'] for res in pose_results])
        poses2d[time_stamp] = curr_pose2d_kpts
        # Visualization
        if visualization:
            save_path = os.path.join(vis_pose2d_dir, f'{time_stamp:06d}.jpg')
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
    ################### MOdify as needed #################################
    exo_cam_names = ctx.exo_cam_names # ctx.exo_cam_names  ['cam01','cam02']
    tri_threshold = 0.3
    visualization = False
    ######################################################################

    # Load dataset info
    dset = SyncedEgoExoCaptureDset(
        data_dir=config.data_dir,
        dataset_json_path=ctx.dataset_json_path,
        read_frames=False,
        legacy=config.legacy
    )

    # Hand pose estimation model (same as hand_pose2d)
    hand_pose_config = 'ego4d/internal/human_pose/external/mmlab/mmpose/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/coco_wholebody_hand/hrnetv2_w18_coco_wholebody_hand_256x256_dark.py'
    # debug_hand_pose_config = 'external/mmlab/mmpose/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/coco_wholebody_hand/hrnetv2_w18_coco_wholebody_hand_256x256_dark.py'
    hand_pose_ckpt = 'https://download.openmmlab.com/mmpose/hand/dark/hrnetv2_w18_coco_wholebody_hand_256x256_dark-a9228c9c_20210908.pth'
    hand_pose_model = PoseModel(
        hand_pose_config, 
        hand_pose_ckpt, 
        rgb_keypoint_thres=tri_threshold, 
        rgb_keypoint_vis_thres=tri_threshold)

    # Create both aria and exo camera
    exo_cameras = {
        exo_camera_name: create_camera(dset[0][exo_camera_name]["camera_data"], None)
        for exo_camera_name in exo_cam_names
    }

    # Directory to store pose3d result and visualization
    pose3d_dir = os.path.join(ctx.dataset_dir, f'hand/pose3d')
    if not os.path.exists(pose3d_dir):
        os.makedirs(pose3d_dir)
    # Directory to store pose3d visualization
    if visualization:
        vis_pose3d_dir = os.path.join(ctx.dataset_dir, f'hand/vis_pose3d','exo_camera',f'triThresh={tri_threshold}')
        if not os.path.exists(vis_pose3d_dir):
            os.makedirs(vis_pose3d_dir)

    # Load hand pose2d keypoints from exo cameras
    exo_pose2d_file = os.path.join(ctx.dataset_dir, f'hand/pose2d', "exo_pose2d.pkl")
    assert os.path.exists(exo_pose2d_file), f"{exo_pose2d_file} does not exist"
    with open(exo_pose2d_file, "rb") as f:
        exo_poses2d = pickle.load(f)

    poses3d = {}
    for time_stamp in tqdm(range(len(dset)), total=len(dset)):
        info = dset[time_stamp]

        # # Pose2d estimation from exo camera
        # multi_view_pose2d = {
        #     exo_camera_name: exo_poses2d[time_stamp][exo_camera_name].reshape(-1,3)
        #     for exo_camera_name in exo_cam_names
        # }
        ########### Heuristic Check: Hardcode hand wrist kpt conf to be 1 ################################
        multi_view_pose2d = {}
        for exo_camera_name in exo_cam_names:
            curr_exo_hand_pose2d_kpts = exo_poses2d[time_stamp][exo_camera_name].reshape(-1,3)
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
        
        # triangulate
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

        # visualize pose3d
        if visualization:
            for camera_name in exo_cam_names:
                image_path = info[camera_name]["abs_frame_path"]
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
                hand_pose_model.draw_projected_poses3d([projected_pose3d[:21], projected_pose3d[21:]], image, save_path)

    with open(os.path.join(pose3d_dir, f"exo_pose3d_triThresh={tri_threshold}.pkl"), "wb") as f:
        pickle.dump(poses3d, f)



def mode_egoexo_hand_pose3d(config: Config):
    """
    Hand pose3d estimation with both ego and exo cameras
    """
    ctx = get_context(config)
    ########### Change as needed #############
    exo_cam_names = ctx.exo_cam_names # ctx.exo_cam_names  ['cam01','cam02']
    ego_cam_name = 'aria_rgb'
    tri_threshold = 0.3
    visualization = False
    ##########################################

    dset = SyncedEgoExoCaptureDset(
        data_dir=config.data_dir,
        dataset_json_path=ctx.dataset_json_path,
        read_frames=False,
        legacy=config.legacy
    )
    # Hand pose estimation model (same as hand_pose2d)
    hand_pose_config = 'ego4d/internal/human_pose/external/mmlab/mmpose/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/coco_wholebody_hand/hrnetv2_w18_coco_wholebody_hand_256x256_dark.py'
    hand_pose_ckpt = 'https://download.openmmlab.com/mmpose/hand/dark/hrnetv2_w18_coco_wholebody_hand_256x256_dark-a9228c9c_20210908.pth'
    # # Debug config file:
    # hand_pose_config = 'external/mmlab/mmpose/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/coco_wholebody_hand/hrnetv2_w18_coco_wholebody_hand_256x256_dark.py'
    # hand_pose_ckpt = 'https://download.openmmlab.com/mmpose/hand/dark/hrnetv2_w18_coco_wholebody_hand_256x256_dark-a9228c9c_20210908.pth'
    hand_pose_model = PoseModel(
        hand_pose_config, 
        hand_pose_ckpt, 
        rgb_keypoint_thres=tri_threshold, 
        rgb_keypoint_vis_thres=tri_threshold)

    # Create exo cameras
    aria_exo_cameras = {
        exo_camera_name: create_camera(dset[0][exo_camera_name]["camera_data"], None)
        for exo_camera_name in exo_cam_names
    }

    # Directory to store pose3d result and visualization
    pose3d_dir = os.path.join(ctx.dataset_dir, 'hand/pose3d')
    if not os.path.exists(pose3d_dir):
        os.makedirs(pose3d_dir)
    # Directory to store pose3d visualization
    if visualization:
        vis_pose3d_dir = os.path.join(ctx.dataset_dir, 'hand/vis_pose3d','ego_exo_camera', f'triThresh={tri_threshold}')
        if not os.path.exists(vis_pose3d_dir):
            os.makedirs(vis_pose3d_dir)

    # Load hand pose2d keypoints from both cam and aria
    cam_pose2d_file = os.path.join(ctx.dataset_dir, 'hand/pose2d', "exo_pose2d.pkl")
    assert os.path.exists(cam_pose2d_file), f"{cam_pose2d_file} does not exist"
    with open(cam_pose2d_file, "rb") as f:
        exo_poses2d = pickle.load(f)
    aria_pose2d_file = os.path.join(ctx.dataset_dir, 'hand/pose2d', "ego_pose2d.pkl")
    assert os.path.exists(aria_pose2d_file), f"{aria_pose2d_file} does not exist"
    with open(aria_pose2d_file, "rb") as f:
        aria_poses2d = pickle.load(f)

    poses3d = {}
    for time_stamp in tqdm(range(len(dset)), total=len(dset)):
        info = dset[time_stamp]

        ########### Heuristic Check: Hardcode hand wrist kpt conf to be 1 #############
        multi_view_pose2d = {}
        # Add exo camera keypoints
        for exo_camera_name in exo_cam_names:
            curr_exo_hand_pose2d_kpts = exo_poses2d[time_stamp][exo_camera_name].reshape(-1,3)
            if np.mean(curr_exo_hand_pose2d_kpts[:,-1]) > 0.3:
                curr_exo_hand_pose2d_kpts[[0,21],2] = 1
            multi_view_pose2d[exo_camera_name] = curr_exo_hand_pose2d_kpts
        # Add ego camera keypoints
        ego_hand_pose2d_kpts = aria_poses2d[time_stamp].reshape(-1,3)
        if np.mean(ego_hand_pose2d_kpts[:,-1]) > 0.3:
            ego_hand_pose2d_kpts[[0,21],2] = 1
        multi_view_pose2d[ego_cam_name] = ego_hand_pose2d_kpts
        ###############################################################################
        # Add ego camera
        aria_exo_cameras['aria_rgb'] = create_camera(info['aria_rgb']["camera_data"], None)

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
        
        # triangulate
        triangulator = Triangulator(
            time_stamp, 
            exo_cam_names+['aria_rgb'], 
            aria_exo_cameras, 
            multi_view_pose2d, 
            keypoint_thres=tri_threshold, 
            num_keypoints=42
        )
        pose3d = triangulator.run(debug=False)  ## N x 4 (x, y, z, confidence)
        poses3d[time_stamp] = pose3d

        # visualize pose3d
        if visualization:
            for camera_name in exo_cam_names + ['aria_rgb']:
                image_path = info[camera_name]["abs_frame_path"]
                image = cv2.imread(image_path)
                curr_camera = aria_exo_cameras[camera_name]

                vis_pose3d_cam_dir = os.path.join(vis_pose3d_dir, camera_name)
                if not os.path.exists(vis_pose3d_cam_dir):
                    os.makedirs(vis_pose3d_cam_dir)

                projected_pose3d = batch_xworld_to_yimage(pose3d[:, :3], curr_camera)
                projected_pose3d = np.concatenate(
                    [projected_pose3d, pose3d[:, 3].reshape(-1, 1)], axis=1
                )  ## N x 3 (17 for body,; 42 for hand)

                save_path = os.path.join(vis_pose3d_cam_dir, f"{time_stamp:05d}.jpg")
                hand_pose_model.draw_projected_poses3d([projected_pose3d[:21], projected_pose3d[21:]], image, save_path)

    with open(os.path.join(pose3d_dir, f"egoexo_pose3d_triThresh={tri_threshold}.pkl"), "wb") as f:
        pickle.dump(poses3d, f)



################# OLD_v2 ####################
# def mode_hand_pose3d_aria(config: Config):
#     """
#     Hand pose3d estimation with cam02, cam03, aria
#     """
#     ########### Change as needed #############
#     exo_cam_names = ['cam02','cam03']
#     ##########################################

#     ctx = get_context(config)
#     dset = SyncedEgoExoCaptureDset(
#         data_dir=config.data_dir,
#         dataset_json_path=ctx.dataset_json_path,
#         read_frames=False,
#     )
#     # Hand pose estimation model (same as hand_pose2d)
#     hand_pose_config = 'ego4d/internal/human_pose/external/mmlab/mmpose/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/coco_wholebody_hand/hrnetv2_w18_coco_wholebody_hand_256x256_dark.py'
#     hand_pose_ckpt = 'https://download.openmmlab.com/mmpose/hand/dark/hrnetv2_w18_coco_wholebody_hand_256x256_dark-a9228c9c_20210908.pth'
#     # Debug config file:
#     # hand_pose_config = 'external/mmlab/mmpose/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/coco_wholebody_hand/hrnetv2_w18_coco_wholebody_hand_256x256_dark.py'
#     # hand_pose_ckpt = 'https://download.openmmlab.com/mmpose/hand/dark/hrnetv2_w18_coco_wholebody_hand_256x256_dark-a9228c9c_20210908.pth'
#     hand_pose_model = PoseModel(
#         hand_pose_config, 
#         hand_pose_ckpt, 
#         rgb_keypoint_thres=0.3, 
#         rgb_keypoint_vis_thres=0.3)

#     # Create both aria and exo camera
#     aria_exo_cameras = {
#         exo_camera_name: create_camera(dset[0][exo_camera_name]["camera_data"], None)
#         for exo_camera_name in exo_cam_names
#     }

#     # DIrectory to store pose3d result and visualization
#     pose3d_dir = os.path.join(ctx.dataset_dir, 'hand/pose3d')
#     if not os.path.exists(pose3d_dir):
#         os.makedirs(pose3d_dir)
#     vis_pose3d_dir = os.path.join(ctx.dataset_dir, 'hand/vis_pose3d','cam0203Aria_thresh=0.3_noNormConfCheck')
#     if not os.path.exists(vis_pose3d_dir):
#         os.makedirs(vis_pose3d_dir)

#     # Load hand pose2d keypoints from both cam and aria
#     cam_pose2d_file = os.path.join(ctx.dataset_dir, 'hand/pose2d', "cam_pose2d.pkl")
#     assert os.path.exists(cam_pose2d_file), f"{cam_pose2d_file} does not exist"
#     with open(cam_pose2d_file, "rb") as f:
#         cam_poses2d = pickle.load(f)
#     aria_pose2d_file = os.path.join(ctx.dataset_dir, 'hand/pose2d', "aria_pose2d.pkl")
#     assert os.path.exists(aria_pose2d_file), f"{aria_pose2d_file} does not exist"
#     with open(aria_pose2d_file, "rb") as f:
#         aria_poses2d = pickle.load(f)

#     poses3d = {}
#     for time_stamp in tqdm(range(41,len(dset)), total=len(dset)):
#         info = dset[time_stamp]

#         # Pose2d estimation from exo camera
#         multi_view_pose2d = {
#             exo_camera_name: cam_poses2d[time_stamp][exo_camera_name].reshape(-1,3)
#             for exo_camera_name in exo_cam_names
#         }

#         # Append aria camera configuration and pose2d estimation result
#         multi_view_pose2d['aria_rgb'] = aria_rotate_kpts(aria_poses2d[time_stamp].reshape(-1,3),(1408,1408,3))
#         aria_exo_cameras['aria_rgb'] = create_camera(dset[time_stamp]['aria_rgb']["camera_data"], None)
        
#         # triangulate
#         triangulator = Triangulator(
#             time_stamp, 
#             exo_cam_names+['aria_rgb'], 
#             aria_exo_cameras, 
#             multi_view_pose2d, 
#             keypoint_thres=0.3, 
#             num_keypoints=42
#         )
#         pose3d = triangulator.run(debug=False)  ## N x 4 (x, y, z, confidence)
#         poses3d[time_stamp] = pose3d

#         # visualize pose3d
#         for camera_name in exo_cam_names + ['aria_rgb']:
#             image_path = info[camera_name]["abs_frame_path"]
#             image = cv2.imread(image_path)
#             curr_camera = aria_exo_cameras[camera_name]

#             vis_pose3d_cam_dir = os.path.join(vis_pose3d_dir, camera_name)
#             if not os.path.exists(vis_pose3d_cam_dir):
#                 os.makedirs(vis_pose3d_cam_dir)

#             projected_pose3d = batch_xworld_to_yimage(pose3d[:, :3], curr_camera)
#             projected_pose3d = np.concatenate(
#                 [projected_pose3d, pose3d[:, 3].reshape(-1, 1)], axis=1
#             )  ## N x 3 (17 for body,; 42 for hand)

#             save_path = os.path.join(vis_pose3d_cam_dir, f"{time_stamp:05d}.jpg")
#             hand_pose_model.draw_projected_poses3d([projected_pose3d[:21], projected_pose3d[21:]], image, save_path)

#     with open(os.path.join(pose3d_dir, "pose3d_cam0203Aria_thresh=0.3_noNormConfCheck.pkl"), "wb") as f:
#         pickle.dump(poses3d, f)



############# OLD ###############
# def mode_hand_pose3d_aria(config: Config):
#     """
#     Hand pose3d estimation with cam02, cam03, aria
#     """
#     ########### Change as needed #############
#     exo_cam_names = ['cam02','cam03']
#     ##########################################

#     ctx = get_context(config)
#     dset = SyncedEgoExoCaptureDset(
#         data_dir=config.data_dir,
#         dataset_json_path=ctx.dataset_json_path,
#         read_frames=False,
#     )
#     # Hand pose estimation model (same as hand_pose2d)
#     hand_pose_config = 'ego4d/internal/human_pose/external/mmlab/mmpose/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/coco_wholebody_hand/hrnetv2_w18_coco_wholebody_hand_256x256_dark.py'
#     hand_pose_ckpt = 'https://download.openmmlab.com/mmpose/hand/dark/hrnetv2_w18_coco_wholebody_hand_256x256_dark-a9228c9c_20210908.pth'
#     hand_pose_model = PoseModel(
#         hand_pose_config, 
#         hand_pose_ckpt, 
#         rgb_keypoint_thres=0.3, 
#         rgb_keypoint_vis_thres=0.3)

#     # Create both aria and exo camera
#     aria_exo_cameras = {
#         exo_camera_name: create_camera(dset[0][exo_camera_name]["camera_data"], None)
#         for exo_camera_name in exo_cam_names
#     }

#     # DIrectory to store pose3d result and visualization
#     pose3d_dir = os.path.join(ctx.dataset_dir, 'hand/pose3d')
#     if not os.path.exists(pose3d_dir):
#         os.makedirs(pose3d_dir)
#     vis_pose3d_dir = os.path.join(ctx.dataset_dir, 'hand/vis_pose3d','cam0203Aria_thresh=0.3_noNormDebug')
#     if not os.path.exists(vis_pose3d_dir):
#         os.makedirs(vis_pose3d_dir)

#     # Load hand pose2d keypoints from both cam and aria
#     cam_pose2d_file = os.path.join(ctx.dataset_dir, 'hand/pose2d', "cam_pose2d.pkl")
#     assert os.path.exists(cam_pose2d_file), f"{cam_pose2d_file} does not exist"
#     with open(cam_pose2d_file, "rb") as f:
#         cam_poses2d = pickle.load(f)
#     aria_pose2d_file = os.path.join(ctx.dataset_dir, 'hand/pose2d', "aria_pose2d.pkl")
#     assert os.path.exists(aria_pose2d_file), f"{aria_pose2d_file} does not exist"
#     with open(aria_pose2d_file, "rb") as f:
#         aria_poses2d = pickle.load(f)

#     poses3d = {}
#     for time_stamp in tqdm(range(len(dset)), total=len(dset)):
#         info = dset[time_stamp]

#         # Pose2d estimation from exo camera
#         multi_view_pose2d = {
#             exo_camera_name: cam_poses2d[time_stamp][exo_camera_name].reshape(-1,3)
#             for exo_camera_name in exo_cam_names
#         }

#         # Append aria camera configuration and pose2d estimation result
#         multi_view_pose2d['aria_rgb'] = aria_rotate_kpts(aria_poses2d[time_stamp].reshape(-1,3),(1408,1408,3))
#         aria_exo_cameras['aria_rgb'] = create_camera(dset[time_stamp]['aria_rgb']["camera_data"], None)
        
#         # triangulate
#         triangulator = Triangulator(
#             time_stamp, 
#             exo_cam_names+['aria_rgb'], 
#             aria_exo_cameras, 
#             multi_view_pose2d, 
#             keypoint_thres=0.3, 
#             num_keypoints=42
#         )
#         pose3d = triangulator.run(debug=False)  ## N x 4 (x, y, z, confidence)
#         poses3d[time_stamp] = pose3d

#         # visualize pose3d
#         for camera_name in exo_cam_names + ['aria_rgb']:
#             image_path = info[camera_name]["abs_frame_path"]
#             image = cv2.imread(image_path)
#             curr_camera = aria_exo_cameras[camera_name]

#             vis_pose3d_cam_dir = os.path.join(vis_pose3d_dir, camera_name)
#             if not os.path.exists(vis_pose3d_cam_dir):
#                 os.makedirs(vis_pose3d_cam_dir)

#             projected_pose3d = batch_xworld_to_yimage(pose3d[:, :3], curr_camera)
#             projected_pose3d = np.concatenate(
#                 [projected_pose3d, pose3d[:, 3].reshape(-1, 1)], axis=1
#             )  ## N x 3 (17 for body,; 42 for hand)

#             save_path = os.path.join(vis_pose3d_cam_dir, f"{time_stamp:05d}.jpg")
#             hand_pose_model.draw_projected_poses3d([projected_pose3d[:21], projected_pose3d[21:]], image, save_path)

#     with open(os.path.join(pose3d_dir, "pose3d_cam0203Aria_thresh=0.3_noNormDebug.pkl"), "wb") as f:
#         pickle.dump(poses3d, f)



# def mode_hand_pose3d_aria(config: Config):
#     """
#     Hand pose3d estimation with cam01, cam02, cam03, aria
#     """
#     ctx = get_context(config)

#     dset = SyncedEgoExoCaptureDset(
#         data_dir=config.data_dir,
#         dataset_json_path=ctx.dataset_json_path,
#         read_frames=False,
#     )

#     # Hand pose estimation model (same as hand_pose2d)
#     hand_pose_config = 'ego4d/internal/human_pose/external/mmlab/mmpose/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/coco_wholebody_hand/hrnetv2_w18_coco_wholebody_hand_256x256_dark.py'
#     hand_pose_ckpt = 'https://download.openmmlab.com/mmpose/hand/dark/hrnetv2_w18_coco_wholebody_hand_256x256_dark-a9228c9c_20210908.pth'
#     hand_pose_model = PoseModel(
#         hand_pose_config, 
#         hand_pose_ckpt, 
#         rgb_keypoint_thres=0.3, 
#         rgb_keypoint_vis_thres=0.3)

#     # Create both aria and exo camera
#     aria_exo_cameras = {
#         exo_camera_name: create_camera(dset[0][exo_camera_name]["camera_data"], None)
#         for exo_camera_name in ctx.exo_cam_names
#     }

#     # DIrectory to store pose3d result and visualization
#     pose3d_dir = os.path.join(ctx.dataset_dir, 'hand/pose3d')
#     if not os.path.exists(pose3d_dir):
#         os.makedirs(pose3d_dir)
#     vis_pose3d_dir = os.path.join(ctx.dataset_dir, 'hand/vis_pose3d','camAria')
#     if not os.path.exists(vis_pose3d_dir):
#         os.makedirs(vis_pose3d_dir)

#     # Load hand pose2d keypoints from both cam and aria
#     cam_pose2d_file = os.path.join(ctx.dataset_dir, 'hand/pose2d', "cam_pose2d.pkl")
#     assert os.path.exists(cam_pose2d_file), f"{cam_pose2d_file} does not exist"
#     with open(cam_pose2d_file, "rb") as f:
#         cam_poses2d = pickle.load(f)
#     aria_pose2d_file = os.path.join(ctx.dataset_dir, 'hand/pose2d', "aria_pose2d.pkl")
#     assert os.path.exists(aria_pose2d_file), f"{aria_pose2d_file} does not exist"
#     with open(aria_pose2d_file, "rb") as f:
#         aria_poses2d = pickle.load(f)

#     poses3d = {}
#     for time_stamp in tqdm(range(len(dset)), total=len(dset)):
#         info = dset[time_stamp]

#         # Pose2d estimation from exo camera
#         multi_view_pose2d = {
#             exo_camera_name: cam_poses2d[time_stamp][exo_camera_name].reshape(-1,3)
#             for exo_camera_name in ctx.exo_cam_names
#         }

#         # Append aria camera configuration and pose2d estimation result
#         multi_view_pose2d['aria_rgb'] = aria_rotate_kpts(aria_poses2d[time_stamp].reshape(-1,3),(1408,1408,3))
#         aria_exo_cameras['aria_rgb'] = create_camera(dset[time_stamp]['aria_rgb']["camera_data"], None)
        
#         # triangulate
#         triangulator = Triangulator(
#             time_stamp, 
#             ctx.exo_cam_names+['aria_rgb'], 
#             aria_exo_cameras, 
#             multi_view_pose2d, 
#             keypoint_thres=0.5, 
#             num_keypoints=42
#         )
#         pose3d = triangulator.run(debug=False)  ## N x 4 (x, y, z, confidence)
#         poses3d[time_stamp] = pose3d

#         # visualize pose3d
#         for camera_name in ctx.exo_cam_names + ['aria_rgb']:
#             image_path = info[camera_name]["abs_frame_path"]
#             image = cv2.imread(image_path)
#             curr_camera = aria_exo_cameras[camera_name]

#             vis_pose3d_cam_dir = os.path.join(vis_pose3d_dir, camera_name)
#             if not os.path.exists(vis_pose3d_cam_dir):
#                 os.makedirs(vis_pose3d_cam_dir)

#             projected_pose3d = batch_xworld_to_yimage(pose3d[:, :3], curr_camera)
#             projected_pose3d = np.concatenate(
#                 [projected_pose3d, pose3d[:, 3].reshape(-1, 1)], axis=1
#             )  ## N x 3 (17 for body,; 42 for hand)

#             save_path = os.path.join(vis_pose3d_cam_dir, f"{time_stamp:05d}.jpg")
#             hand_pose_model.draw_projected_poses3d([projected_pose3d[:21], projected_pose3d[21:]], image, save_path)

#     with open(os.path.join(pose3d_dir, "pose3d_camAria.pkl"), "wb") as f:
#         pickle.dump(poses3d, f)




def mode_multi_view_vis(config: Config):
    """
    Visualize pose3d results as concatenated multi-view videos
    """
    ########### Modify ##############
    camera_names = ['cam02','cam03','aria_rgb'] 
    #################################
    ctx = get_context(config)
    
    # Read and write directory
    read_dir = os.path.join(ctx.dataset_dir, 'hand/vis_pose3d','cam0203Aria_thresh=0.3')
    write_dir = os.path.join(ctx.dataset_dir, 'hand/vis_pose3d', "multi_view")
    os.makedirs(write_dir, exist_ok=True)
    vis_pose3d_dir = os.path.join(ctx.dataset_dir, 'hand/vis_pose3d')

    # Video parameters
    write_image_width = 3840
    write_image_height = 2160
    read_image_width = 3840
    read_image_height = 2160
    fps = 30
    padding = 5
    total_width_with_padding = 2 * read_image_width + padding
    total_height_with_padding = 2 * read_image_height + padding
    total_width = 2 * read_image_width
    total_height = 2 * read_image_height
    divide_val = 2

    # Collect all image path
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
    command = "rm -rf {}/exo.mp4".format(write_dir)
    os.system(command)

    command = (
        "ffmpeg -r {} -f image2 -i {}/%05d.jpg -pix_fmt yuv420p {}/exo.mp4".format(
            fps, write_dir, vis_pose3d_dir
        )
    )
    os.system(command)




# def mode_multi_view_vis(config: Config):
#     ctx = get_context(config)
#     camera_names = ctx.exo_cam_names

#     read_dir = ctx.vis_pose3d_dir
#     write_dir = os.path.join(ctx.vis_pose3d_dir, "multi_view")
#     os.makedirs(write_dir, exist_ok=True)

#     write_image_width = 3840
#     write_image_height = 2160

#     read_image_width = 3840
#     read_image_height = 2160

#     fps = 30
#     padding = 5

#     total_width_with_padding = 2 * read_image_width + padding
#     total_height_with_padding = 2 * read_image_height + padding

#     total_width = 2 * read_image_width
#     total_height = 2 * read_image_height
#     divide_val = 2

#     image_names = [
#         image_name
#         for image_name in sorted(os.listdir(os.path.join(read_dir, camera_names[0])))
#         if image_name.endswith(".jpg")
#     ]

#     for _t, image_name in enumerate(tqdm(image_names)):
#         canvas = 255 * np.ones((total_height_with_padding, total_width_with_padding, 3))

#         for idx, camera_name in enumerate(camera_names):
#             camera_image = cv2.imread(os.path.join(read_dir, camera_name, image_name))
#             camera_image = cv2.resize(
#                 camera_image, (read_image_width, read_image_height)
#             )

#             ##------------paste-----------------
#             col_idx = idx % divide_val
#             row_idx = idx // divide_val

#             origin_x = read_image_width * col_idx + col_idx * padding
#             origin_y = read_image_height * row_idx + row_idx * padding
#             image = camera_image

#             canvas[
#                 origin_y : origin_y + image.shape[0],
#                 origin_x : origin_x + image.shape[1],
#                 :,
#             ] = image[:, :, :]

#         # ---------resize to target size, ffmpeg does not work with offset image sizes---------
#         canvas = cv2.resize(canvas, (total_width, total_height))
#         canvas = cv2.resize(canvas, (write_image_width, write_image_height))

#         cv2.imwrite(os.path.join(write_dir, image_name), canvas)

#     # ----------make video--------------
#     command = "rm -rf {}/exo.mp4".format(write_dir)
#     os.system(command)

#     command = (
#         "ffmpeg -r {} -f image2 -i {}/%05d.jpg -pix_fmt yuv420p {}/exo.mp4".format(
#             fps, write_dir, ctx.vis_pose3d_dir
#         )
#     )
#     os.system(command)


def add_arguments(parser):
    parser.add_argument("--config-name", default="iu_music_jinxu")
    parser.add_argument(
        "--config_path", default="configs", help="Path to the config folder"
    )
    parser.add_argument(
        "--steps",
        default="",
        help="steps to run concatenated by '+', e.g., preprocess+bbox+pose2d+pose3d",
    )


def config_single_job(args, job_id):
    args.job_id = job_id
    args.name = args.name_list[job_id]
    args.work_dir = args.work_dir_list[job_id]
    args.output_dir = args.work_dir

    args.config_name = args.config_list[job_id]


def create_job_list(args):
    args.config_list = args.config_name.split("+")

    args.job_list = []
    args.name_list = []

    for config in args.config_list:
        name = args.name + "_" + config
        args.name_list.append(name)
        args.job_list.append(name)

    args.job_num = len(args.job_list)


def parse_args():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    args.steps = args.steps.split("+")
    print(args)

    return args


def get_hydra_config(args):
    # https://stackoverflow.com/questions/60674012/how-to-get-a-hydra-config-without-using-hydra-main
    hydra.initialize(config_path=args.config_path)
    cfg = hydra.compose(
        config_name=args.config_name,
        # args.opts contains config overrides, e.g., ["inputs.from_frame_number=7000",]
        # overrides=args.opts,
    )
    print("Final config:", cfg)
    return cfg


def main(args):
    # Note: this function is called from launch_train.py
    config = get_hydra_config(args)
    mode_preprocess(config)

#     if "preprocess" in args.steps:
#         mode_preprocess(config)
#     if "bbox" in args.steps:
#         mode_bbox(config)
#     if "pose2d" in args.steps:
#         mode_pose2d(config)
#     if "pose3d" in args.steps:
#         mode_pose3d(config)
#     if "multi_view_vis" in args.steps:
#         mode_multi_view_vis(config)


@hydra.main(config_path="configs", config_name=None, version_base=None)
def run(config: Config):
    if config.mode == "preprocess":
        mode_preprocess(config)
    elif config.mode == "body_bbox":
        """
        mode_body_bbox(config): Detect bbox with pretrained detector (NOTE:Make sure only one person in the frame)
        mode_bbox(config): Propose body bbox with aria position as heuristics
        """
        # mode_bbox(config)
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
    elif config.mode == 'hand_pose3d_egoexo':
        mode_egoexo_hand_pose3d(config)
    elif config.mode == "multi_view_vis":
        mode_multi_view_vis(config)
    elif config.mode == "show_all_config":
        ctx = get_context(config)
        print(ctx)
    else:
        raise AssertionError(f"unknown mode: {config.mode}")
    


if __name__ == "__main__":
    # Using hydra:
    run()

    # # Not using hydra:
    # args = parse_args()
    # main(args)