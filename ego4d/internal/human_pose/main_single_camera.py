import json
import os
import pickle
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List

import cv2

import hydra

import numpy as np
import pandas as pd
from ego4d.internal.colmap.preprocess import download_andor_generate_streams
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
from ego4d.internal.human_pose.readers import read_frame_idx_set

from ego4d.internal.human_pose.utils import (
    check_and_convert_bbox,
    draw_bbox_xyxy,
    draw_points_2d,
    get_exo_camera_plane,
    get_region_proposal,
)

from iopath.common.file_io import PathManager
from iopath.common.s3 import S3PathHandler
from tqdm.auto import tqdm

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


def get_context(config: Config) -> Context:
    metadata_json = json.load(pathmgr.open(config.inputs.metadata_json_path))
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
    )


##-------------------------------------------------------------------------------
def mode_pose2d(config: Config, camera_name: str):
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
    aria_path = (
        "/home/rawalk/Desktop/datasets/ego4d_data/cache/unc_T1/aria01/aria01.vrs"
    )
    assert not aria_path.startswith("https:") or not aria_path.startswith("s3:")
    # aria_camera_models = get_aria_camera_models(aria_path)

    # ---------------construct pose model-------------------
    from ego4d.internal.human_pose.pose_estimator import PoseModel

    pose_model = PoseModel(
        pose_config=ctx.pose_config, pose_checkpoint=ctx.pose_checkpoint
    )

    ##--------construct ground plane, it is parallel to the plane with all gopro camera centers----------------
    exo_cameras = {
        exo_camera_name: create_camera(dset[0][exo_camera_name]["camera_data"], None)
        for exo_camera_name in ctx.exo_cam_names
    }

    if not os.path.exists(ctx.pose2d_dir):
        os.makedirs(ctx.pose2d_dir)

    ## if ctx.vis_pose_dir does not exist make it
    if not os.path.exists(ctx.vis_pose2d_dir):
        os.makedirs(ctx.vis_pose2d_dir)

    poses2d = {}

    ## load bboxes from bbox_dir/bbox.pkl
    bbox_file = os.path.join(ctx.bbox_dir, "bbox_{}.pkl".format(camera_name))

    with open(bbox_file, "rb") as f:
        bboxes = pickle.load(f)

    for time_stamp in tqdm(range(len(dset)), total=len(dset)):
        info = dset[time_stamp]

        poses2d[time_stamp] = {}

        for exo_camera_name in [camera_name]:
            image_path = info[exo_camera_name]["abs_frame_path"]
            image = cv2.imread(image_path)

            vis_pose2d_cam_dir = os.path.join(ctx.vis_pose2d_dir, exo_camera_name)
            if not os.path.exists(vis_pose2d_cam_dir):
                os.makedirs(vis_pose2d_cam_dir)

            exo_camera = create_camera(info[exo_camera_name]["camera_data"], None)

            bbox_xyxy = bboxes[time_stamp][exo_camera_name]  ## x1, y1, x2, y2

            if bbox_xyxy is not None:
                ## add confidence score to the bbox
                bbox_xyxy = np.append(bbox_xyxy, 1.0)

                pose_results = pose_model.get_poses2d(
                    bboxes=[{"bbox": bbox_xyxy}],
                    image_name=image_path,
                )

                assert len(pose_results) == 1

                save_path = os.path.join(vis_pose2d_cam_dir, f"{time_stamp:05d}.jpg")
                pose_model.draw_poses2d(pose_results, image, save_path)

                pose_result = pose_results[0]
                pose2d = pose_result["keypoints"]

            else:
                pose2d = None
                save_path = os.path.join(vis_pose2d_cam_dir, f"{time_stamp:05d}.jpg")
                cv2.imwrite(save_path, image)

            poses2d[time_stamp][exo_camera_name] = pose2d

    ## save poses2d to pose2d_dir/pose2d_cam01.pkl
    with open(
        os.path.join(ctx.pose2d_dir, "pose2d_{}.pkl".format(camera_name)), "wb"
    ) as f:
        pickle.dump(poses2d, f)

    return


# ###-------------------------------------------------------------------------------
def mode_bbox(config: Config, camera_name: str):
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

    # ---------------construct bbox detector----------------
    from ego4d.internal.human_pose.bbox_detector import DetectorModel

    detector_model = DetectorModel(
        detector_config=ctx.detector_config,
        detector_checkpoint=ctx.detector_checkpoint,
        min_bbox_score=ctx.min_bbox_score,
    )

    ##--------construct ground plane, it is parallel to the plane with all gopro camera centers----------------
    exo_cameras = {
        exo_camera_name: create_camera(dset[0][exo_camera_name]["camera_data"], None)
        for exo_camera_name in ctx.exo_cam_names
    }
    exo_camera_centers = np.array(
        [exo_camera.center for exo_camera_name, exo_camera in exo_cameras.items()]
    )
    camera_plane, camera_plane_unit_normal = get_exo_camera_plane(exo_camera_centers)

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

        for exo_camera_name in [camera_name]:
            image_path = info[exo_camera_name]["abs_frame_path"]
            image = cv2.imread(image_path)

            vis_bbox_cam_dir = os.path.join(ctx.vis_bbox_dir, exo_camera_name)
            if not os.path.exists(vis_bbox_cam_dir):
                os.makedirs(vis_bbox_cam_dir)

            exo_camera = create_camera(info[exo_camera_name]["camera_data"], None)
            left_camera = create_camera(
                info["aria_slam_left"]["camera_data"], None
            )  ## TODO: use the camera model of the aria camera
            right_camera = create_camera(
                info["aria_slam_right"]["camera_data"], None
            )  ## TODO: use the camera model of the aria camera
            human_center_3d = (left_camera.center + right_camera.center) / 2

            proposal_points_3d = get_region_proposal(
                human_center_3d,
                radius=ctx.human_radius,
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

                ## uncomment to visualize the bounding box
                # bbox_image = draw_points_2d(image, proposal_points_2d, radius=5, color=(0, 255, 0))
                bbox_image = draw_bbox_xyxy(image, bbox_xyxy, color=(0, 255, 0))
            #
            else:
                bbox_image = image

            # bbox_image = draw_points_2d(image, proposal_points_2d, radius=5, color=(0, 255, 0))

            cv2.imwrite(
                os.path.join(vis_bbox_cam_dir, f"{time_stamp:05d}.jpg"), bbox_image
            )
            bboxes[time_stamp][exo_camera_name] = bbox_xyxy

    ## save the bboxes as a pickle file
    with open(os.path.join(ctx.bbox_dir, "bbox_{}.pkl".format(camera_name)), "wb") as f:
        pickle.dump(bboxes, f)

    return


@hydra.main(config_path="configs", config_name=None)
def run(config: Config):
    if config.mode == "bbox":
        mode_bbox(config, camera_name=config.exo_camera_name)
    elif config.mode == "pose2d":
        mode_pose2d(config, camera_name=config.exo_camera_name)
    else:
        raise AssertionError(f"unknown mode: {config.mode}")


if __name__ == "__main__":
    run()  # pyre-ignore
