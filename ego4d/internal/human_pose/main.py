import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass

import cv2

import hydra
import pandas as pd
import torch
from ego4d.internal.colmap.preprocess import download_andor_generate_streams
from ego4d.internal.human_pose.camera import (
    create_camera,
    create_camera_data,
    get_aria_camera_models,
    xworld_to_yimage,
)

from ego4d.internal.human_pose.config import Config
from ego4d.internal.human_pose.dataset import (
    get_synced_timesync_df,
    SyncedEgoExoCaptureDset,
)
from ego4d.internal.human_pose.readers import read_frame_idx_set

from iopath.common.file_io import PathManager
from iopath.common.s3 import S3PathHandler
from mmpose.apis import inference_top_down_pose_model, init_pose_model
from tqdm.auto import tqdm
import numpy as np
import trimesh
from pyntcloud.geometry.models.plane import Plane
from pyntcloud import PyntCloud

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
    )


def mode_preprocess(config: Config):
    """
    Does the following:
    - extract all frames for all cameras
    """
    ctx = get_context(config)
    by_dev_id = download_andor_generate_streams(
        metadata=ctx.metadata_json,
        download_video_files=config.mode_preprocess.download_video_files,
        force_download=config.mode_preprocess.force_download,
        output_dir=ctx.cache_dir,
    )

    # shutil.rmtree(ctx.frame_dir, ignore_errors=True)
    os.makedirs(ctx.frame_dir, exist_ok=True)
    exo_camera_centers = np.array([exo_camera.center for exo_camera_name, exo_camera in exo_cameras.items()])
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
    for cam in ["cam01", "cam02", "cam03", "cam04"]:
        cam_frame_dir = os.path.join(ctx.frame_dir, cam)
        os.makedirs(cam_frame_dir, exist_ok=True)
        frame_indices = [int(x) for x in synced_df[f"{cam}_frame_number"].iloc[i1:i2+1].tolist()]
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

        for cam_id in ["cam01", "cam02", "cam03", "cam04"]:
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


###-------------------------------------------------------------------------------
"""
region_proposal of the human using aria camera location in 3D
We fit a fixed height cylinder to the aria camera location
The orientation of the cylinder is computed using the exo camera plane
"""
def get_region_proposal(point_3d, unit_normal, radius=0.3, human_height=1.1, padding=1.1, axis=[0, 1, 0]):
    transform = trimesh.transformations.rotation_matrix(np.deg2rad(180), axis) ## angle and axis

    ## compute the height of the human
    cylinder_center = point_3d - human_height*unit_normal*0.5
    transform[:3, 3] = cylinder_center ## place the cylinder at the hip of the human

    mesh = trimesh.primitives.Cylinder(radius=radius, height=human_height)
    mesh.apply_transform(transform)
    bbox_3d = mesh.vertices ## slower but smoother, all the vertices of the cylinder
    return bbox_3d

"""
return the ground plane and the unit normal of the ground plane
"""
def get_exo_camera_plane(points):
    point_cloud = PyntCloud(pd.DataFrame(data=points, columns=["x", "y", "z"]))
    plane = Plane()
    plane.from_point_cloud(point_cloud.xyz)
    a, b, c, d = plane.get_equation()
    unit_normal = np.array([a, b, c])/np.sqrt(a*a + b*b + c*c)
    return plane, unit_normal

def mode_bounding_box_detection(config: Config):
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

    ##--------construct ground plane, it is parallel to the plane with all gopro camera centers----------------
    exo_cameras = {exo_camera_name: create_camera(dset[0][exo_camera_name]["camera_data"], None) for exo_camera_name in ["cam01", "cam02", "cam03", "cam04"]}
    exo_camera_centers = np.array([exo_camera.center for exo_camera_name, exo_camera in exo_cameras.items()])
    camera_plane, camera_plane_unit_normal  = get_exo_camera_plane(exo_camera_centers)

    for time_stamp in tqdm(range(len(dset)), total=len(dset)):
        info = dset[time_stamp]

        for exo_camera_name in ["cam01", "cam02", "cam03", "cam04"]:
            exo_camera = camera_for_data(info[exo_camera_name]["camera_data"], None)
            left_camera = camera_for_data(info["aria_slam_left"]["camera_data"], None)
            right_camera = camera_for_data(info["aria_slam_right"]["camera_data"], None)
            human_center_3d = (left_camera.center + right_camera.center) / 2

            region_proposal_points = get_region_proposal(human_center_3d, unit_normal=camera_plane_unit_normal)

            import pdb; pdb.set_trace()

    # TODO perform object detection with FasterRCNN ?
    # TODO match on person classes using `camera_wearer_centers` ?

###-------------------------------------------------------------------------------

def mode_pose_keypoint_detection(config: Config):
    ctx = get_context(config)
    dset = SyncedEgoExoCaptureDset(
        root_dir=config.root_dir,
        dataset_json_path=ctx.dataset_json_path,
        read_frames=False,
    )

    pose_model = init_pose_model(
        config.mode_pose_estimation.pose_model_config,
        config.mode_pose_estimation.pose_model_checkpoint,
        device=f"cuda:{config.gpu_id}" if config.gpu_id > 0 else "cpu",
    )

    images = [
        (
            dset[i]["cam01"]["abs_frame_path"],
            dset[i]["cam01"]["frame_path"],
        )
        for i in range(len(dset))
    ]
    # TODO: fixme
    bbox = [300, 300, 500, 500]
    persons = [
        {
            "bbox": bbox + [1.0],
            "track_id": 0,
        }
    ]
    images = images[0:1]

    # NOTE
    # we cannot provide a batch of inputs as a model config requires
    # "frame_weight_test" which is available in a PoseTrack trained model
    # but this model was trained on COCO-WholeBody
    result = []
    for person, (abs_image_path, rel_path) in tqdm(
        zip(persons, images), total=len(images)
    ):
        pose = inference_top_down_pose_model(
            pose_model,
            abs_image_path,
            person_results=persons,
            format="xyxy",
            return_heatmap=False,
            bbox_thr=0.9,  # TODO add argument
            outputs=None,
        )
        result.append(
            {
                "person": person,
                "path": rel_path,
                "pose": pose,  # TODO: fixme
            }
        )

    torch.save(result, os.path.join(ctx.dataset_dir, "keypoints.pth"))


def mode_triangulation(config: Config):
    # NOTE
    # feel free to write the implementation of this code to another file and
    # call it here as a function
    pass


@hydra.main(config_path="configs", config_name=None)
def run(config: Config):
    if config.mode == "preprocess":
        mode_preprocess(config)
    elif config.mode == "bbox":
        mode_bounding_box_detection(config)
    elif config.mode == "pose_estimation":
        mode_pose_keypoint_detection(config)
    elif config.mode == "triangulate":
        mode_triangulation(config)
    else:
        raise AssertionError(f"unknown mode: {config.mode}")


if __name__ == "__main__":
    run()  # pyre-ignore
