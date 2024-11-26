import json

import cv2
import numpy as np
import pandas as pd
import trimesh
from pyntcloud import PyntCloud
from pyntcloud.geometry.models.plane import Plane


COCO_KP_ORDER = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

COCO_SKELETON = {
    "left_leg": [13, 15],  ## l-knee to l-ankle
    "right_leg": [14, 16],  ## r-knee to r-ankle
    "left_thigh": [11, 13],  ## l-hip to l-knee
    "right_thigh": [12, 14],  ## r-hip to r-knee
    "hip": [11, 12],  ## l-hip to r-hip
    "left_torso": [5, 11],  ## l-shldr to l-hip
    "right_torso": [6, 12],  ## r-shldr to r-hip
    "left_bicep": [5, 7],  ## l-shldr to l-elbow
    "right_bicep": [6, 8],  ## r-shldr to r-elbow
    "shoulder": [5, 6],  ## l-shldr to r-shldr
    "left_hand": [7, 9],  ## l-elbow to l-wrist
    "right_hand": [8, 10],  ## r-elbow to r-wrist
    "left_face": [1, 0],  ## l-eye to nose
    "right_face": [2, 0],  ## l-eye to nose
    "face": [1, 2],  ## l-eye to r-eye
    "left_ear": [1, 3],  ## l-eye to l-ear
    "right_ear": [2, 4],  ## l-eye to r-ear
    "left_neck": [3, 5],  ## l-ear to l-shldr
    "right_neck": [4, 6],  ## r-ear to r-shldr
}

COCO_SKELETON_FLIP_PAIRS = {
    "leg": ("left_leg", "right_leg"),
    "thigh": ("left_thigh", "right_thigh"),
    "torso": ("left_torso", "right_torso"),
    "bicep": ("left_bicep", "right_bicep"),
    "hand": ("left_hand", "right_hand"),
    "face": ("left_face", "right_face"),
    "ear": ("left_ear", "right_ear"),
    "neck": ("left_neck", "right_neck"),
}


def check_and_convert_bbox(
    bbox_2d_all,
    image_width,
    image_height,
    min_area_ratio=0.005,
    max_aspect_ratio_thres=5.0,
    min_aspect_ratio_thres=0.5,
):
    is_valid = (
        (bbox_2d_all[:, 0] >= 0)
        * (bbox_2d_all[:, 0] <= image_width)
        * (bbox_2d_all[:, 1] >= 0)
        * (bbox_2d_all[:, 1] <= image_height)
    )

    bbox_2d_all = bbox_2d_all[is_valid]

    ## out of frame
    if len(bbox_2d_all) == 0:
        return None

    x1 = bbox_2d_all[:, 0].min()
    x2 = bbox_2d_all[:, 0].max()
    y1 = bbox_2d_all[:, 1].min()
    y2 = bbox_2d_all[:, 1].max()

    bbox_width = x2 - x1
    bbox_height = y2 - y1
    area = bbox_width * bbox_height

    image_area = image_width * image_height
    bbox_area_ratio = area * 1.0 / image_area

    ## if bbox is too small
    if bbox_area_ratio < min_area_ratio:
        return None

    aspect_ratio = bbox_height / bbox_width  ## height/width

    ## the bbox is too skewed, height is large compared to width
    if aspect_ratio > max_aspect_ratio_thres or aspect_ratio < min_aspect_ratio_thres:
        return None

    bbox_2d_xyxy = np.array(
        [round(x1), round(y1), round(x2), round(y2)]
    )  ## convert to integers

    return bbox_2d_xyxy


def draw_points_2d(image, points_2d, color=(255, 0, 0), radius=2):
    for point_2d in points_2d:
        image = cv2.circle(
            image, (int(point_2d[0]), int(point_2d[1])), radius, color, -1
        )

    return image


def draw_bbox_xyxy(image, bbox_xyxy, color=(255, 0, 0), thickness=4):
    x1, y1, x2, y2 = bbox_xyxy
    image = cv2.rectangle(
        image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness
    )
    return image


def get_region_proposal(
    point_3d, unit_normal, radius=0.3, human_height=1.8, axis=[0, 1, 0]
):
    """
    region_proposal of the human using aria camera location in 3D
    We fit a fixed height cylinder to the aria camera location
    The orientation of the cylinder is computed using the exo camera plane
    """
    transform = trimesh.transformations.rotation_matrix(
        np.deg2rad(180), axis
    )  ## angle and axis

    ## compute the height of the human
    cylinder_center = point_3d - human_height * unit_normal * 0.5
    transform[:3, 3] = cylinder_center  ## place the cylinder at the hip of the human

    mesh = trimesh.primitives.Cylinder(radius=radius, height=human_height)
    mesh.apply_transform(transform)

    # Note: using all vertices of the cylinder (bbox_3d = mesh.vertices) is not enough
    # since the bottom could be easily out of the image boundary and
    # the proposal filter would shrink to just the top of the cylinder
    # Therefore, we do a uniform sampling around the whole cylinder instead
    bbox_3d, _face_index = trimesh.sample.sample_surface_even(mesh, count=100)

    return bbox_3d


def get_exo_camera_plane(points):
    """
    return the ground plane and the unit normal of the ground plane
    """
    point_cloud = PyntCloud(pd.DataFrame(data=points, columns=["x", "y", "z"]))
    plane = Plane()
    plane.from_point_cloud(point_cloud.xyz)
    a, b, c, d = plane.get_equation()
    unit_normal = np.array([a, b, c]) / np.sqrt(a * a + b * b + c * c)
    return plane, unit_normal


def get_bbox_from_kpts(kpts, img_W, img_H, padding=50):
    # Get proposed hand bounding box from hand keypoints
    x1, y1, x2, y2 = (
        kpts[:, 0].min(),
        kpts[:, 1].min(),
        kpts[:, 0].max(),
        kpts[:, 1].max(),
    )

    # Proposed hand bounding box with padding
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = (
        np.clip(x1 - padding, 0, img_W - 1),
        np.clip(y1 - padding, 0, img_H - 1),
        np.clip(x2 + padding, 0, img_W - 1),
        np.clip(y2 + padding, 0, img_H - 1),
    )

    # Return bbox result
    return np.array([bbox_x1, bbox_y1, bbox_x2, bbox_y2])


def aria_extracted_to_original(kpts, img_shape=(1408, 1408)):
    """
    Rotate kpts coordinates from extracted view (hand vertical) to original view (hand horizontal)
    img_shape is the shape of original view image
    """
    assert len(kpts.shape) == 2, "Only can rotate 2D arrays"
    H, W = img_shape
    new_kpts = kpts.copy()
    new_kpts[:, 0] = kpts[:, 1]
    new_kpts[:, 1] = H - kpts[:, 0]
    return new_kpts


def aria_original_to_extracted(kpts, img_shape=(1408, 1408)):
    """
    Rotate kpts coordinates from original view (hand horizontal) to extracted view (hand vertical)
    img_shape is the shape of original view image
    """
    # assert len(kpts.shape) == 2, "Only can rotate 2D arrays"
    H, W = img_shape
    new_kpts = kpts.copy()
    new_kpts[:, 0] = H - kpts[:, 1]
    new_kpts[:, 1] = kpts[:, 0]
    return new_kpts


def compute_hand_pose3d_joint_angles(hand_pose3d):
    """
    Compute the joint angles from pose3d estimation result
    """
    # Joint index of interest
    joint_index = [
        1,
        2,
        3,
        5,
        6,
        7,
        9,
        10,
        11,
        13,
        14,
        15,
        17,
        18,
        19,
        22,
        23,
        24,
        26,
        27,
        28,
        30,
        31,
        32,
        34,
        35,
        36,
        38,
        39,
        40,
    ]
    wrist_angle_index = [1, 5, 9, 13, 17, 22, 26, 30, 34, 38]

    # Exclude invalid joints (with zero confidence)
    hand_pose3d[hand_pose3d[:, -1] == 0] = None

    # Compute joint angles
    joint_angles = []
    for joint_idx in joint_index:
        # If current joint has pose3d estimation
        if joint_idx in wrist_angle_index:
            if joint_idx < 21:
                vec1 = hand_pose3d[0, :3] - hand_pose3d[joint_idx, :3]
            else:
                vec1 = hand_pose3d[21, :3] - hand_pose3d[joint_idx, :3]
        else:
            vec1 = hand_pose3d[joint_idx - 1, :3] - hand_pose3d[joint_idx, :3]
        vec2 = hand_pose3d[joint_idx + 1, :3] - hand_pose3d[joint_idx, :3]
        # Compute angle
        v1_u = vec1 / np.linalg.norm(vec1)
        v2_u = vec2 / np.linalg.norm(vec2)
        angles = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) * 180.0 / np.pi
        joint_angles.append(angles)

    return np.array(joint_angles)


def whether_use_selector(joint_angles, pose3d, num_threshold):
    """
    Compute the number of joint angle outliers and invalid joints; return True if it
    is higher than the max limit and False otherwise.
    """
    # Compute number of joint angle outliers
    joint_angle_min_threshold = np.array(
        [
            100,
            90,
            90,
            60,
            70,
            80,
            60,
            70,
            80,
            60,
            70,
            80,
            60,
            70,
            80,
        ]
    )
    right_joint_angles, left_joint_angles = joint_angles[:15], joint_angles[15:]
    # All joints can reach maximum 180 degrees so only check for minimum threshold
    right_joint_angle_outliers, left_joint_angle_outliers = (
        np.sum(right_joint_angles < joint_angle_min_threshold),
        np.sum(left_joint_angles < joint_angle_min_threshold),
    )

    # Compute number of invalid joints
    right_invalid_count, left_invalid_count = (
        np.sum(pose3d[:21, -1] == 0),
        np.sum(pose3d[21:, -1] == 0),
    )

    # Determine whether use selector for both hands
    right_selector_flag = (
        right_joint_angle_outliers + right_invalid_count
    ) > num_threshold
    left_selector_flag = (
        left_joint_angle_outliers + left_invalid_count
    ) > num_threshold
    return right_selector_flag, left_selector_flag


def wholebody_hand_selector(pose3d, wholebody_hand_poses3d, num_threshold=5):
    """
    Check estimated hand pose3d based on two standards:
        1. Number of joint angle outliers
        2. Number of invalid joints
    If the total count of those two standards exceeds the limit
    then assign wholebody-hand pose3d as the final pose3d results.
    Lower the value of num_threshold to make the selector more easily to be triggered.
    """
    # Compute joint angles
    pred_pose3d = pose3d.copy()
    joint_angles = compute_hand_pose3d_joint_angles(pred_pose3d)

    # Determine whether use wholebody-Hand pose3d based on two standards
    right_selector_flag, left_selector_flag = whether_use_selector(
        joint_angles, pose3d, num_threshold
    )

    # Assign wholebody-Hand pose3d if necessary
    if right_selector_flag:
        pose3d[:21, :] = wholebody_hand_poses3d[21:, :]
    if left_selector_flag:
        pose3d[21:, :] = wholebody_hand_poses3d[:21, :]
    return pose3d


def normalize_reprojection_error(reproj_error, bboxes, skel_type):
    """
    Normalize the reprojection error to account for the scale effect across different views
    Args:
        reproj_error: Dict of (N,) ~ Dict of (# of views,) ~ (42,1)
        bbox: Dict of (N,) ~ Dict of (# of views,) ~ List of (2,)
    """
    new_reproj_error = {}
    for ts in reproj_error.keys():
        new_reproj_error[ts] = {}
        for cam in reproj_error[ts].keys():
            # Get current camera view's reprojection error and bboxes
            curr_reproj_error = reproj_error[ts][cam].flatten()
            # Get index of invalid reprojection error (-1)
            invalid_reproj_index = np.argwhere(curr_reproj_error == -1).flatten()

            # Normalize reprojection error
            if skel_type == "hand":
                curr_right_bbox, curr_left_bbox = bboxes[ts][cam]
                # If bbox is None then don't normalize reprojection error
                if curr_right_bbox is not None:
                    # Caculate left and right hand's bbox area
                    right_hand_bbox_area = (curr_right_bbox[2] - curr_right_bbox[0]) * (
                        curr_right_bbox[3] - curr_right_bbox[1]
                    )
                    # Normalize reprojection error with corresponding hand bbox
                    curr_reproj_error[:21] /= (
                        right_hand_bbox_area
                        if right_hand_bbox_area != 0
                        else curr_reproj_error[:21]
                    )
                if curr_left_bbox is not None:
                    left_hand_bbox_area = (curr_left_bbox[2] - curr_left_bbox[0]) * (
                        curr_left_bbox[3] - curr_left_bbox[1]
                    )
                    curr_reproj_error[21:] /= (
                        left_hand_bbox_area
                        if left_hand_bbox_area != 0
                        else curr_reproj_error[:21]
                    )
            elif skel_type == "body":
                # Calculate body bbox area
                curr_body_bbox = bboxes[ts][cam]

                if curr_body_bbox is not None:
                    body_bbox_area = (curr_body_bbox[2] - curr_body_bbox[0]) * (
                        curr_body_bbox[3] - curr_body_bbox[1]
                    )
                    # Normalize body reprojection error with body bbox
                    curr_reproj_error /= (
                        body_bbox_area if body_bbox_area != 0 else curr_reproj_error
                    )
            else:
                raise Exception(
                    f"Unknown skeleton type: {skel_type}. Valid skel_type: [body, hand]."
                )

            # Re-assign invalid reprojection error
            curr_reproj_error[invalid_reproj_index] = -1
            # Re-assign reprojection error
            new_reproj_error[ts][cam] = curr_reproj_error

    return new_reproj_error
