import json
import os

import cv2
import numpy as np
from PIL import Image

from projectaria_tools.core import calibration

body_keypoints_list = [
    {"label": "Nose", "id": "fee3cbd2", "color": "#f77189"},
    {"label": "Left-eye", "id": "ab12de34", "color": "#d58c32"},
    {"label": "Right-eye", "id": "7f2g1h6k", "color": "#a4a031"},
    {"label": "Left-ear", "id": "mn0pqrst", "color": "#50b131"},
    {"label": "Right-ear", "id": "yz89wx76", "color": "#34ae91"},
    {"label": "Left-shoulder", "id": "5a4b3c2d", "color": "#37abb5"},
    {"label": "Right-shoulder", "id": "e1f2g3h4", "color": "#3ba3ec"},
    {"label": "Left-elbow", "id": "6i7j8k9l", "color": "#bb83f4"},
    {"label": "Right-elbow", "id": "uv0wxy12", "color": "#f564d4"},
    {"label": "Left-wrist", "id": "3z4ab5cd", "color": "#2fd4aa"},
    {"label": "Right-wrist", "id": "efgh6789", "color": "#94d14f"},
    {"label": "Left-hip", "id": "ijklmnop", "color": "#b3d32c"},
    {"label": "Right-hip", "id": "qrstuvwx", "color": "#f9b530"},
    {"label": "Left-knee", "id": "yz012345", "color": "#83f483"},
    {"label": "Right-knee", "id": "6bc7defg", "color": "#32d58c"},
    {"label": "Left-ankle", "id": "hijk8lmn", "color": "#3ba3ec"},
    {"label": "Right-ankle", "id": "opqrs1tu", "color": "#f564d4"},
]

hand_keypoints_list = [
    {"label": "Right_Wrist", "id": "fee3cbd2", "color": "#f77189"},
    {"label": "Right_Thumb_1", "id": "yz012345", "color": "#83f483"},
    {"label": "Right_Thumb_2", "id": "6bc7defg", "color": "#32d58c"},
    {"label": "Right_Thumb_3", "id": "hijk8lmn", "color": "#3ba3ec"},
    {"label": "Right_Thumb_4", "id": "opqrs1tu", "color": "#f564d4"},
    {"label": "Right_Index_1", "id": "ab12de34", "color": "#d58c32"},
    {"label": "Right_Index_2", "id": "7f2g1h6k", "color": "#a4a031"},
    {"label": "Right_Index_3", "id": "mn0pqrst", "color": "#50b131"},
    {"label": "Right_Index_4", "id": "9vwxyzab", "color": "#32d58c"},
    {"label": "Right_Middle_1", "id": "yz89wx76", "color": "#34ae91"},
    {"label": "Right_Middle_2", "id": "5a4b3c2d", "color": "#37abb5"},
    {"label": "Right_Middle_3", "id": "e1f2g3h4", "color": "#3ba3ec"},
    {"label": "Right_Middle_4", "id": "cdefgh23", "color": "#3ba3ec"},
    {"label": "Right_Ring_1", "id": "efgh6789", "color": "#94d14f"},
    {"label": "Right_Ring_2", "id": "ijklmnop", "color": "#b3d32c"},
    {"label": "Right_Ring_3", "id": "qrstuvwx", "color": "#f9b530"},
    {"label": "Right_Ring_4", "id": "ijkl4567", "color": "#bb83f4"},
    {"label": "Right_Pinky_1", "id": "6i7j8k9l", "color": "#bb83f4"},
    {"label": "Right_Pinky_2", "id": "uv0wxy12", "color": "#f564d4"},
    {"label": "Right_Pinky_3", "id": "3z4ab5cd", "color": "#2fd4aa"},
    {"label": "Right_Pinky_4", "id": "mnop8qrs", "color": "#f564d4"},
    {"label": "Left_Wrist", "id": "fee3cbd2_left", "color": "#f77189"},
    {"label": "Left_Thumb_1", "id": "yz012345_left", "color": "#83f483"},
    {"label": "Left_Thumb_2", "id": "6bc7defg_left", "color": "#32d58c"},
    {"label": "Left_Thumb_3", "id": "hijk8lmn_left", "color": "#3ba3ec"},
    {"label": "Left_Thumb_4", "id": "opqrs1tu_left", "color": "#f564d4"},
    {"label": "Left_Index_1", "id": "ab12de34_left", "color": "#d58c32"},
    {"label": "Left_Index_2", "id": "7f2g1h6k_left", "color": "#a4a031"},
    {"label": "Left_Index_3", "id": "mn0pqrst_left", "color": "#50b131"},
    {"label": "Left_Index_4", "id": "9vwxyzab_left", "color": "#32d58c"},
    {"label": "Left_Middle_1", "id": "yz89wx76_left", "color": "#34ae91"},
    {"label": "Left_Middle_2", "id": "5a4b3c2d_left", "color": "#37abb5"},
    {"label": "Left_Middle_3", "id": "e1f2g3h4_left", "color": "#3ba3ec"},
    {"label": "Left_Middle_4", "id": "cdefgh23_left", "color": "#3ba3ec"},
    {"label": "Left_Ring_1", "id": "efgh6789_left", "color": "#94d14f"},
    {"label": "Left_Ring_2", "id": "ijklmnop_left", "color": "#b3d32c"},
    {"label": "Left_Ring_3", "id": "qrstuvwx_left", "color": "#f9b530"},
    {"label": "Left_Ring_4", "id": "ijkl4567_left", "color": "#bb83f4"},
    {"label": "Left_Pinky_1", "id": "6i7j8k9l_left", "color": "#bb83f4"},
    {"label": "Left_Pinky_2", "id": "uv0wxy12_left", "color": "#f564d4"},
    {"label": "Left_Pinky_3", "id": "3z4ab5cd_left", "color": "#2fd4aa"},
    {"label": "Left_Pinky_4", "id": "mnop8qrs_left", "color": "#f564d4"},
]


def get_default_attachment():
    return [
        {
            "type": "JSON",
            "role": "SUPPLEMENT",
            "payload": [
                {
                    "data": {
                        "capture_id": "cam_name",
                        "camera_intrinsics": [
                            [0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0],
                        ],
                        "camera_extrinsics": [
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ],
                        "camera_id": "cam_id",
                        "frame_number": 0,
                        "layout_row": 1,
                        "layout_col": 7,
                    }
                }
            ],
        }
    ]


def undistort_aria(image_array, provider, sensor_name, focal_length, size):
    device_calib = provider.get_device_calibration()
    src_calib = device_calib.get_camera_calib(sensor_name)

    # create output calibration: a linear model of image size 512x512 and focal length 150
    # Invisible pixels are shown as black.
    dst_calib = calibration.get_linear_camera_calibration(
        size, size, focal_length, sensor_name
    )

    # distort image
    rectified_array = calibration.distort_by_calibration(
        image_array, dst_calib, src_calib
    )
    return (
        rectified_array,
        dst_calib.get_principal_point(),
        dst_calib.get_focal_lengths(),
    )


def write_attachment(
    default_attachment_json,
    camera_id,
    frame_number,
    intrinsic_list,
    extrinsic_list,
    capture_id,
    pose3d,
    id_to_halo_map,
    frame_name,
    high_conf_frame_list,
):
    default_attachment_json[0]["payload"][0]["data"]["capture_id"] = capture_id
    default_attachment_json[0]["payload"][0]["data"]["camera_id"] = camera_id
    default_attachment_json[0]["payload"][0]["data"]["frame_number"] = frame_number
    default_attachment_json[0]["payload"][0]["data"]["camera_intrinsics"] = (
        intrinsic_list
    )
    default_attachment_json[0]["payload"][0]["data"]["camera_extrinsics"] = (
        extrinsic_list[:-1]
    )

    T_extrinsic = np.array(extrinsic_list)
    T_intrinsic = np.zeros((3, 4))
    T_intrinsic[:, :3] = np.array(intrinsic_list)

    T_in_x_ex = np.matmul(T_intrinsic, T_extrinsic)

    projected_pose = np.matmul(T_in_x_ex, pose3d.T).T

    assert len(default_attachment_json) == 1
    keypoints_3d = {}
    keypoints_2d = {}

    for i in range(len(pose3d)):
        halo_id = id_to_halo_map[i]

        if projected_pose[i][2] > 0.001:
            keypoints_3d[halo_id] = {
                "x": pose3d[i][0],
                "y": pose3d[i][1],
                "z": pose3d[i][2],
                "id": halo_id,
                "mediaIDs": high_conf_frame_list[i],
            }
            keypoints_2d[halo_id] = {
                "x": projected_pose[i][0] / projected_pose[i][2],
                "y": projected_pose[i][1] / projected_pose[i][2],
                "id": halo_id,
                "mediaID": frame_name,
                "placement": (
                    "manual" if frame_name in high_conf_frame_list[i] else "auto"
                ),
            }

    gt_payload = {
        "type": "JSON",
        "role": "PREDEFINED",
        "payload": [{"data": {"keypoints3D": keypoints_3d, "keypoints": keypoints_2d}}],
    }
    default_attachment_json.append(gt_payload)

    return default_attachment_json


def process_aria_data(
    aria_name,
    aria_data,
    default_attachment_json,
    frames_folder,
    provider,
    capture_id,
    pose3d,
    id_to_halo_map,
    high_conf_frame_list,
    output_images_dir,
    output_attachments_dir,
):
    frame_path = aria_data["frame_path"]
    frame_path = os.path.join(frames_folder, frame_path)
    save_name = ("_").join(frame_path.split("/")[-2:])

    # undistort and save images
    img = Image.open(frame_path)
    #############################################################
    # Caution: rotating back the image before doing undistortion!
    img = img.rotate(90)
    #############################################################
    image_array = np.asarray(img)
    rectified_array, principal_points, focal_lengths = undistort_aria(
        image_array, provider, "camera-rgb", 150, 512
    )
    img = Image.fromarray(rectified_array, "RGB")
    img.save(os.path.join(output_images_dir, save_name))

    # getting extrinsic matrix
    frame_number = aria_data["frame_number"]
    T_device_world = aria_data["camera_data"]["T_device_world"]
    T_camera_device = aria_data["camera_data"]["T_camera_device"]
    T_camera_world = np.matmul(T_camera_device, T_device_world)

    # forming a intrinsic matrix
    instrinsic_list = [
        [focal_lengths[0], 0.0, principal_points[0]],
        [0.0, focal_lengths[1], principal_points[1]],
        [0.0, 0.0, 1.0],
    ]
    # save attachment
    frame_name = save_name.split(".")[0]
    default_attachment_json = write_attachment(
        default_attachment_json,
        aria_name,
        frame_number,
        instrinsic_list,
        T_camera_world.tolist(),
        capture_id,
        pose3d,
        id_to_halo_map,
        frame_name,
        high_conf_frame_list,
    )
    with open(
        os.path.join(output_attachments_dir, save_name.replace(".jpg", ".json")), "w"
    ) as f:
        json.dump(default_attachment_json, f, indent=2)


def undistort_exocam(image_path, intrinsics, distortion_coeffs, dimension=(3840, 2160)):
    DIM = dimension
    dim2 = None
    dim3 = None
    balance = 0.8
    # Load the distortion parameters
    distortion_coeffs = distortion_coeffs
    # Load the camera intrinsic parameters
    intrinsics = intrinsics

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    dim1 = image.shape[:2][::-1]  # dim1 is the dimension of input image to un-distort

    # Change the calibration dim dynamically
    # (e.g., bouldering cam01 and cam04 are verticall)
    if DIM[0] != dim1[0]:
        DIM = (DIM[1], DIM[0])

    assert (
        dim1[0] / dim1[1] == DIM[0] / DIM[1]
    ), "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1
    scaled_K = (
        intrinsics * dim1[0] / DIM[0]
    )  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0

    # This is how scaled_K, dim2 and balance are used to
    # determine the final K used to un-distort image.
    # OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        scaled_K, distortion_coeffs, dim2, np.eye(3), balance=balance
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        scaled_K, distortion_coeffs, np.eye(3), new_K, dim3, cv2.CV_16SC2
    )
    undistorted_image = cv2.remap(
        image,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    # map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    #    intrinsics, distortion_coeffs, np.eye(3), intrinsics, DIM, cv2.CV_16SC2)
    # undistorted_image = cv2.remap(
    #    image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_image, new_K


def get_distortion_and_intrinsics(exocam_data):
    cam = exocam_data
    _raw_camera = cam["_raw_camera"]
    intrinsics = np.array(
        [
            [_raw_camera["intrinsics_0"], 0, _raw_camera["intrinsics_2"]],
            [0, _raw_camera["intrinsics_1"], _raw_camera["intrinsics_3"]],
            [0, 0, 1],
        ]
    )
    distortion_coeffs = np.array(
        [
            _raw_camera["intrinsics_4"],
            _raw_camera["intrinsics_5"],
            _raw_camera["intrinsics_6"],
            _raw_camera["intrinsics_7"],
        ]
    )
    return distortion_coeffs, intrinsics


def process_exocam_data(
    exocam_name,
    exocam_data,
    default_attachment_json,
    frames_folder,
    capture_id,
    pose3d,
    id_to_halo_map,
    high_conf_frame_list,
    output_images_dir,
    output_attachments_dir,
):
    frame_path = exocam_data["frame_path"]
    frame_path = os.path.join(frames_folder, frame_path)
    save_name = ("_").join(frame_path.split("/")[-2:])

    # undistort and save images
    distortion_coeffs, intrinsics = get_distortion_and_intrinsics(exocam_data)
    undistorted_image, new_K = undistort_exocam(
        frame_path, intrinsics, distortion_coeffs, (3840, 2160)
    )
    img = Image.fromarray(undistorted_image, "RGB")
    img.save(os.path.join(output_images_dir, save_name))

    # getting extrinsic matrix
    frame_number = exocam_data["frame_number"]
    T_device_world = exocam_data["camera_data"]["T_device_world"]

    # getting instrisic matrix
    new_K = new_K.tolist()

    # save attachment
    frame_name = save_name.split(".")[0]

    default_attachment_json = write_attachment(
        default_attachment_json,
        exocam_name,
        frame_number,
        new_K,
        T_device_world,
        capture_id,
        pose3d,
        id_to_halo_map,
        frame_name,
        high_conf_frame_list,
    )
    with open(
        os.path.join(output_attachments_dir, save_name.replace(".jpg", ".json")), "w"
    ) as f:
        json.dump(default_attachment_json, f, indent=2)
