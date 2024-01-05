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

def get_aria_distortion_params(sensor_name, focal_length, size):    
    # create output calibration: a linear model of image size 512x512 and focal length 150
    # Invisible pixels are shown as black.
    dst_calib = calibration.get_linear_camera_calibration(
        size, size, focal_length, sensor_name
    )    
    return (        
        dst_calib.get_principal_point(),
        dst_calib.get_focal_lengths(),
    )


def get_aria_intrinsics():    
    principal_points, focal_lengths = get_aria_distortion_params("camera-rgb", 150, 512)        
    # forming a intrinsic matrix
    instrinsic_list = [
        [focal_lengths[0], 0.0, principal_points[0]],
        [0.0, focal_lengths[1], principal_points[1]],
        [0.0, 0.0, 1.0],
    ]
    return instrinsic_list
   

def get_aria_extrinsics(aria_data):    
    T_device_world = aria_data["camera_data"]["T_device_world"]
    T_camera_device = aria_data["camera_data"]["T_camera_device"]
    T_camera_world = np.matmul(T_camera_device, T_device_world)
    extrinsic_list = T_camera_world.tolist()
    extrinsic_list = extrinsic_list[:-1]
    return extrinsic_list

def undistort_exocam(intrinsics, distortion_coeffs, dimension=(3840, 2160)):
    DIM = dimension
    dim2 = None
    dim3 = None
    balance = 0.8
    # Load the distortion parameters
    distortion_coeffs = distortion_coeffs
    # Load the camera intrinsic parameters
    intrinsics = intrinsics

    #HACKY FIX FOR NOW without loading all the images
    dim1 = DIM

    #TODO Deal with vertical gopros
    '''
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    dim1 = image.shape[:2][::-1]  # dim1 is the dimension of input image to un-distort

    # Change the calibration dim dynamically
    # (e.g., bouldering cam01 and cam04 are verticall)
    if DIM[0] != dim1[0]:
        DIM = (DIM[1], DIM[0])
    '''

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
    '''
    undistorted_image = cv2.remap(
        image,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    '''
    
    return new_K


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


def process_exocam_data(exocam_data):    
    # undistort and save images
    distortion_coeffs, intrinsics = get_distortion_and_intrinsics(exocam_data)
    
    # getting update instrisic matrix
    new_K = undistort_exocam(intrinsics, distortion_coeffs, (3840, 2160))
    intrinsic_list = new_K.tolist()
    
    # getting extrinsic matrix    
    extrinsic_list = exocam_data["camera_data"]["T_device_world"] 

    processed_cam_data = dict()
    processed_cam_data["camera_intrinsics"] = intrinsic_list
    processed_cam_data["camera_extrinsics"] = extrinsic_list[:-1]
    processed_cam_data["distortion_coeffs"] = distortion_coeffs.tolist()
    return processed_cam_data