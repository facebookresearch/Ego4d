import cv2
import numpy as np
import pandas as pd
import json
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
    bbox_3d = mesh.vertices  ## slower but smoother, all the vertices of the cylinder
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


def get_bbox_fromKpts(kpts, img_W, img_H, padding=50):
    # Get proposed hand bounding box from hand keypoints
    x1, y1, x2, y2 = kpts[:,0].min(), \
                     kpts[:,1].min(), \
                     kpts[:,0].max(), \
                     kpts[:,1].max()
    
    # Proposed hand bounding box with padding
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = np.clip(x1-padding,0,img_W-1), \
                                         np.clip(y1-padding,0,img_H-1), \
                                         np.clip(x2+padding,0,img_W-1), \
                                         np.clip(y2+padding,0,img_H-1)
    
    # Return bbox result
    return np.array([bbox_x1, bbox_y1, bbox_x2, bbox_y2])


def aria_extracted_to_original(kpts, img_shape=(1408,1408)):
    """
    Rotate kpts coordinates from extracted view (hand vertical) to original view (hand horizontal)
    img_shape is the shape of original view image
    """
    assert len(kpts.shape) == 2, "Only can rotate 2D arrays"
    H,W = img_shape
    new_kpts = kpts.copy()
    new_kpts[:,0] = kpts[:,1]
    new_kpts[:,1] = H - kpts[:,0]
    return new_kpts


def aria_original_to_extracted(kpts, img_shape=(1408,1408)):
    """
    Rotate kpts coordinates from original view (hand horizontal) to extracted view (hand vertical)
    img_shape is the shape of original view image
    """
    # assert len(kpts.shape) == 2, "Only can rotate 2D arrays"
    H,W = img_shape
    new_kpts = kpts.copy()
    new_kpts[:,0] = H - kpts[:,1]
    new_kpts[:,1] = kpts[:,0]
    return new_kpts


def aria_slamOriginal_to_extracted(kpts, img_shape=(1408,1408)):
    """
    Rotate kpts coordinates from original view (hand horizontal) to extracted view (hand vertical)
    img_shape is the shape of original view image
    """
    # assert len(kpts.shape) == 2, "Only can rotate 2D arrays"
    H,W = img_shape
    new_kpts = kpts.copy()
    new_kpts[:,0] = H - kpts[:,1] + 80
    new_kpts[:,1] = kpts[:,0]
    return new_kpts


def get_two_small_bboxes(bboxes):
    """
    Since the hand bbox detection might contain multiple results where one bbox capture both hands, while we want each bbox
    to contain only one hand, we select most two small bboxes from multiple results.
    """
    bbox_area = (bboxes[:,2] - bboxes[:,0]) * (bboxes[:,3] - bboxes[:,1])
    min_idx = np.argsort(bbox_area)[:2]
    return bboxes[min_idx,:4]


def get_largest_bboxes(bboxes):
    """
    If there are multiple persons detected, then assume the largest one is the subject of interest. Used in 0630_Cooking_4 as heuristics
    """
    bbox_area = (bboxes[:,2] - bboxes[:,0]) * (bboxes[:,3] - bboxes[:,1])
    max_idx = np.argmax(bbox_area)
    return bboxes[max_idx, :4]



def left_right_bboxes_div(bboxes):
    """
    Since the bbox for hand detection in aria view is from pretrained hand detector that doesn't provide left/right hand label,
    we apply some heuristics to separate left/right bboxes from their center location. Bbox whose center has smaller x-coor is 
    considered as left hand bbox.
    Return:
        -  (2,4) where first one is right hand bbox and second is left hand's bbox
    """
    # Extract x-coor of bbox center
    assert len(bboxes) >= 2, "Less than two bboxes detected. Might need manual correction"
    bbox_1, bbox_2 = bboxes[0], bboxes[1]
    bbox_1_xc, bbox_2_xc = (bbox_1[2] + bbox_1[0]) / 2, (bbox_2[2] + bbox_2[0]) / 2
    
    # Check left/right hand bbox
    if bbox_1_xc < bbox_2_xc:
        return np.array([bbox_2,bbox_1])
    elif bbox_2_xc < bbox_1_xc:
        return np.array([bbox_1, bbox_2])
    else:
        raise Exception("Two hand's bbox cannot be the same.")
    

def extri_intri_from_json(extr_dir, intri_dir, time_stamp=None, ego=False):
    """
    For self-collected data (aprt from Meta's processed data), return camera extrinsic and 
    instrinsic from calibration JSON file
    """
    # Load extrinisc from json file
    if not ego:
        extri_raw = json.load(open(extr_dir))
        extrinsics = np.array(extri_raw['T_camera_world'])
    else:
        extri_raw = json.load(open(extr_dir))
        # index extrinsics with time_stamp
        curr_key = sorted(list(extri_raw.keys()))[time_stamp]
        extrinsics = np.array(extri_raw[curr_key]['T_camera_world'])
    
    # Load intrinsic from json file 
    intri_raw = json.load(open(intri_dir))
    intrinsics = np.zeros(8)
    intrinsics[0], intrinsics[1] = intri_raw['f'], intri_raw['f'] # f_x,f_y
    intrinsics[2], intrinsics[3] = float(intri_raw['width'])/2 + float(intri_raw['cx']), float(intri_raw['height'])/2 + float(intri_raw['cy']) # c_x,c_y
    intrinsics[4],intrinsics[5],intrinsics[6] = intri_raw['k1'], intri_raw['k2'], intri_raw['k3'] # k_1,k_2,k_3,k_4

    return extrinsics, intrinsics
