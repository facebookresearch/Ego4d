# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np

from mmpose.core.post_processing import (affine_transform, fliplr_joints,
                                         get_affine_transform, get_warp_matrix,
                                         warp_affine_joints)
from mmpose.datasets.builder import PIPELINES

##--------------------------------------------------------------------------------------------------------------------##
COCO_JOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', \
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', \
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip', \
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

COCO_TORSO_JOINT_NAMES = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
COCO_TORSO_JOINT_INDEXES = [COCO_JOINT_NAMES.index(joint_name) for joint_name in COCO_TORSO_JOINT_NAMES]


##--------------------------------------------------------------------------------------------------------------------##

@PIPELINES.register_module()
class TopDownCenterAffine:
    """Affine transform the image to make input.

    Required keys:'img', 'joints_3d', 'joints_3d_visible', 'ann_info','scale',
    'rotation' and 'center'.

    Modified keys:'img', 'joints_3d', and 'joints_3d_visible'.

    Args:
        use_udp (bool): To use unbiased data processing.
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).
    """

    def __init__(self, use_udp=False):
        self.use_udp = use_udp

    def __call__(self, results):
        image_size = results['ann_info']['image_size']

        img = results['img']
        joints_3d = results['joints_3d']
        joints_3d_visible = results['joints_3d_visible']
        c = results['center']
        s = results['scale']
        r = results['rotation']

        if self.use_udp:
            trans = get_warp_matrix(r, c * 2.0, image_size - 1.0, s * 200.0)
            if not isinstance(img, list):
                img = cv2.warpAffine(
                    img,
                    trans, (int(image_size[0]), int(image_size[1])),
                    flags=cv2.INTER_LINEAR)
            else:
                img = [
                    cv2.warpAffine(
                        i,
                        trans, (int(image_size[0]), int(image_size[1])),
                        flags=cv2.INTER_LINEAR) for i in img
                ]

            joints_3d[:, 0:2] = \
                warp_affine_joints(joints_3d[:, 0:2].copy(), trans)

        else:
            trans = get_affine_transform(c, s, r, image_size)
            if not isinstance(img, list):
                img = cv2.warpAffine(
                    img,
                    trans, (int(image_size[0]), int(image_size[1])),
                    flags=cv2.INTER_LINEAR)
            else:
                img = [
                    cv2.warpAffine(
                        i,
                        trans, (int(image_size[0]), int(image_size[1])),
                        flags=cv2.INTER_LINEAR) for i in img
                ]
            for i in range(results['ann_info']['num_joints']):
                if joints_3d_visible[i, 0] > 0.0:
                    joints_3d[i,
                              0:2] = affine_transform(joints_3d[i, 0:2], trans)

        if 'body_center' in results.keys():
            body_center = results['body_center'].copy()
            body_center = affine_transform(body_center, trans)
            results['body_center'] = body_center

        results['img'] = img
        results['joints_3d'] = joints_3d
        results['joints_3d_visible'] = joints_3d_visible

        return results



@PIPELINES.register_module()
class TopDownCenter:
    """Compute the center of the person.
    """

    def __init__(self):
        return

    def __call__(self, results):
        image_size = results['ann_info']['image_size']

        img = results['img']
        joints_3d = results['joints_3d']
        joints_3d_visible = results['joints_3d_visible']
        c = results['center']
        s = results['scale']
        r = results['rotation']

        ## body center already present for either demo or test
        if 'body_center' in results.keys():
            return results

        is_valid = joints_3d_visible[:, 0] * \
                    (joints_3d[:, 0] >= 0) * (joints_3d[:, 0] < image_size[0]) * \
                    (joints_3d[:, 1] >= 0) * (joints_3d[:, 1] < image_size[1])

        is_valid = is_valid == 1

        joints_3d[~is_valid, :] = 0
        joints_3d_visible[~is_valid, :] = 0

        # in case all joints are invalid
        if is_valid.sum() == 0:
            body_center = np.array([image_size[0] / 2, image_size[1] / 2])
        else:
            body_center = joints_3d[is_valid, :2].mean(axis=0)

        # set body center using torso joints
        if is_valid[COCO_TORSO_JOINT_INDEXES].sum() >= 3:
            torso_center = joints_3d[COCO_TORSO_JOINT_INDEXES, :2][is_valid[COCO_TORSO_JOINT_INDEXES], :].mean(axis=0)
            body_center = torso_center

        results['body_center'] = body_center
        return results


@PIPELINES.register_module()
class TopDownGenerateBodyCenterTarget:
    """Generate the target heatmap.

    Required keys: 'joints_3d', 'joints_3d_visible', 'ann_info'.

    Modified keys: 'target', and 'target_weight'.

    Args:
        sigma: Sigma of heatmap gaussian for 'MSRA' approach.
        kernel: Kernel of heatmap gaussian for 'Megvii' approach.
        encoding (str): Approach to generate target heatmaps.
            Currently supported approaches: 'MSRA', 'Megvii', 'UDP'.
            Default:'MSRA'
        unbiased_encoding (bool): Option to use unbiased
            encoding methods.
            Paper ref: Zhang et al. Distribution-Aware Coordinate
            Representation for Human Pose Estimation (CVPR 2020).
        keypoint_pose_distance: Keypoint pose distance for UDP.
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).
        target_type (str): supported targets: 'GaussianHeatmap',
            'CombinedTarget'. Default:'GaussianHeatmap'
            CombinedTarget: The combination of classification target
            (response map) and regression target (offset map).
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).
    """

    def __init__(self,
                 sigma=2,
                 kernel=(11, 11),
                 valid_radius_factor=0.0546875,
                 target_type='GaussianHeatmap',
                 encoding='MSRA',
                 unbiased_encoding=False):
        self.sigma = sigma
        self.unbiased_encoding = unbiased_encoding
        self.kernel = kernel
        self.valid_radius_factor = valid_radius_factor
        self.target_type = target_type
        self.encoding = encoding
        return

    def _udp_generate_target(self, cfg, body_center, factor, target_type):
        """Generate the target heatmap via 'UDP' approach. Paper ref: Huang et
        al. The Devil is in the Details: Delving into Unbiased Data Processing
        for Human Pose Estimation (CVPR 2020).

        Note:
            - num keypoints: K
            - heatmap height: H
            - heatmap width: W
            - num target channels: C
            - C = K if target_type=='GaussianHeatmap'
            - C = 3*K if target_type=='CombinedTarget'

        Args:
            cfg (dict): data config
            joints_3d (np.ndarray[K, 3]): Annotated keypoints.
            joints_3d_visible (np.ndarray[K, 3]): Visibility of keypoints.
            factor (float): kernel factor for GaussianHeatmap target or
                valid radius factor for CombinedTarget.
            target_type (str): 'GaussianHeatmap' or 'CombinedTarget'.
                GaussianHeatmap: Heatmap target with gaussian distribution.
                CombinedTarget: The combination of classification target
                (response map) and regression target (offset map).

        Returns:
            tuple: A tuple containing targets.

            - target (np.ndarray[C, H, W]): Target heatmaps.
            - target_weight (np.ndarray[K, 1]): (1: visible, 0: invisible)
        """
        image_size = cfg['image_size']
        heatmap_size = cfg['heatmap_size']

        target = np.zeros((1, heatmap_size[1], heatmap_size[0]),
                            dtype=np.float32)

        tmp_size = factor * 3

        # prepare for gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, None]

        for joint_id in range(1):
            feat_stride = (image_size - 1.0) / (heatmap_size - 1.0)
            mu_x = int(body_center[0] / feat_stride[0] + 0.5)
            mu_y = int(body_center[1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                    or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                continue
            # # Generate gaussian
            mu_x_ac = body_center[0] / feat_stride[0]
            mu_y_ac = body_center[1] / feat_stride[1]
            x0 = y0 = size // 2
            x0 += mu_x_ac - mu_x
            y0 += mu_y_ac - mu_y
            g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * factor**2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], heatmap_size[1])
        
            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target

    def __call__(self, results):
        """Generate the target heatmap."""

        body_center = results['body_center']

        assert self.encoding == 'UDP'
        assert self.target_type == 'GaussianHeatmap'

        body_center_target = self._udp_generate_target(
            results['ann_info'], body_center, self.sigma,
            self.target_type)

        results['body_center_heatmap'] = body_center_target

        return results