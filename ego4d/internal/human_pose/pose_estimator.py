import os

import cv2
import numpy as np
from mmdet.apis import inference_detector, init_detector

from mmpose.apis import (
    inference_top_down_pose_model,
    init_pose_model,
    process_mmdet_results,
    vis_pose_result,
)

from mmpose.core.bbox.transforms import (
    bbox_cs2xywh,
    bbox_xywh2cs,
    bbox_xywh2xyxy,
    bbox_xyxy2xywh,
)
from mmpose.datasets import DatasetInfo
from tqdm import tqdm


##------------------------------------------------------------------------------------
class PoseModel:
    def __init__(
        self,
        pose_config=None,
        pose_checkpoint=None,
        rgb_keypoint_thres=0.7,
        rgb_keypoint_vis_thres=0.7,
        refine_bbox=True,
        thickness=6,
    ):
        self.pose_config = pose_config
        self.pose_checkpoint = pose_checkpoint

        ## load pose model
        self.pose_model = init_pose_model(
            self.pose_config, self.pose_checkpoint, device="cuda:0".lower()
        )
        self.dataset = self.pose_model.cfg.data["test"]["type"]
        self.dataset_info = DatasetInfo(
            self.pose_model.cfg.data["test"].get("dataset_info", None)
        )
        self.return_heatmap = False
        self.output_layer_names = None
        self.num_keypoints = len(
            self.pose_model.cfg.dataset_info["keypoint_info"].keys()
        )
        self.coco_17_keypoints_idxs = np.array(
            range(17)
        )  ## indexes of the COCO keypoints
        self.coco_17_sigmas = (
            np.array(
                [
                    0.26,
                    0.25,
                    0.25,
                    0.35,
                    0.35,
                    0.79,
                    0.79,
                    0.72,
                    0.72,
                    0.62,
                    0.62,
                    1.07,
                    1.07,
                    0.87,
                    0.87,
                    0.89,
                    0.89,
                ]
            )
            / 10.0
        )
        self.refine_bbox = refine_bbox

        ##------hyperparameters-----
        self.bbox_thres = 0.1  ## Bounding box score threshold
        self.rgb_keypoint_thres = rgb_keypoint_thres  ## Keypoint score threshold
        self.rgb_keypoint_vis_thres = (
            rgb_keypoint_vis_thres  ## Keypoint score threshold
        )

        ## Keypoint radius for visualization
        self.radius = 4

        ## Link thickness for visualization
        self.thickness = thickness

        self.min_vis_keypoints = 5  ## coco format, 17 keypoints!

    ####--------------------------------------------------------
    def get_poses2d(self, bboxes, image_name):
        pose_results, returned_outputs = inference_top_down_pose_model(
            self.pose_model,
            image_name,
            bboxes,
            bbox_thr=self.bbox_thres,
            format="xyxy",
            dataset=self.dataset,
            dataset_info=self.dataset_info,
            return_heatmap=self.return_heatmap,
            outputs=self.output_layer_names,
        )

        ##---------refine the bboxes-------------------
        if self.refine_bbox:
            pose_results = self.refine_bboxes(pose_results)

        if len(pose_results) < len(bboxes):
            for bbox in bboxes:
                pose_result = bbox.copy()
                pose_result["keypoints"] = np.zeros(
                    (self.num_keypoints, 3)
                )  ## dummy pose
                pose_results.append(pose_result)

        return pose_results

    ####--------------------------------------------------------
    def refine_bboxes(self, pose_results, padding=1.2, aspect_ratio=3 / 4):
        valid_pose_results = []

        for i in range(len(pose_results)):
            bbox = pose_results[i]["bbox"]
            pose = pose_results[i]["keypoints"]

            is_valid = pose[:, 2] > self.rgb_keypoint_thres

            if is_valid.sum() == 0:
                continue

            x1 = pose[is_valid, 0].min()
            x2 = pose[is_valid, 0].max()
            y1 = pose[is_valid, 1].min()
            y2 = pose[is_valid, 1].max()

            bbox_xyxy = np.array([[x1, y1, x2, y2]])

            # https://github.com/open-mmlab/mmpose/blob/master/mmpose/core/bbox/transforms.py
            bbox_center, bbox_scale = bbox_xywh2cs(
                bbox=bbox_xyxy2xywh(bbox_xyxy).reshape(-1),
                aspect_ratio=aspect_ratio,
                padding=padding,
            )  ## aspect_ratio is w/h

            bbox_xywh = bbox_cs2xywh(center=bbox_center, scale=bbox_scale, padding=1.0)

            refined_bbox = bbox_xywh2xyxy(bbox_xywh.reshape(1, 4)).reshape(
                -1
            )  ## (4,) ## tight fitting bbox to the detected pose

            ## if exo camera, completely replace the bbox
            pose_results[i]["bbox"][:4] = refined_bbox.astype(int)
            valid_pose_results.append(pose_results[i])

        return valid_pose_results

    ####--------------------------------------------------------
    def draw_poses2d(self, pose_results, image_name, save_path):
        keypoint_thres = self.rgb_keypoint_vis_thres
        bbox_colors = [(0, 255, 0)]

        vis_pose_result(
            self.pose_model,
            image_name,
            pose_results,
            dataset=self.dataset,
            dataset_info=self.dataset_info,
            kpt_score_thr=keypoint_thres,
            radius=self.radius,
            thickness=self.thickness,
            bbox_color=bbox_colors,
            bbox_thickness=self.thickness,
            show=False,
            out_file=save_path,
        )

    ####--------------------------------------------------------
    def draw_projected_poses3d(self, pose_results, image_name, save_path):
        keypoint_thres = self.rgb_keypoint_vis_thres

        ##-----------restructure to the desired format used by mmpose---------
        pose_results_ = []
        for pose in pose_results:
            pose_ = np.zeros(
                (self.num_keypoints, 3)
            )  ## N x 3 (17 for body; 21 for hand)

            pose_[: len(pose), :3] = pose[:, :]

            pose_result = {"keypoints": pose_}
            pose_results_.append(pose_result)

        pose_results = pose_results_

        vis_pose_result(
            self.pose_model,
            image_name,
            pose_results,
            dataset=self.dataset,
            dataset_info=self.dataset_info,
            kpt_score_thr=keypoint_thres,
            radius=self.radius,
            thickness=self.thickness,
            show=False,
            out_file=save_path,
        )
