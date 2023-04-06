# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings

import mmcv
import numpy as np
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from PIL import Image

from mmpose.core.post_processing import oks_nms
from mmpose.datasets.dataset_info import DatasetInfo
from mmpose.datasets.pipelines import Compose
from mmpose.models import build_posenet
from mmpose.utils.hooks import OutputHook
from mmpose.core.visualization.visualize_hook import *

from .inference import _box2cs, _xyxy2xywh, _xywh2xyxy

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

##------------------------------------------------------------------------------------
def _inference_single_center_pose_model(model,
                                 img_or_path,
                                 bboxes,
                                 body_centers,
                                 dataset='TopDownCocoDataset',
                                 dataset_info=None,
                                 return_heatmap=False):
    """Inference human bounding boxes.

    Note:
        - num_bboxes: N
        - num_keypoints: K

    Args:
        model (nn.Module): The loaded pose model.
        img_or_path (str | np.ndarray): Image filename or loaded image.
        bboxes (list | np.ndarray): All bounding boxes (with scores),
            shaped (N, 4) or (N, 5). (left, top, width, height, [score])
            where N is number of bounding boxes.
        dataset (str): Dataset name. Deprecated.
        dataset_info (DatasetInfo): A class containing all dataset info.
        outputs (list[str] | tuple[str]): Names of layers whose output is
            to be returned, default: None

    Returns:
        ndarray[NxKx3]: Predicted pose x, y, score.
        heatmap[N, K, H, W]: Model output heatmap.
    """

    cfg = model.cfg
    device = next(model.parameters()).device
    if device.type == 'cpu':
        device = -1

    # build the data pipeline
    test_pipeline = Compose(cfg.test_pipeline)

    assert len(bboxes[0]) in [4, 5]

    if dataset_info is not None:
        dataset_name = dataset_info.dataset_name
        flip_pairs = dataset_info.flip_pairs
        if dataset in ('TopDownCocoDataset', 'TopDownOCHumanDataset',
                       'AnimalMacaqueDataset'):
            flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12],
                          [13, 14], [15, 16]]
        dataset_name = dataset

    batch_data = []
    for body_center, bbox in zip(body_centers, bboxes):
        center, scale = _box2cs(cfg, bbox)

        # prepare data
        data = {
            'center':
            center,
            'scale':
            scale,
            'bbox_score':
            bbox[4] if len(bbox) == 5 else 1,
            'bbox_id':
            0,  # need to be assigned if batch_size > 1
            'dataset':
            dataset_name,
            'joints_3d':
            np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
            'joints_3d_visible':
            np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
            'rotation':
            0,
            'body_center': body_center,
            'ann_info': {
                'image_size': np.array(cfg.data_cfg['image_size']),
                'num_joints': cfg.data_cfg['num_joints'],
                'flip_pairs': flip_pairs,
                'heatmap_size': np.array(cfg.data_cfg['heatmap_size']),
            }
        }
        if isinstance(img_or_path, np.ndarray):
            data['img'] = img_or_path
        else:
            data['image_file'] = img_or_path

        data = test_pipeline(data)
        batch_data.append(data)

    batch_data = collate(batch_data, samples_per_gpu=len(batch_data))
    batch_data = scatter(batch_data, [device])[0]

    # return_heatmap = True

    # forward the model
    with torch.no_grad():
        result = model(
            img=batch_data['img'],
            body_center_heatmap=batch_data['body_center_heatmap'],
            img_metas=batch_data['img_metas'],
            return_loss=False,
            return_heatmap=return_heatmap)

    ## --debugging--
    # prefix = os.path.join('.', 'val')
    # suffix = str(100).zfill(6)
    # image = batch_data['img']
    # output = result['output_heatmap']
    # body_center_heatmap=batch_data['body_center_heatmap']

    # original_image = batch_unnormalize_image(image) ## recolored image
    # save_batch_heatmaps(original_image, output, '{}_{}_hm_pred.jpg'.format(prefix, suffix), normalize=False, scale=4)

    # save_batch_image_with_joints_and_body_center(255*original_image, output, torch.ones(len(image), 1, 17), body_center_heatmap, '{}_{}_gt.jpg'.format(prefix, suffix))
    # save_batch_image_with_joints(255*original_image, output, torch.ones_like(target_weight), '{}_{}_pred.jpg'.format(prefix, suffix))

    return result['preds'], result['output_heatmap']

def inference_top_down_center_pose_model(model,
                                  img_or_path,
                                  person_results=None,
                                  bbox_thr=None,
                                  format='xywh',
                                  dataset='TopDownCocoDataset',
                                  dataset_info=None,
                                  return_heatmap=False,
                                  outputs=None):
    """Inference a single image with a list of person bounding boxes.

    Note:
        - num_people: P
        - num_keypoints: K
        - bbox height: H
        - bbox width: W

    Args:
        model (nn.Module): The loaded pose model.
        img_or_path (str| np.ndarray): Image filename or loaded image.
        person_results (list(dict), optional): a list of detected persons that
            contains ``bbox`` and/or ``track_id``:

            - ``bbox`` (4, ) or (5, ): The person bounding box, which contains
                4 box coordinates (and score).
            - ``track_id`` (int): The unique id for each human instance. If
                not provided, a dummy person result with a bbox covering
                the entire image will be used. Default: None.
        bbox_thr (float | None): Threshold for bounding boxes. Only bboxes
            with higher scores will be fed into the pose detector.
            If bbox_thr is None, all boxes will be used.
        format (str): bbox format ('xyxy' | 'xywh'). Default: 'xywh'.

            - `xyxy` means (left, top, right, bottom),
            - `xywh` means (left, top, width, height).
        dataset (str): Dataset name, e.g. 'TopDownCocoDataset'.
            It is deprecated. Please use dataset_info instead.
        dataset_info (DatasetInfo): A class containing all dataset info.
        return_heatmap (bool) : Flag to return heatmap, default: False
        outputs (list(str) | tuple(str)) : Names of layers whose outputs
            need to be returned. Default: None.

    Returns:
        tuple:
        - pose_results (list[dict]): The bbox & pose info. \
            Each item in the list is a dictionary, \
            containing the bbox: (left, top, right, bottom, [score]) \
            and the pose (ndarray[Kx3]): x, y, score.
        - returned_outputs (list[dict[np.ndarray[N, K, H, W] | \
            torch.Tensor[N, K, H, W]]]): \
            Output feature maps from layers specified in `outputs`. \
            Includes 'heatmap' if `return_heatmap` is True.
    """
    # get dataset info
    if (dataset_info is None and hasattr(model, 'cfg')
            and 'dataset_info' in model.cfg):
        dataset_info = DatasetInfo(model.cfg.dataset_info)
    if dataset_info is None:
        warnings.warn(
            'dataset is deprecated.'
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663'
            ' for details.', DeprecationWarning)

    # only two kinds of bbox format is supported.
    assert format in ['xyxy', 'xywh']

    pose_results = []
    returned_outputs = []

    if person_results is None:
        # create dummy person results
        if isinstance(img_or_path, str):
            width, height = Image.open(img_or_path).size
        else:
            height, width = img_or_path.shape[:2]
        person_results = [{'bbox': np.array([0, 0, width, height])}]

    if len(person_results) == 0:
        return pose_results, returned_outputs

    # Change for-loop preprocess each bbox to preprocess all bboxes at once.
    bboxes = np.array([box['bbox'] for box in person_results])
    body_centers = np.array([box['body_center'] for box in person_results])

    # Select bboxes by score threshold
    if bbox_thr is not None:
        assert bboxes.shape[1] == 5
        valid_idx = np.where(bboxes[:, 4] > bbox_thr)[0]
        bboxes = bboxes[valid_idx]
        body_centers = body_centers[valid_idx]
        person_results = [person_results[i] for i in valid_idx]

    if format == 'xyxy':
        bboxes_xyxy = bboxes
        bboxes_xywh = _xyxy2xywh(bboxes)
    else:
        # format is already 'xywh'
        bboxes_xywh = bboxes
        bboxes_xyxy = _xywh2xyxy(bboxes)

    # if bbox_thr remove all bounding box
    if len(bboxes_xywh) == 0:
        return [], []

    with OutputHook(model, outputs=outputs, as_tensor=False) as h:
        # poses is results['pred'] # N x 17x 3
        poses, heatmap = _inference_single_center_pose_model(
            model,
            img_or_path,
            bboxes_xywh,
            body_centers,
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap)

        if return_heatmap:
            h.layer_outputs['heatmap'] = heatmap

        returned_outputs.append(h.layer_outputs)

    assert len(poses) == len(person_results), print(
        len(poses), len(person_results), len(bboxes_xyxy))
    for pose, person_result, bbox_xyxy in zip(poses, person_results,
                                              bboxes_xyxy):
        pose_result = person_result.copy()
        pose_result['keypoints'] = pose
        pose_result['bbox'] = bbox_xyxy
        pose_results.append(pose_result)

    return pose_results, returned_outputs

