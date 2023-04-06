# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from mmpose.core import (keypoint_auc, keypoint_epe, keypoint_pck_accuracy,
                         keypoints_from_heatmaps, keypoints_from_heatmaps3d,
                         multilabel_classification_accuracy, pose_pck_accuracy)


def test_pose_pck_accuracy():
    output = np.zeros((1, 5, 64, 64), dtype=np.float32)
    target = np.zeros((1, 5, 64, 64), dtype=np.float32)
    mask = np.array([[True, True, False, False, False]])
    # first channel
    output[0, 0, 20, 20] = 1
    target[0, 0, 10, 10] = 1
    # second channel
    output[0, 1, 30, 30] = 1
    target[0, 1, 30, 30] = 1

    acc, avg_acc, cnt = pose_pck_accuracy(output, target, mask)

    assert_array_almost_equal(acc, np.array([0, 1, -1, -1, -1]), decimal=4)
    assert abs(avg_acc - 0.5) < 1e-4
    assert abs(cnt - 2) < 1e-4


def test_keypoints_from_heatmaps():
    heatmaps = np.ones((1, 1, 64, 64), dtype=np.float32)
    heatmaps[0, 0, 31, 31] = 2
    center = np.array([[127, 127]])
    scale = np.array([[64 / 200.0, 64 / 200.0]])

    udp_heatmaps = np.ones((32, 17, 64, 64), dtype=np.float32)
    udp_heatmaps[:, :, 31, 31] = 2
    udp_center = np.tile([127, 127], (32, 1))
    udp_scale = np.tile([32, 32], (32, 1))

    preds, maxvals = keypoints_from_heatmaps(heatmaps, center, scale)

    assert_array_almost_equal(preds, np.array([[[126, 126]]]), decimal=4)
    assert_array_almost_equal(maxvals, np.array([[[2]]]), decimal=4)
    assert isinstance(preds, np.ndarray)
    assert isinstance(maxvals, np.ndarray)

    with pytest.raises(AssertionError):
        # kernel should > 0
        _ = keypoints_from_heatmaps(
            heatmaps, center, scale, post_process='unbiased', kernel=0)

    preds, maxvals = keypoints_from_heatmaps(
        heatmaps, center, scale, post_process='unbiased')
    assert_array_almost_equal(preds, np.array([[[126, 126]]]), decimal=4)
    assert_array_almost_equal(maxvals, np.array([[[2]]]), decimal=4)
    assert isinstance(preds, np.ndarray)
    assert isinstance(maxvals, np.ndarray)

    # test for udp dimension problem
    preds, maxvals = keypoints_from_heatmaps(
        udp_heatmaps,
        udp_center,
        udp_scale,
        post_process='default',
        target_type='GaussianHeatMap',
        use_udp=True)
    assert_array_almost_equal(preds, np.tile([76, 76], (32, 17, 1)), decimal=0)
    assert_array_almost_equal(maxvals, np.tile([2], (32, 17, 1)), decimal=4)
    assert isinstance(preds, np.ndarray)
    assert isinstance(maxvals, np.ndarray)

    preds1, maxvals1 = keypoints_from_heatmaps(
        heatmaps,
        center,
        scale,
        post_process='default',
        target_type='GaussianHeatMap',
        use_udp=True)
    preds2, maxvals2 = keypoints_from_heatmaps(
        heatmaps,
        center,
        scale,
        post_process='default',
        target_type='GaussianHeatmap',
        use_udp=True)
    assert_array_almost_equal(preds1, preds2, decimal=4)
    assert_array_almost_equal(maxvals1, maxvals2, decimal=4)
    assert isinstance(preds2, np.ndarray)
    assert isinstance(maxvals2, np.ndarray)


def test_keypoint_pck_accuracy():
    output = np.zeros((2, 5, 2))
    target = np.zeros((2, 5, 2))
    mask = np.array([[True, True, False, True, True],
                     [True, True, False, True, True]])
    thr = np.full((2, 2), 10, dtype=np.float32)
    # first channel
    output[0, 0] = [10, 0]
    target[0, 0] = [10, 0]
    # second channel
    output[0, 1] = [20, 20]
    target[0, 1] = [10, 10]
    # third channel
    output[0, 2] = [0, 0]
    target[0, 2] = [-1, 0]
    # fourth channel
    output[0, 3] = [30, 30]
    target[0, 3] = [30, 30]
    # fifth channel
    output[0, 4] = [0, 10]
    target[0, 4] = [0, 10]

    acc, avg_acc, cnt = keypoint_pck_accuracy(output, target, mask, 0.5, thr)

    assert_array_almost_equal(acc, np.array([1, 0.5, -1, 1, 1]), decimal=4)
    assert abs(avg_acc - 0.875) < 1e-4
    assert abs(cnt - 4) < 1e-4

    acc, avg_acc, cnt = keypoint_pck_accuracy(output, target, mask, 0.5,
                                              np.zeros((2, 2)))
    assert_array_almost_equal(acc, np.array([-1, -1, -1, -1, -1]), decimal=4)
    assert abs(avg_acc) < 1e-4
    assert abs(cnt) < 1e-4

    acc, avg_acc, cnt = keypoint_pck_accuracy(output, target, mask, 0.5,
                                              np.array([[0, 0], [10, 10]]))
    assert_array_almost_equal(acc, np.array([1, 1, -1, 1, 1]), decimal=4)
    assert abs(avg_acc - 1) < 1e-4
    assert abs(cnt - 4) < 1e-4


def test_keypoint_auc():
    output = np.zeros((1, 5, 2))
    target = np.zeros((1, 5, 2))
    mask = np.array([[True, True, False, True, True]])
    # first channel
    output[0, 0] = [10, 4]
    target[0, 0] = [10, 0]
    # second channel
    output[0, 1] = [10, 18]
    target[0, 1] = [10, 10]
    # third channel
    output[0, 2] = [0, 0]
    target[0, 2] = [0, -1]
    # fourth channel
    output[0, 3] = [40, 40]
    target[0, 3] = [30, 30]
    # fifth channel
    output[0, 4] = [20, 10]
    target[0, 4] = [0, 10]

    auc = keypoint_auc(output, target, mask, 20, 4)
    assert abs(auc - 0.375) < 1e-4


def test_keypoint_epe():
    output = np.zeros((1, 5, 2))
    target = np.zeros((1, 5, 2))
    mask = np.array([[True, True, False, True, True]])
    # first channel
    output[0, 0] = [10, 4]
    target[0, 0] = [10, 0]
    # second channel
    output[0, 1] = [10, 18]
    target[0, 1] = [10, 10]
    # third channel
    output[0, 2] = [0, 0]
    target[0, 2] = [-1, -1]
    # fourth channel
    output[0, 3] = [40, 40]
    target[0, 3] = [30, 30]
    # fifth channel
    output[0, 4] = [20, 10]
    target[0, 4] = [0, 10]

    epe = keypoint_epe(output, target, mask)
    assert abs(epe - 11.5355339) < 1e-4


def test_keypoints_from_heatmaps3d():
    heatmaps = np.ones((1, 1, 64, 64, 64), dtype=np.float32)
    heatmaps[0, 0, 10, 31, 40] = 2
    center = np.array([[127, 127]])
    scale = np.array([[64 / 200.0, 64 / 200.0]])
    preds, maxvals = keypoints_from_heatmaps3d(heatmaps, center, scale)

    assert_array_almost_equal(preds, np.array([[[135, 126, 10]]]), decimal=4)
    assert_array_almost_equal(maxvals, np.array([[[2]]]), decimal=4)
    assert isinstance(preds, np.ndarray)
    assert isinstance(maxvals, np.ndarray)


def test_multilabel_classification_accuracy():
    output = np.array([[0.7, 0.8, 0.4], [0.8, 0.1, 0.1]])
    target = np.array([[1, 0, 0], [1, 0, 1]])
    mask = np.array([[True, True, True], [True, True, True]])
    thr = 0.5
    acc = multilabel_classification_accuracy(output, target, mask, thr)
    assert acc == 0

    output = np.array([[0.7, 0.2, 0.4], [0.8, 0.1, 0.9]])
    thr = 0.5
    acc = multilabel_classification_accuracy(output, target, mask, thr)
    assert acc == 1

    thr = 0.3
    acc = multilabel_classification_accuracy(output, target, mask, thr)
    assert acc == 0.5

    mask = np.array([[True, True, False], [True, True, True]])
    acc = multilabel_classification_accuracy(output, target, mask, thr)
    assert acc == 1
