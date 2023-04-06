import functools
from mmcv.runner import HOOKS, Hook
import os
import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

#####--------------------------------------------------------
@HOOKS.register_module()
class VisualizeHook(Hook):

    def __init__(self, vis_every_iters=200, max_samples=16):
        self.vis_every_iters = vis_every_iters
        self.max_samples = max_samples
        return

    def before_run(self, runner):
        pass
        return

    def after_run(self, runner):
        pass
        return

    def before_epoch(self, runner):
        pass
        return

    def after_epoch(self, runner):
        pass
        return

    def before_iter(self, runner):
        pass
        return

    def after_iter(self, runner):
        if runner._iter % self.vis_every_iters != 0:
            return

        ##------------------------------------
        data_batch = runner.data_batch
        image = data_batch['img'] ## this is normalized
        target = data_batch['target']
        target_weight = data_batch['target_weight']

        outputs = runner.outputs
        output = outputs['output'] 

        if len(image) > self.max_samples:
            image = image[:self.max_samples]
            target = target[:self.max_samples]
            target_weight = target_weight[:self.max_samples]
            output = output[:self.max_samples]

        ##------------------------------------
        vis_dir = os.path.join(runner.work_dir, 'vis')
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir, exist_ok=True)

        prefix = os.path.join(vis_dir, 'train')
        suffix = str(runner._iter).zfill(6)

        original_image = batch_unnormalize_image(image) ## recolored

        save_batch_heatmaps(original_image, target, '{}_{}_hm_gt.jpg'.format(prefix, suffix), normalize=False, scale=4)
        save_batch_heatmaps(original_image, output, '{}_{}_hm_pred.jpg'.format(prefix, suffix), normalize=False, scale=4)
        save_batch_image_with_joints(255*original_image, target, target_weight, '{}_{}_gt.jpg'.format(prefix, suffix))
        save_batch_image_with_joints(255*original_image, output, torch.ones_like(target_weight), '{}_{}_pred.jpg'.format(prefix, suffix))

        return


#####--------------------------------------------------------
@HOOKS.register_module()
class VisualizeCenterHook(Hook):

    def __init__(self, vis_every_iters=200, max_samples=16):
        self.vis_every_iters = vis_every_iters
        self.max_samples = max_samples
        return

    def before_run(self, runner):
        pass
        return

    def after_run(self, runner):
        pass
        return

    def before_epoch(self, runner):
        pass
        return

    def after_epoch(self, runner):
        pass
        return

    def before_iter(self, runner):
        pass
        return
    
    def after_val_iter(self, runner):
        if runner._inner_iter % self.vis_every_iters != 0:
            return

        data_batch = runner.data_batch
        image = data_batch['img'] ## this is normalized
        target = data_batch['target']
        target_weight = data_batch['target_weight']
        body_center_heatmap = data_batch['body_center_heatmap']

        output = runner.outputs['results']['output_heatmap']

        if len(image) > self.max_samples:
            image = image[:self.max_samples]
            target = target[:self.max_samples]
            target_weight = target_weight[:self.max_samples]
            body_center_heatmap = body_center_heatmap[:self.max_samples]
            output = output[:self.max_samples]

        ##------------------------------------
        vis_dir = os.path.join(runner.work_dir, 'vis')
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir, exist_ok=True)

        prefix = os.path.join(vis_dir, 'val')
        suffix = str(runner._iter).zfill(6)

        original_image = batch_unnormalize_image(image) ## recolored

        save_batch_heatmaps(original_image, target, '{}_{}_hm_gt.jpg'.format(prefix, suffix), normalize=False, scale=4)
        save_batch_heatmaps(original_image, output, '{}_{}_hm_pred.jpg'.format(prefix, suffix), normalize=False, scale=4)

        save_batch_image_with_joints_and_body_center(255*original_image, target, target_weight, body_center_heatmap, '{}_{}_gt.jpg'.format(prefix, suffix))
        save_batch_image_with_joints(255*original_image, output, torch.ones_like(target_weight), '{}_{}_pred.jpg'.format(prefix, suffix))

        return


    def after_iter(self, runner):
        if runner._iter % self.vis_every_iters != 0:
            return

        ##------------------------------------
        data_batch = runner.data_batch
        image = data_batch['img'] ## this is normalized
        target = data_batch['target']
        target_weight = data_batch['target_weight']
        body_center_heatmap = data_batch['body_center_heatmap']

        outputs = runner.outputs
        output = outputs['output'] ## primary

        if len(image) > self.max_samples:
            image = image[:self.max_samples]
            target = target[:self.max_samples]
            target_weight = target_weight[:self.max_samples]
            body_center_heatmap = body_center_heatmap[:self.max_samples]
            output = output[:self.max_samples]

        ##------------------------------------
        vis_dir = os.path.join(runner.work_dir, 'vis')
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir, exist_ok=True)

        prefix = os.path.join(vis_dir, 'train')
        suffix = str(runner._iter).zfill(6)

        original_image = batch_unnormalize_image(image) ## recolored

        save_batch_heatmaps(original_image, target, '{}_{}_hm_gt.jpg'.format(prefix, suffix), normalize=False, scale=4)
        save_batch_heatmaps(original_image, output, '{}_{}_hm_pred.jpg'.format(prefix, suffix), normalize=False, scale=4)

        save_batch_image_with_joints_and_body_center(255*original_image, target, target_weight, body_center_heatmap, '{}_{}_gt.jpg'.format(prefix, suffix))
        save_batch_image_with_joints(255*original_image, output, torch.ones_like(target_weight), '{}_{}_pred.jpg'.format(prefix, suffix))

        return

###------------------------------------------------------
def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2) ## B x 17
    maxvals = np.amax(heatmaps_reshaped, 2) ## B x 17

    maxvals = maxvals.reshape((batch_size, num_joints, 1)) ## B x 17 x 1
    idx = idx.reshape((batch_size, num_joints, 1)) ## B x 17 x 1

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32) ## B x 17 x 2, like repeat in pytorch

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def save_batch_heatmaps(batch_image, batch_heatmaps, file_name, normalize=True, scale=4):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    ## normalize image
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    ## check if type of batch_heatmaps is numpy.ndarray
    if isinstance(batch_heatmaps, np.ndarray):
        preds, maxvals = get_max_preds(batch_heatmaps)
        batch_heatmaps = torch.from_numpy(batch_heatmaps)
    else:
        preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())
    
    preds = preds*scale ## scale to original image size

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)*scale
    heatmap_width = batch_heatmaps.size(3)*scale

    grid_image = np.zeros((batch_size*heatmap_height,
                           (num_joints+1)*heatmap_width,
                           3),
                          dtype=np.uint8)

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        resized_image = cv2.resize(image, (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            colored_heatmap = cv2.resize(colored_heatmap, (int(heatmap_width), int(heatmap_height)))
            masked_image = colored_heatmap*0.7 + resized_image*0.3
            cv2.circle(masked_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)


def save_batch_image_with_joints(batch_image, batch_heatmaps, batch_target_weight, file_name, scale=4, nrow=8, padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''

    B, C, H, W = batch_image.size()

    ## check if type of batch_heatmaps is numpy.ndarray
    if isinstance(batch_heatmaps, np.ndarray):
        batch_joints, _ = get_max_preds(batch_heatmaps)
    else:
        batch_joints, _ = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    batch_joints = batch_joints*scale ## 4 is the ratio of output heatmap and input image

    if isinstance(batch_joints, torch.Tensor):
        batch_joints = batch_joints.cpu().numpy()

    if isinstance(batch_target_weight, torch.Tensor):
        batch_target_weight = batch_target_weight.cpu().numpy()
        batch_target_weight = batch_target_weight.reshape(B, 17) ## B x 17

    grid = []

    for i in range(B):
        image = batch_image[i].permute(1, 2, 0).cpu().numpy() #image_size x image_size x RGB
        image = image.copy()
        kps = batch_joints[i]

        kps_vis = batch_target_weight[i].reshape(17, 1)
        kps = np.concatenate((kps, kps_vis), axis=1)
        kp_vis_image = coco_vis_keypoints(image, kps, vis_thres=0.3, alpha=0.7) ## H, W, C
        kp_vis_image = kp_vis_image.transpose((2, 0, 1)).astype(np.float32)
        kp_vis_image = torch.from_numpy(kp_vis_image.copy())
        grid.append(kp_vis_image)

    grid = torchvision.utils.make_grid(grid, nrow, padding)
    ndarr = grid.byte().permute(1, 2, 0).cpu().numpy()
    ndarr = cv2.cvtColor(ndarr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_name, ndarr)
    return

def save_batch_image_with_joints_and_body_center(batch_image, batch_heatmaps, batch_target_weight, batch_body_center_heatmap, file_name, nrow=8, padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''

    B, C, H, W = batch_image.size()

    if isinstance(batch_heatmaps, np.ndarray):
        batch_joints, _ = get_max_preds(batch_heatmaps)
    else:
        batch_joints, _ = get_max_preds(batch_heatmaps.detach().cpu().numpy())
    
    batch_joints = batch_joints*4 ## 4 is the ratio of output heatmap and input image

    if isinstance(batch_joints, torch.Tensor):
        batch_joints = batch_joints.cpu().numpy()

    if isinstance(batch_target_weight, torch.Tensor):
        batch_target_weight = batch_target_weight.cpu().numpy()
        batch_target_weight = batch_target_weight.reshape(B, 17) ## B x 17

    grid = []

    for i in range(B):
        image = batch_image[i].permute(1, 2, 0).cpu().numpy() #image_size x image_size x RGB
        image = image.copy()

        heatmap = batch_body_center_heatmap[i][0].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()
        
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        colored_heatmap = cv2.resize(colored_heatmap, (int(image.shape[1]), int(image.shape[0])))
        masked_image = colored_heatmap*0.7 + image*0.3

        ## bgr to rgb image
        masked_image = masked_image[:, :, [2, 1, 0]]

        kps = batch_joints[i]
        kps_vis = batch_target_weight[i].reshape(17, 1)
        kps = np.concatenate((kps, kps_vis), axis=1)
        kp_vis_image = coco_vis_keypoints(image, kps, vis_thres=0.3, alpha=0.7) ## H, W, C
        kp_vis_image = kp_vis_image.transpose((2, 0, 1)).astype(np.float32)
        masked_image = masked_image.transpose((2, 0, 1)).astype(np.float32)
        
        kp_vis_image = np.concatenate((kp_vis_image, masked_image), axis=1)
        kp_vis_image = torch.from_numpy(kp_vis_image.copy())
        grid.append(kp_vis_image)

    grid = torchvision.utils.make_grid(grid, nrow, padding)
    ndarr = grid.byte().permute(1, 2, 0).cpu().numpy()
    ndarr = cv2.cvtColor(ndarr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_name, ndarr)
    return

###-------------------------------------------------------------------------
def batch_unnormalize_image(images, normalize=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])):
    images[:, 0, :, :] = (images[:, 0, :, :]*normalize.std[0]) + normalize.mean[0] 
    images[:, 1, :, :] = (images[:, 1, :, :]*normalize.std[1]) + normalize.mean[1] 
    images[:, 2, :, :] = (images[:, 2, :, :]*normalize.std[2]) + normalize.mean[2] 
    return images



# ------------------------------------------------------------------------------------
# standard COCO format, 17 joints
COCO_KP_ORDER = [
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle'
    ]


def kp_connections(keypoints):
    kp_lines = [
        [keypoints.index('left_eye'), keypoints.index('right_eye')],
        [keypoints.index('left_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('right_shoulder'), keypoints.index('left_shoulder')],
        [keypoints.index('right_hip'), keypoints.index('left_hip')],
    ]
    return kp_lines


COCO_KP_CONNECTIONS = kp_connections(COCO_KP_ORDER)

# ------------------------------------------------------------------------------------
def coco_vis_keypoints(image, kps, vis_thres=0.3, alpha=0.7):
    # image is [image_size, image_size, RGB] #numpy array
    # kps is [17, 3] #numpy array
    kps = kps.astype(np.int16)
    bgr_image = image[:, :, ::-1] ##if this is directly in function call, this produces weird opecv cv2 Umat errors
    kp_image = vis_keypoints(bgr_image, kps.T, vis_thres, alpha) #convert to bgr
    kp_image = kp_image[:, :, ::-1] #bgr to rgb

    return kp_image

# ------------------------------------------------------------------------------------
def vis_keypoints(img, kps, kp_thresh=-1, alpha=0.7):
    """Visualizes keypoints (adapted from vis_one_image).
    kps has shape (3, #keypoints) where 3 rows are (x, y, depth z).
    needs a BGR image as it only uses opencv functions, returns a bgr image
    """
    dataset_keypoints = COCO_KP_ORDER
    kp_lines = COCO_KP_CONNECTIONS

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw mid shoulder / mid hip first for better visualization.
    mid_shoulder = (
        kps[:2, dataset_keypoints.index('right_shoulder')] +
        kps[:2, dataset_keypoints.index('left_shoulder')]) // 2
    sc_mid_shoulder = np.minimum(
        kps[2, dataset_keypoints.index('right_shoulder')],
        kps[2, dataset_keypoints.index('left_shoulder')])
    mid_hip = (
        kps[:2, dataset_keypoints.index('right_hip')] +
        kps[:2, dataset_keypoints.index('left_hip')]) // 2
    sc_mid_hip = np.minimum(
        kps[2, dataset_keypoints.index('right_hip')],
        kps[2, dataset_keypoints.index('left_hip')])
    nose_idx = dataset_keypoints.index('nose')

    if sc_mid_shoulder > kp_thresh and kps[2, nose_idx] > kp_thresh:
        kp_mask = cv2.line(
            kp_mask, tuple(mid_shoulder), tuple(kps[:2, nose_idx]),
            color=colors[len(kp_lines)], thickness=2, lineType=cv2.LINE_AA)
    if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
        kp_mask = cv2.line(
            kp_mask, tuple(mid_shoulder), tuple(mid_hip),
            color=colors[len(kp_lines) + 1], thickness=2, lineType=cv2.LINE_AA)

    # Draw the keypoints.
    for l in range(len(kp_lines)):
        i1 = kp_lines[l][0]
        i2 = kp_lines[l][1]
        p1 = kps[0, i1], kps[1, i1]
        p2 = kps[0, i2], kps[1, i2]
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            kp_mask = cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            kp_mask = cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            kp_mask = cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    ## weird opencv bug on cv2UMat vs numpy
    if type(kp_mask) != type(img):
        kp_mask = kp_mask.get()

    # Blend the keypoints.
    result = cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)
    return result