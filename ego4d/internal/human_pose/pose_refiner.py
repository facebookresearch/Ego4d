from typing import List

import numpy as np
import torch

from ego4d.internal.human_pose.utils import COCO_SKELETON, COCO_SKELETON_FLIP_PAIRS
from mmcv.runner import build_optimizer


class OptimizableParameters:
    """Collects parameters for optimization."""

    def __init__(self):
        self.opt_params = []
        return

    def set_param(self, param: torch.Tensor) -> None:
        """Set requires_grad and collect parameters for optimization.
        Args:
            fit_param: whether to optimize this body model parameter
            param: body model parameter
        Returns:
            None
        """
        param.requires_grad = True
        self.opt_params.append(param)
        return

    def parameters(self) -> List[torch.Tensor]:
        """Returns parameters. Compatible with mmcv's build_parameters()
        Returns:
            opt_params: a list of body model parameters for optimization
        """
        return self.opt_params


class SkeletonFit:
    def __init__(self, global_iter, total_global_iters):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = None
        self.num_iter = 500
        self.num_epochs = 10
        self.ftol = 1e-4
        self.optimizer = dict(
            type="LBFGS", max_iter=20, lr=0.1, line_search_fn="strong_wolfe"
        )
        self.verbose = True

        self.init_pose_loss_weight = 1.0
        self.limb_length_loss = 1.0
        self.symmetry_loss_weight = 1.0
        self.temporal_loss_weight = 1.0

        self.global_iter = global_iter
        self.total_global_iters = total_global_iters

        return

    def __call__(self, poses):
        poses = poses[:, :, :3]  ## drop the validity flag
        assert poses.shape[1] == 17 and poses.shape[2] == 3
        init_poses = poses.clone()

        for i in range(self.num_epochs):
            self._optimize_stage(poses, init_poses=init_poses, epoch_idx=i)
        return poses

    def _optimize_stage(self, poses, init_poses, epoch_idx):
        parameters = OptimizableParameters()
        parameters.set_param(poses)
        optimizer = build_optimizer(parameters, self.optimizer)
        pre_loss = None

        for iter_idx in range(self.num_iter):

            def closure():
                optimizer.zero_grad()
                loss_dict = self.evaluate(poses, init_poses, iter_idx, epoch_idx)
                loss = loss_dict["total_loss"]
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            if iter_idx > 0 and pre_loss is not None and self.ftol > 0:
                loss_rel_change = self._compute_relative_change(pre_loss, loss.item())
                if loss_rel_change < self.ftol:
                    if self.verbose:
                        print(f"[ftol={self.ftol}] Early stop at {iter_idx} iter!")
                    break
            pre_loss = loss.item()

        return

    def evaluate(self, poses, init_poses, iter_idx, epoch_idx):
        total_time = poses.shape[0]
        losses = {}

        ##----------compute limbs-----------------
        limb_lengths = {}
        for limb_name in COCO_SKELETON.keys():
            limb_lengths[limb_name] = self.get_limb_length(
                poses, COCO_SKELETON[limb_name]
            )

        ##------------limb length loss---------------------
        limb_length_loss = 0
        for limb_name in COCO_SKELETON.keys():
            average_limb_length = limb_lengths[limb_name].mean()
            limb_length_loss += (
                torch.abs(limb_lengths[limb_name] - average_limb_length)
            ).mean()
        losses["limb_length_loss"] = self.limb_length_loss * limb_length_loss

        ##------------symmetry limb loss---------------------
        symmetry_loss = 0
        for flip_pair in COCO_SKELETON_FLIP_PAIRS.keys():
            limb_pair_name = COCO_SKELETON_FLIP_PAIRS[flip_pair]
            symmetry_loss += (
                torch.abs(
                    limb_lengths[limb_pair_name[0]] - limb_lengths[limb_pair_name[1]]
                )
            ).sum()
        losses["symmetry_loss"] = self.symmetry_loss_weight * symmetry_loss

        ##-------------initpose loss----------------------
        init_pose_loss = ((poses - init_poses) ** 2).sum(dim=2).mean()
        losses["init_pose_loss"] = self.init_pose_loss_weight * init_pose_loss

        ##--------------temporal loss------------------------------------
        temporal_loss = ((poses[1:, :, :] - poses[:-1, :, :]) ** 2).sum(dim=2).mean()
        losses["temporal_loss"] = temporal_loss

        ##-------------------------------------------------------
        if self.verbose:
            msg = "global:{}/{} epoch:{}/{}, iter:{}/{}, loss:".format(
                self.global_iter,
                self.total_global_iters,
                epoch_idx,
                self.num_epochs,
                iter_idx,
                self.num_iter,
            )
            for loss_name, loss in losses.items():
                msg += f"{loss_name}={loss.mean().item():.6f}, "
            if self.verbose:
                print(msg.strip(", "))

        ##-------------------------------------------------------
        total_loss = 0
        for loss_name, loss in losses.items():
            if loss.ndim == 3:
                total_loss = total_loss + loss.sum(dim=(2, 1))
            elif loss.ndim == 2:
                total_loss = total_loss + loss.sum(dim=-1)
            else:
                total_loss = total_loss + loss
        losses["total_loss"] = total_loss

        return losses

    def get_limb_length(self, poses, idxs):
        diff = (poses[:, idxs[0]] - poses[:, idxs[1]]) ** 2  ## T x 3
        length = (torch.sqrt(diff.sum(dim=1))).view(-1, 1)  ## [T, 1]
        return length

    def _compute_relative_change(self, pre_v, cur_v):
        """Compute relative loss change. If relative change is small enough, we
        can apply early stop to accelerate the optimization. (1) When one of
        the value is larger than 1, we calculate the relative change by diving
        their max value. (2) When both values are smaller than 1, it degrades
        to absolute change. Intuitively, if two values are small and close,
        dividing the difference by the max value may yield a large value.
        Args:
            pre_v: previous value
            cur_v: current value
        Returns:
            float: relative change
        """
        return np.abs(pre_v - cur_v) / max([np.abs(pre_v), np.abs(cur_v), 1])


###----------------------------------------------------------------------------
## poses = 716 x 17 x 4
def get_refined_pose3d(poses_numpy, total_global_iters=2):
    fitted_poses_numpy = poses_numpy.copy()
    for i in range(total_global_iters):
        model = SkeletonFit(i, total_global_iters=total_global_iters)
        poses = torch.from_numpy(fitted_poses_numpy.copy()).to(model.device)
        fitted_poses = model(poses)
        fitted_poses_numpy[:, :, :3] = fitted_poses.cpu().detach().numpy()

    return fitted_poses_numpy
