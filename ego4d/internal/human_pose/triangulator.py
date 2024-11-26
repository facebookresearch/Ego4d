import itertools
import random

import numpy as np
import torch
from ego4d.internal.human_pose.camera import ximage_to_xdevice
from ego4d.internal.human_pose.utils import COCO_KP_ORDER
from scipy.optimize import least_squares


# ------------------------------------------------------------------------------------
## performs triangulation
class Triangulator:
    def __init__(
        self,
        time_stamp,
        camera_names,
        cameras,
        multiview_pose2d,
        keypoint_thres=0.7,
        num_keypoints=17,
        sample_all_combinations=True,
        inlier_reproj_error_check=True,
    ):
        self.camera_names = camera_names
        self.cameras = cameras
        self.time_stamp = time_stamp
        self.keypoint_thres = keypoint_thres  ## Keypoint score threshold
        self.n_iters = 1000
        self.reprojection_error_epsilon = 0.01
        self.min_views = 2
        self.min_inliers = 2
        self.include_confidence = False
        self.num_keypoints = num_keypoints
        self.keypoints_idxs = np.array(range(self.num_keypoints))
        self.sample_all_combinations = sample_all_combinations
        self.inlier_reproj_error_check = inlier_reproj_error_check

        # parse the pose2d results, reaarange from camera view to human
        # pose2d is a dictionary,
        # key = (camera_name, camera_mode), val = pose2d_results
        # restructure to (human_id) = [(camera_name, camera_mode): pose2d]
        self.pose2d = {}
        for camera_name, pose2d in multiview_pose2d.items():
            num_humans = 1  ## number of humans detected in this view
            human_name = "aria01"

            for i in range(num_humans):
                ## initialize if not in dict
                if human_name not in self.pose2d.keys():
                    self.pose2d[human_name] = {}

                if multiview_pose2d[camera_name] is not None:
                    keypoints = multiview_pose2d[camera_name][
                        self.keypoints_idxs
                    ]  ##only get the coco keypoints
                else:
                    keypoints = np.zeros((self.num_keypoints, 3))

                self.pose2d[human_name][camera_name] = keypoints

    # https://github.com/karfly/learnable-triangulation-pytorch/blob/9d1a26ea893a513bdff55f30ecbfd2ca8217bf5d/mvn/models/triangulation.py#L72
    def run(self, debug=False):
        points_3d = {}

        # proj_matricies is the extrinsics
        # points are the rays in 3D
        for human_name in sorted(self.pose2d.keys()):
            points_3d[human_name] = np.zeros((self.num_keypoints, 4))
            error = 0

            for keypoint_idx in range(self.num_keypoints):
                proj_matricies = []
                points = []
                choosen_cameras = []

                for view_idx, camera_name in enumerate(self.pose2d[human_name].keys()):
                    point_2d = self.pose2d[human_name][camera_name][
                        keypoint_idx, :2
                    ]  # chop off the confidnece
                    confidence = self.pose2d[human_name][camera_name][keypoint_idx, 2]
                    camera = self.cameras[camera_name]

                    # ---------use high confidence predictions------------------
                    if confidence > self.keypoint_thres:
                        extrinsics = camera.extrinsics[:3, :]  # 3x4

                        # get the ray in 3D

                        ray_3d = ximage_to_xdevice(point_2d, camera)  # 1 x 2
                        ray_3d = np.append(ray_3d, 1)

                        assert len(ray_3d) == 3

                        point = ray_3d.copy()
                        point[2] = confidence  # add keypoint confidence to the point
                        points.append(point)
                        proj_matricies.append(extrinsics)
                        choosen_cameras.append(
                            camera_name
                        )  # camera chosen for triangulation for this point

                # ---------------------------------------------------------------------------------------------
                if len(points) >= self.min_views:
                    # triangulate for a single point
                    (
                        point_3d,
                        inlier_views,
                        reprojection_error_vector,
                    ) = self.triangulate_ransac(
                        proj_matricies,
                        points,
                        n_iters=self.n_iters,
                        reprojection_error_epsilon=self.reprojection_error_epsilon,
                        direct_optimization=True,
                        sample_all_combinations=self.sample_all_combinations,
                        inlier_reproj_error_check=self.inlier_reproj_error_check,
                    )

                    if debug:
                        print(
                            " ".join(
                                [
                                    f"ts:{self.time_stamp}",
                                    f"kp_idx:{keypoint_idx}",
                                    f"kp_name:{COCO_KP_ORDER[keypoint_idx]}",
                                    f"kps_error:{reprojection_error_vector.mean():.5f}",
                                    f"inliers:{len(inlier_views)}",
                                    f"{[choosen_cameras[index] for index in inlier_views]}",
                                ]
                            )
                        )
                    error += reprojection_error_vector.mean()

                    points_3d[human_name][keypoint_idx, :3] = point_3d
                    points_3d[human_name][keypoint_idx, 3] = 1  # mark as valid

            if debug:
                print("{}, error:{}".format(human_name, error))

        return points_3d["aria01"]

    # Original implementation
    # https://github.com/karfly/learnable-triangulation-pytorch/blob/9d1a26ea893a513bdff55f30ecbfd2ca8217bf5d/mvn/models/triangulation.py#L72
    def triangulate_ransac(
        self,
        proj_matricies,
        points,
        n_iters=50,
        reprojection_error_epsilon=0.1,
        direct_optimization=True,
        sample_all_combinations=True,
        inlier_reproj_error_check=True,
    ):
        assert len(proj_matricies) == len(points)
        assert len(points) >= 2

        proj_matricies = np.array(proj_matricies)
        points = np.array(points)
        n_views = len(points)

        # determine inliers
        view_set = set(range(n_views))
        inlier_set = set()
        # Initializing averge reprojection error
        best_avg_reproj_error = np.inf

        # All possible combinations of camera view
        all_comb = itertools.combinations(list(range(n_views)), 2)
        all_comb_list = [list(curr_comb) for curr_comb in all_comb]
        if sample_all_combinations:
            n_iters = len(all_comb_list)

        for i in range(n_iters):
            # Whether sample two views from combination or random sample
            if sample_all_combinations:
                sampled_views = all_comb_list[i]
            else:
                sampled_views = sorted(random.sample(view_set, 2))  ## sample two views

            keypoint_3d_in_base_camera = (
                self.triangulate_point_from_multiple_views_linear(
                    proj_matricies[sampled_views], points[sampled_views]
                )
            )
            reprojection_error_vector = self.calc_reprojection_error_matrix(
                np.array([keypoint_3d_in_base_camera]), points, proj_matricies
            )[0]

            new_inlier_set = set(sampled_views)
            for view in view_set:
                current_reprojection_error = reprojection_error_vector[view]

                if current_reprojection_error < reprojection_error_epsilon:
                    new_inlier_set.add(view)

            if inlier_reproj_error_check:
                # Update best inlier selection only if
                # it has more or equal number of inlier views, and
                # the average reprojection error of those points
                # with lower than threshold error is lower
                if (
                    len(new_inlier_set) >= len(inlier_set)
                    and np.mean(
                        reprojection_error_vector[
                            reprojection_error_vector < reprojection_error_epsilon
                        ]
                    )
                    < best_avg_reproj_error
                ):
                    inlier_set = new_inlier_set
                    best_avg_reproj_error = np.mean(reprojection_error_vector)
            else:
                if len(new_inlier_set) > len(inlier_set):
                    inlier_set = new_inlier_set

        # triangulate using inlier_set
        if len(inlier_set) == 0:
            inlier_set = view_set.copy()

        # -------------------------------
        inlier_list = np.array(sorted(inlier_set))
        inlier_proj_matricies = proj_matricies[inlier_list]
        inlier_points = points[inlier_list]

        keypoint_3d_in_base_camera = self.triangulate_point_from_multiple_views_linear(
            inlier_proj_matricies, inlier_points, self.include_confidence
        )
        reprojection_error_vector = self.calc_reprojection_error_matrix(
            np.array([keypoint_3d_in_base_camera]), inlier_points, inlier_proj_matricies
        )[0]
        # reprojection_error_mean = np.mean(reprojection_error_vector)

        # keypoint_3d_in_base_camera_before_direct_optimization = (
        #     keypoint_3d_in_base_camera
        # )
        # reprojection_error_before_direct_optimization = reprojection_error_mean

        # direct reprojection error minimization
        if direct_optimization:

            def residual_function(x):
                reprojection_error_vector = self.calc_reprojection_error_matrix(
                    np.array([x]), inlier_points, inlier_proj_matricies
                )[0]
                residuals = reprojection_error_vector
                return residuals

            x_0 = np.array(keypoint_3d_in_base_camera)
            res = least_squares(residual_function, x_0, loss="huber", method="trf")

            keypoint_3d_in_base_camera = res.x
            reprojection_error_vector = self.calc_reprojection_error_matrix(
                np.array([keypoint_3d_in_base_camera]),
                inlier_points,
                inlier_proj_matricies,
            )[0]
            res = least_squares(residual_function, x_0, loss="huber", method="trf")

            keypoint_3d_in_base_camera = res.x
            reprojection_error_vector = self.calc_reprojection_error_matrix(
                np.array([keypoint_3d_in_base_camera]),
                inlier_points,
                inlier_proj_matricies,
            )[0]
            # reprojection_error_mean = np.mean(reprojection_error_vector)

        return keypoint_3d_in_base_camera, inlier_list, reprojection_error_vector

    # https://github.com/karfly/learnable-triangulation-pytorch/blob/9d1a26ea893a513bdff55f30ecbfd2ca8217bf5d/mvn/utils/multiview.py#L113
    def triangulate_point_from_multiple_views_linear(
        self, proj_matricies, points, include_confidence=True
    ):
        """Triangulates one point from multiple (N) views using direct linear transformation (DLT).
        For more information look at "Multiple view geometry in computer vision",
        Richard Hartley and Andrew Zisserman, 12.2 (p. 312).
        Args:
            proj_matricies numpy array of shape (N, 3, 4): sequence of projection matricies (3x4)
            points numpy array of shape (N, 3): sequence of points' coordinates and confidence
        Returns:
            point_3d numpy array of shape (3,): triangulated point
        """
        assert len(proj_matricies) == len(points)

        points_confidence = points[:, 2].copy()
        points = points[:, :2].copy()

        # normalize points_confidence
        points_confidence /= points_confidence.max()

        n_views = len(proj_matricies)
        A = np.zeros((2 * n_views, 4))
        for j in range(len(proj_matricies)):
            A[j * 2 + 0] = (
                points[j][0] * proj_matricies[j][2, :] - proj_matricies[j][0, :]
            )
            A[j * 2 + 1] = (
                points[j][1] * proj_matricies[j][2, :] - proj_matricies[j][1, :]
            )

            # weight by the point confidence
            if include_confidence == True:
                A[j * 2 + 0] *= points_confidence[j]
                A[j * 2 + 1] *= points_confidence[j]

            u, s, vh = np.linalg.svd(A, full_matrices=False)

        point_3d_homo = vh[3, :]

        point_3d = self.homogeneous_to_euclidean(point_3d_homo)
        return point_3d

    # https://github.com/karfly/learnable-triangulation-pytorch/blob/9d1a26ea893a513bdff55f30ecbfd2ca8217bf5d/mvn/utils/multiview.py#L186
    def calc_reprojection_error_matrix(
        self, keypoints_3d, keypoints_2d_list, proj_matricies
    ):
        reprojection_error_matrix = []
        for keypoints_2d, proj_matrix in zip(keypoints_2d_list, proj_matricies):
            if len(keypoints_2d) == 3:
                keypoints_2d = keypoints_2d[:2]  ## chop off the confidence

            keypoints_2d_projected = (
                self.project_3d_points_to_image_plane_without_distortion(
                    proj_matrix, keypoints_3d
                )
            )
            reprojection_error = (
                1
                / 2
                * np.sqrt(np.sum((keypoints_2d - keypoints_2d_projected) ** 2, axis=1))
            )
            reprojection_error_matrix.append(reprojection_error)

        return np.vstack(reprojection_error_matrix).T

    def project_3d_points_to_image_plane_without_distortion(
        self, proj_matrix, points_3d, convert_back_to_euclidean=True
    ):
        """Project 3D points to image plane not taking into account distortion
        Args:
            proj_matrix numpy array or torch tensor of shape (3, 4): projection matrix
            points_3d numpy array or torch tensor of shape (N, 3): 3D points
            convert_back_to_euclidean bool: if True, then resulting points will be converted to euclidean coordinates
                                            NOTE: division by zero can be here if z = 0
        Returns:
            numpy array or torch tensor of shape (N, 2): 3D points projected to image plane
        """
        if isinstance(proj_matrix, np.ndarray) and isinstance(points_3d, np.ndarray):
            result = self.euclidean_to_homogeneous(points_3d) @ proj_matrix.T
            if convert_back_to_euclidean:
                result = self.homogeneous_to_euclidean(result)
            return result
        elif torch.is_tensor(proj_matrix) and torch.is_tensor(points_3d):
            result = self.euclidean_to_homogeneous(points_3d) @ proj_matrix.t()
            if convert_back_to_euclidean:
                result = self.homogeneous_to_euclidean(result)
            return result
        else:
            raise TypeError("Works only with numpy arrays and PyTorch tensors.")

    # https://github.com/karfly/learnable-triangulation-pytorch/blob/9d1a26ea893a513bdff55f30ecbfd2ca8217bf5d/mvn/utils/multiview.py#L72
    def homogeneous_to_euclidean(self, points):
        """Converts homogeneous points to euclidean
        Args:
            points numpy array or torch tensor of shape (N, M + 1): N homogeneous points of dimension M
        Returns:
            numpy array or torch tensor of shape (N, M): euclidean points
        """
        if isinstance(points, np.ndarray):
            return (points.T[:-1] / points.T[-1]).T
        elif torch.is_tensor(points):
            return (points.transpose(1, 0)[:-1] / points.transpose(1, 0)[-1]).transpose(
                1, 0
            )
        else:
            raise TypeError("Works only with numpy arrays and PyTorch tensors.")

    # https://github.com/karfly/learnable-triangulation-pytorch/blob/9d1a26ea893a513bdff55f30ecbfd2ca8217bf5d/mvn/utils/multiview.py#L55
    def euclidean_to_homogeneous(self, points):
        """Converts euclidean points to homogeneous
        Args:
            points numpy array or torch tensor of shape (N, M): N euclidean points of dimension M
        Returns:
            numpy array or torch tensor of shape (N, M + 1): homogeneous points
        """
        if isinstance(points, np.ndarray):
            return np.hstack([points, np.ones((len(points), 1))])
        elif torch.is_tensor(points):
            return torch.cat(
                [
                    points,
                    torch.ones(
                        (points.shape[0], 1), dtype=points.dtype, device=points.device
                    ),
                ],
                dim=1,
            )
        else:
            raise TypeError("Works only with numpy arrays and PyTorch tensors.")
