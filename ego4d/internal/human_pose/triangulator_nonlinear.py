import numpy as np
import os
import cv2
from scipy.optimize import least_squares
import random
import pycolmap
from utils import COCO_KP_ORDER


COCO_SKELETON = {
                'left_leg': [13, 15], ## l-knee to l-ankle
                'right_leg': [14, 16], ## r-knee to r-ankle
                'left_thigh': [11, 13], ## l-hip to l-knee
                'right_thigh': [12, 14], ## r-hip to r-knee
                'hip': [11, 12], ## l-hip to r-hip
                'left_torso': [5, 11], ## l-shldr to l-hip
                'right_torso': [6, 12], ## r-shldr to r-hip
                'left_bicep': [5, 7], ## l-shldr to l-elbow
                'right_bicep': [6, 8], ## r-shldr to r-elbow
                'shoulder': [5, 6], ## l-shldr to r-shldr
                'left_hand': [7, 9], ## l-elbow to l-wrist
                'right_hand': [8, 10], ## r-elbow to r-wrist
                'left_face': [1, 0], ## l-eye to nose
                'right_face': [2, 0], ## l-eye to nose
                'face': [1, 2], ## l-eye to r-eye
                'left_ear': [1, 3], ## l-eye to l-ear
                'right_ear': [2, 4], ## l-eye to r-ear
                'left_neck': [3, 5], ## l-ear to l-shldr
                'right_neck': [4, 6], ## r-ear to r-shldr
}

###----------------------------------------------------------------------------
COCO_SKELETON_FLIP_PAIRS = {
                    'leg':    ('left_leg', 'right_leg'),
                    'thigh':    ('left_thigh', 'right_thigh'),
                    'torso':    ('left_torso', 'right_torso'),
                    'bicep':    ('left_bicep', 'right_bicep'),
                    'hand':    ('left_hand', 'right_hand'),
                    'face':    ('left_face', 'right_face'),
                    'ear':    ('left_ear', 'right_ear'),
                    'neck':    ('left_neck', 'right_neck'),
                    }
##------------------------------------------------------------------------------------
## performs triangulation
class TriangulatorNonLinear:
    def __init__(self, time_stamp, camera_names, cameras, multiview_pose2d):
        self.camera_names = camera_names
        self.cameras = cameras
        self.time_stamp = time_stamp
        self.keypoint_thres = 0.7 ## Keypoint score threshold
        self.n_iters = 1000
        self.reprojection_error_epsilon = 0.01
        self.min_views = 2
        self.min_inliers = 2
        self.include_confidence = False
        
        self.coco_17_keypoints_idxs = np.array(range(17)) ## indexes of the COCO keypoints
        self.keypoints_idxs = self.coco_17_keypoints_idxs
        self.num_keypoints = len(self.keypoints_idxs)

        self.pose2d = {}

        for camera_name, pose2d in multiview_pose2d.items():

            if multiview_pose2d[camera_name] is not None:
                keypoints =  multiview_pose2d[camera_name][self.keypoints_idxs] ##only get the coco keypoints
            else:
                keypoints = np.zeros((self.num_keypoints, 3))
            
            self.pose2d[camera_name] = keypoints

        return

    ##-----------------------------------------
    def objective_function_weighted(self, params, keypoint_indices, cam_matrices, detected_keypoints):
        num_keypoints = len(keypoint_indices)
        num_cameras = len(cam_matrices)
        
        # Reshape the parameters into 3D keypoints and camera weights
        triangulated_keypoints = params.reshape(-1, 3)

        # Calculate the reprojection error with camera weights
        reprojection_error = 0
        for cam_idx in range(num_cameras):
            cam_matrix = cam_matrices[cam_idx]
            for kp_idx, keypoint_index in enumerate(keypoint_indices):
                proj_keypoint = cam_matrix @ np.append(triangulated_keypoints[keypoint_index], 1)
                proj_keypoint = proj_keypoint[:2] / proj_keypoint[2] ## 2

                # Apply the camera weight for the current keypoint
                confidence = detected_keypoints[cam_idx, kp_idx, 2]
                this_reprojection_error = confidence * np.linalg.norm(proj_keypoint - detected_keypoints[cam_idx, kp_idx, :2]) ** 2
                reprojection_error += this_reprojection_error

        # Calculate the human body constraints error
        constraints_error = 0

         ##----------compute limbs-----------------
        limb_lengths = {}
        for limb_name in COCO_SKELETON.keys():
            limb_length = np.linalg.norm(triangulated_keypoints[COCO_SKELETON[limb_name][0]] \
                                         - triangulated_keypoints[COCO_SKELETON[limb_name][1]])

            limb_lengths[limb_name] = limb_length

        ## -----------symmetry--------------------
        for limb_name, (left_kp, right_kp) in COCO_SKELETON_FLIP_PAIRS.items():
            limb_error = np.linalg.norm(limb_lengths[left_kp] - limb_lengths[right_kp])
            constraints_error += limb_error

        return reprojection_error + constraints_error
    
    ##-----------------------------------------
    # https://github.com/karfly/learnable-triangulation-pytorch/blob/9d1a26ea893a513bdff55f30ecbfd2ca8217bf5d/mvn/models/triangulation.py#L72
    def run(self, init_pose3d=None):
        points_3d = np.zeros((self.num_keypoints, 4))
        camera_names = sorted(self.pose2d.keys())
        num_views = len(camera_names)

        ###-----------------------------------------
        ## proj_matricies is the extrinsics
        ## points are the rays in 3D
        error = 0
            
        ## add extrinsics in sorted camera order
        cam_matrices = [self.cameras[camera_name].extrinsics[:3, :].reshape(1, 3, 4) for camera_name in camera_names]
        cam_matrices = np.concatenate(cam_matrices, axis=0) ## 4 x 3 x 4

        detected_keypoints = [] ## all rays in 3D

        for _, camera_name in enumerate(camera_names):
            points_2d = self.pose2d[camera_name][:, :2] ## 17
            confidences = self.pose2d[camera_name][:, 2] ## 17 x 1

            ## if confidence is below threshold, set to 0
            confidences[confidences < self.keypoint_thres] = 0.0

            rays_3d = self.cameras[camera_name].camera_model.image_to_world(points_2d) ## returns list of size 17, each element is 1 x 2
            rays_3d = [ray_3d.reshape(1, -1) for ray_3d in rays_3d]
            rays_3d = np.concatenate(rays_3d, axis=0) ## 17 x 2

            ## concatenate the confidence to the rays
            rays_3d = np.concatenate((rays_3d, confidences.reshape(-1, 1)), axis=1) ## 17 x 3
            detected_keypoints.append(rays_3d.reshape(1, -1, 3))

        detected_keypoints = np.concatenate(detected_keypoints, axis=0) ## 17 x 3

        ###------------------------optimize!------------------------
        # Provide initial estimates for the 3D keypoints (use triangulation or any other method)

        if init_pose3d is not None:
            initial_keypoint_estimates = init_pose3d[:, :3]
        else:
            initial_keypoint_estimates = np.zeros((self.num_keypoints, 3))

        # Flatten the initial estimates for keypoints and camera weights
        initial_params = initial_keypoint_estimates.flatten()

        # Define the indices of the keypoints
        keypoint_indices = list(range(self.num_keypoints))

        # Run the optimization with the new objective function
        result = least_squares(self.objective_function_weighted, initial_params, \
                               args=(keypoint_indices, cam_matrices, detected_keypoints),
                                 method='trf', loss='soft_l1', f_scale=0.1, verbose=2, max_nfev=1000)

        # Reshape the result to obtain the optimized 3D keypoints and camera weights
        optimized_keypoints = result.x[:self.num_keypoints * 3].reshape(-1, 3)
        
        points_3d[:, :3] = optimized_keypoints
        points_3d[:, 3] = 1.0

        return points_3d

    ##-----------------------------------------
    # https://github.com/karfly/learnable-triangulation-pytorch/blob/9d1a26ea893a513bdff55f30ecbfd2ca8217bf5d/mvn/models/triangulation.py#L72
    def triangulate_ransac(self, proj_matricies, points, n_iters=50, reprojection_error_epsilon=0.1, direct_optimization=True):
        assert len(proj_matricies) == len(points)
        assert len(points) >= 2

        proj_matricies = np.array(proj_matricies)
        points = np.array(points)

        n_views = len(points)

        # determine inliers
        view_set = set(range(n_views))
        inlier_set = set()
        for i in range(n_iters):
            sampled_views = sorted(random.sample(view_set, 2)) ## sample two views

            keypoint_3d_in_base_camera = self.triangulate_point_from_multiple_views_linear(proj_matricies[sampled_views], points[sampled_views])
            reprojection_error_vector = self.calc_reprojection_error_matrix(np.array([keypoint_3d_in_base_camera]), points, proj_matricies)[0]

            new_inlier_set = set(sampled_views)
            for view in view_set:
                current_reprojection_error = reprojection_error_vector[view]

                if current_reprojection_error < reprojection_error_epsilon:
                    new_inlier_set.add(view)

            if len(new_inlier_set) > len(inlier_set):
                inlier_set = new_inlier_set

        # triangulate using inlier_set
        if len(inlier_set) == 0:
            inlier_set = view_set.copy()

        ##-------------------------------
        inlier_list = np.array(sorted(inlier_set))
        inlier_proj_matricies = proj_matricies[inlier_list]
        inlier_points = points[inlier_list]

        keypoint_3d_in_base_camera = self.triangulate_point_from_multiple_views_linear(inlier_proj_matricies, inlier_points, self.include_confidence)
        reprojection_error_vector = self.calc_reprojection_error_matrix(np.array([keypoint_3d_in_base_camera]), inlier_points, inlier_proj_matricies)[0]
        reprojection_error_mean = np.mean(reprojection_error_vector)

        keypoint_3d_in_base_camera_before_direct_optimization = keypoint_3d_in_base_camera
        reprojection_error_before_direct_optimization = reprojection_error_mean

        # direct reprojection error minimization
        if direct_optimization:
            def residual_function(x):
                reprojection_error_vector = self.calc_reprojection_error_matrix(np.array([x]), inlier_points, inlier_proj_matricies)[0]
                residuals = reprojection_error_vector
                return residuals

            x_0 = np.array(keypoint_3d_in_base_camera)
            res = least_squares(residual_function, x_0, loss='huber', method='trf')

            keypoint_3d_in_base_camera = res.x
            reprojection_error_vector = self.calc_reprojection_error_matrix(np.array([keypoint_3d_in_base_camera]), inlier_points, inlier_proj_matricies)[0]
            reprojection_error_mean = np.mean(reprojection_error_vector)

        return keypoint_3d_in_base_camera, inlier_list, reprojection_error_vector

    ##-----------------------------------------
    # https://github.com/karfly/learnable-triangulation-pytorch/blob/9d1a26ea893a513bdff55f30ecbfd2ca8217bf5d/mvn/utils/multiview.py#L113
    def triangulate_point_from_multiple_views_linear(self, proj_matricies, points, include_confidence=True):
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

        ###-----normalize points_confidence-----
        points_confidence /= points_confidence.max()

        n_views = len(proj_matricies)
        A = np.zeros((2 * n_views, 4))
        for j in range(len(proj_matricies)):
            A[j * 2 + 0] = points[j][0] * proj_matricies[j][2, :] - proj_matricies[j][0, :]
            A[j * 2 + 1] = points[j][1] * proj_matricies[j][2, :] - proj_matricies[j][1, :]

            ## weight by the point confidence
            if include_confidence == True:
                A[j * 2 + 0] *= points_confidence[j]
                A[j * 2 + 1] *= points_confidence[j]

        u, s, vh =  np.linalg.svd(A, full_matrices=False)
        point_3d_homo = vh[3, :]

        point_3d = self.homogeneous_to_euclidean(point_3d_homo)

        return point_3d

    ##-----------------------------------------
    # https://github.com/karfly/learnable-triangulation-pytorch/blob/9d1a26ea893a513bdff55f30ecbfd2ca8217bf5d/mvn/utils/multiview.py#L186
    def calc_reprojection_error_matrix(self, keypoints_3d, keypoints_2d_list, proj_matricies):
        reprojection_error_matrix = []
        for keypoints_2d, proj_matrix in zip(keypoints_2d_list, proj_matricies):

            if len(keypoints_2d) == 3:
                keypoints_2d = keypoints_2d[:2] ## chop off the confidence

            keypoints_2d_projected = self.project_3d_points_to_image_plane_without_distortion(proj_matrix, keypoints_3d)
            reprojection_error = 1 / 2 * np.sqrt(np.sum((keypoints_2d - keypoints_2d_projected) ** 2, axis=1))
            reprojection_error_matrix.append(reprojection_error)

        return np.vstack(reprojection_error_matrix).T

    ##-----------------------------------------
    def project_3d_points_to_image_plane_without_distortion(self, proj_matrix, points_3d, convert_back_to_euclidean=True):
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

    ##-----------------------------------------
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
            return (points.transpose(1, 0)[:-1] / points.transpose(1, 0)[-1]).transpose(1, 0)
        else:
            raise TypeError("Works only with numpy arrays and PyTorch tensors.")

    ##-----------------------------------------
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
            return torch.cat([points, torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)], dim=1)
        else:
            raise TypeError("Works only with numpy arrays and PyTorch tensors.")