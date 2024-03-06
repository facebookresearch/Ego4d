# This is stored in the new camera format
# python halo_triangulate.py /large_experiments/egoexo/egopose/suyogjain/project_retriangulation_production/camera_pose/ /large_experiments/egoexo/egopose/suyogjain/project_retriangulation_production/ego_pose_latest/hand/annotation/ new

# This is stored in the old camera format
# python halo_triangulate.py /large_experiments/egoexo/egopose/suyogjain/project_retriangulation_production/ego_pose_latest/hand/camera_pose/ /large_experiments/egoexo/egopose/suyogjain/project_retriangulation_production/ego_pose_latest/hand/annotation/ old
import numpy as np
import json
import sys
import os

def load_json(fname):
    with open(fname) as f:
        data = json.load(f)    
    return data


def triangulate(camera_calibrations):
    # Check if input is valid
    if not camera_calibrations or len(camera_calibrations) < 2:
        return None
    for calibration in camera_calibrations:
        if not calibration or len(calibration) != 4:
            return None
        x, y, M, cam_name = calibration
        if not (isinstance(x, float) and isinstance(y, float) and isinstance(M, np.ndarray)):
            return None

    # Create reducer function
    def reducer(calibration):
        x, y, M, _ = calibration
        #print(x, y, M.shape)
        M1 = M[0, :]
        M2 = M[1, :]
        M3 = M[2, :]
        uM3 = x*M3
        vM3 = y*M3

        c1 = M1 - uM3
        c2 = M2 - vM3
        return c1, c2

    
    # Apply reducer to input array of camera calibrations
    constraints = []
    for calibration in camera_calibrations:
        c1, c2 = reducer(calibration)
        #print(c1, c2)        
        constraints.append(c1)
        constraints.append(c2)

    constraints= np.array(constraints)
    #print(constraints)

    A = constraints[:, :3]
    b = -1*constraints[:, 3]
    A_inv = np.linalg.inv(np.dot(A.T, A))
    P = np.dot(A_inv, np.dot(A.T, b))
    point3D = dict()
    point3D['x'], point3D['y'], point3D['z'] = P[0], P[1], P[2]         
    return point3D



def process_annotation(pose2d, camera_matrices):
    pose2d_transformed = dict()    
    for cam_name in pose2d:       
        pose_list = pose2d[cam_name]    
        pose_array = list()        
        keypoints_list = pose_list.keys()
                
        for kp_name in keypoints_list:            
            if kp_name not in pose2d_transformed:
                pose2d_transformed[kp_name] = list()
            
            ann = pose_list[kp_name]                
            if ann['placement']=='auto':
                continue
            else:
                #print('Manual annotation (x, y):', cam_name, kp_name, ann['x'], ann['y'])                                
                pose2d_transformed[kp_name].append([ann['x'], ann['y'], camera_matrices[cam_name]['camera_matrix'], cam_name])
        
    return pose2d_transformed

def projectPoint(point3D, cmatrix):
    R = cmatrix['camera_extrinsics'][:, :3]
    t = cmatrix['camera_extrinsics'][:, -1]
    K = cmatrix['camera_intrinsics']

    # 3D world point
    X_w, Y_w, Z_w = point3D['x'], point3D['y'], point3D['z']
    P_w = np.array([X_w, Y_w, Z_w, 1])  # Homogeneous coordinates
    
    # Convert to camera coordinates
    P_c = R.dot(P_w[:3]) + t
    
    # Project onto 2D image plane
    P_2D_homogeneous = K.dot(P_c)

    # Convert to Cartesian coordinates
    projected_pose = P_2D_homogeneous[:2] / P_2D_homogeneous[2]   
    point2D = {'x': projected_pose[0], 'y': projected_pose[1]}
       
    return point2D

def project2d(point3D, camera_matrices):
    points2D = dict()    
    for cam_name in camera_matrices:        
        point2D = projectPoint(point3D, camera_matrices[cam_name])
        points2D[cam_name] = point2D
    return points2D

def process_camera_pose(camera_pose):
    camera_names = camera_pose.keys()
    result = dict()
    for cam_name in camera_names:
        if cam_name == 'metadata':
            continue
        camera_intrinsics = np.array(camera_pose[cam_name]['camera_intrinsics'])
        camera_extrinsics = np.array(camera_pose[cam_name]['camera_extrinsics'])        
        camera_matrix = np.dot(camera_intrinsics, camera_extrinsics)        
        
        result[cam_name] = dict()
        result[cam_name]['camera_intrinsics'] = camera_intrinsics
        result[cam_name]['camera_extrinsics'] = camera_extrinsics
        result[cam_name]['camera_matrix'] = camera_matrix             
    return result

def process_camera_pose_new(camera_data, frame_number):
    camera_names = camera_data.keys()
    result = dict()    
    for cam_name in camera_names:        
        if cam_name == 'metadata':
            continue        
        if cam_name.find('aria')!=-1:
            camera_intrinsics = np.array(camera_data[cam_name]['camera_intrinsics'])
            camera_extrinsics = np.array(camera_data[cam_name]['camera_extrinsics'][frame_number])                        
        else:
            camera_intrinsics = np.array(camera_data[cam_name]['camera_intrinsics'])
            camera_extrinsics = np.array(camera_data[cam_name]['camera_extrinsics'])                
        camera_matrix = np.dot(camera_intrinsics, camera_extrinsics)

        result[cam_name] = dict() 
        result[cam_name]['camera_intrinsics'] = camera_intrinsics
        result[cam_name]['camera_extrinsics'] = camera_extrinsics
        result[cam_name]['camera_matrix'] = camera_matrix

        
    return result

def compute_3d_reconstruction_error(orig, pred):    
    err = np.sum([np.abs(orig[k]-pred[k]) for k in ['x', 'y', 'z']])
    print(orig, pred, err)    


def compute_2d_reconstruction_error(orig, projected):
    projection2d = dict()        
    for item in orig:
        key = item[-1]
        projection2d[key] = dict()
        projection2d[key]['type'] = 'manual'
        
        proj_2d = projected[key]
        projection2d[key]['projection'] = proj_2d
        
        orig_2d = dict()
        orig_2d['x'], orig_2d['y'] = item[0], item[1]
        projection2d[key]['orig'] = orig_2d        
        
        projection2d[key]['err'] = np.sum([np.abs(orig_2d[k]-proj_2d[k]) for k in ['x', 'y']])

    for key in projected:        
        if key not in projection2d:        
            projection2d[key] = dict()
            projection2d[key]['type'] = 'auto'
            projection2d[key]['projection'] = projected[key]    
    return projection2d
        

def run_triangulation(annotation, camera_matrices):
    output = dict()    
    pose2d_transformed = process_annotation(annotation['annotation2D'], camera_matrices)
    for kp_name in pose2d_transformed:        
        kp_data = pose2d_transformed[kp_name]            
        output[kp_name] = dict()
        output[kp_name]['num_views'] = len(kp_data)        
        annotation3D_new = triangulate(kp_data)
        if kp_name in annotation['annotation3D']:
            annotation3D_orig  = annotation['annotation3D'][kp_name]                             
            output[kp_name]['orig_3d'] = annotation3D_orig
            output[kp_name]['new_3d'] = annotation3D_new
            projected_points2D = project2d(annotation3D_new, camera_matrices)              
            projection2d = compute_2d_reconstruction_error(kp_data, projected_points2D)                         
            output[kp_name]['data_2d'] = projection2d
    return output
        

def triangulate_take(camera_dir, annotation_dir, camera_format, annotation_file):    
    camera_data = load_json(os.path.join(camera_dir, annotation_file))    
    annotation_data = load_json(os.path.join(annotation_dir, annotation_file))    
    frame_numbers = annotation_data.keys()
    for frame_number in frame_numbers:
        print(frame_number)
        annotation = annotation_data[frame_number][0]
        if camera_format=='old':
            camera_pose = camera_data[frame_number]
            camera_matrices = process_camera_pose(camera_pose)
        else:
            camera_matrices = process_camera_pose_new(camera_data, frame_number)            
                    
        output = run_triangulation(annotation, camera_matrices)
        print(json.dumps(output, indent=2))
        print('=='*10)
        break

def main():
    camera_dir = sys.argv[1]
    annotation_dir = sys.argv[2]
    camera_format = sys.argv[3]

    annotation_files = os.listdir(annotation_dir)
    for annotation_file in annotation_files[:5]:
        print(annotation_file)
        triangulate_take(camera_dir, annotation_dir, camera_format, annotation_file)
        break

if __name__ == "__main__":
    main()



