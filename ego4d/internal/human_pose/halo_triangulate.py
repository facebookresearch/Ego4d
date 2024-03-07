# This is stored in the new camera format
# python halo_triangulate.py /large_experiments/egoexo/egopose/suyogjain/project_retriangulation_production/camera_pose/ /large_experiments/egoexo/egopose/suyogjain/project_retriangulation_production/ego_pose_latest/hand/annotation/ new

from collections import defaultdict
import numpy as np
import json
import sys
import os

def load_json(fname):
    with open(fname) as f:
        data = json.load(f)    
    return data


def write_json(data, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file)

def triangulate(camera_calibrations):
    # Check if input is valid
    if not camera_calibrations or len(camera_calibrations) < 2:
        print('Missing camera calibrations')
        return None
    for calibration in camera_calibrations:        
        if not calibration or len(calibration) != 4:
            return None
        #x, y, M, cam_name = calibration        
        #if not (isinstance(x, float) and isinstance(y, float) and isinstance(M, np.ndarray)):            
        #    return None
    
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

def summarize_error(results, level=0):    
    frame_numbers = list(results.keys())
    overall_errors = defaultdict(list) 
    for frame_number in frame_numbers:
        num_raters = len(results[frame_number])
        for rater_idx in range(num_raters):
            result = results[frame_number][rater_idx]
            errors = defaultdict(list) 
            for kp_name in result:
                data_2d = result[kp_name]['data_2d']
                for cam_name in data_2d:
                    if 'err' in data_2d[cam_name]:
                        errors[cam_name].append(data_2d[cam_name]['err'])
                        overall_errors[cam_name].append(data_2d[cam_name]['err'])
            
            if level==1:
                for cam_name in errors:
                    print(cam_name, np.mean(np.array(errors[cam_name])))
    
    for cam_name in overall_errors:
        print(cam_name, len(overall_errors[cam_name]), np.mean(np.array(overall_errors[cam_name])))
    
def generate_output_json(results):
    transformed_annotation = dict()
    frame_numbers = list(results.keys())    
    for frame_number in frame_numbers:
        num_raters = len(results[frame_number])
        frame_results = list()
        for rater_idx in range(num_raters):
            result = results[frame_number][rater_idx]
            frame_result = dict()
            frame_result['annotation3D'] = dict()
            
            for kp_name in result:                
                frame_result['annotation3D'][kp_name] = result[kp_name]['new_3d'] 
                frame_result['annotation3D'][kp_name]['num_views_for_3d'] = result[kp_name]['num_views'] 

            frame_result['annotation2D'] = dict()
            for kp_name in result:                
                for camera_name in result[kp_name]['data_2d']:
                    if camera_name not in frame_result['annotation2D']:
                        frame_result['annotation2D'][camera_name] = dict()                    
                
            for kp_name in result:
                data_2d = result[kp_name]['data_2d']                
                for camera_name in data_2d:
                    pt_type = data_2d[camera_name]['type']                    
                    if pt_type == 'manual':
                        x_coordinate = data_2d[camera_name]['orig']['x']
                        y_coordinate = data_2d[camera_name]['orig']['y']
                        placement = 'manual'                        
                    else:
                        x_coordinate = data_2d[camera_name]['projection']['x']
                        y_coordinate = data_2d[camera_name]['projection']['y']
                        placement = 'auto'
                  
                    frame_result['annotation2D'][camera_name][kp_name] = dict()
                    frame_result['annotation2D'][camera_name][kp_name]['x'] = x_coordinate
                    frame_result['annotation2D'][camera_name][kp_name]['y'] = y_coordinate
                    frame_result['annotation2D'][camera_name][kp_name]['placement'] = placement
                                                                                   
            frame_results.append(frame_result)

        transformed_annotation[frame_number] = frame_results
        
    return transformed_annotation

                

def run_triangulation(annotation, camera_matrices):
    output = dict()    
    pose2d_transformed = process_annotation(annotation['annotation2D'], camera_matrices)
    for kp_name in pose2d_transformed:        
        kp_data = pose2d_transformed[kp_name]
        
        if len(kp_data)<2: # Cannot triangulate
            continue
        
        if kp_name in annotation['annotation3D']:
            output[kp_name] = dict()
            output[kp_name]['num_views'] = len(kp_data)        
            annotation3D_orig  = annotation['annotation3D'][kp_name]                             
            output[kp_name]['orig_3d'] = annotation3D_orig
            
            annotation3D_new = triangulate(kp_data)
            output[kp_name]['new_3d'] = annotation3D_new
                    
            projected_points2D = project2d(annotation3D_new, camera_matrices)              
            projection2d = compute_2d_reconstruction_error(kp_data, projected_points2D)                         
            output[kp_name]['data_2d'] = projection2d        
    return output
        

def triangulate_take(camera_dir, annotation_dir, camera_format, annotation_file):    
    camera_data = load_json(os.path.join(camera_dir, annotation_file))    
    annotation_data = load_json(os.path.join(annotation_dir, annotation_file))    
    frame_numbers = annotation_data.keys()
    output = dict()
    for frame_number in frame_numbers:
        #print(frame_number)
        num_raters = len(annotation_data[frame_number])
        output[frame_number] = list()
        for rater_idx in range(num_raters):
            annotation = annotation_data[frame_number][rater_idx]        
            if camera_format=='old':
                camera_pose = camera_data[frame_number]
                camera_matrices = process_camera_pose(camera_pose)
            else:
                camera_matrices = process_camera_pose_new(camera_data, frame_number)            
                        
            output[frame_number].append(run_triangulation(annotation, camera_matrices))
    
    #print(json.dumps(output, indent=2))
    summarize_error(output, level=0)
    output_json = generate_output_json(output)    
    print('=='*10)
    return output_json
        
def main():    
    config_file = "halo_triangulate_config.json"
    config = load_json(config_file)
    annotation_type = config["annotation_type"]
    body_or_hand = config["body_or_hand"]
    camera_format = config["camera_format"]
    
    camera_dir = config["camera_dir"]
    annotation_dir = os.path.join(config["annotation_dir"], body_or_hand, annotation_type)
    output_dir =  os.path.join(config["output_dir"], camera_format, body_or_hand, annotation_type)
    if not os.path.isdir(output_dir):
        cmd = 'mkdir -p ' + output_dir
        os.system(cmd)

    annotation_files = os.listdir(annotation_dir)
    for annotation_file in annotation_files[:5]:
        print(annotation_file)
        output_json = triangulate_take(camera_dir, annotation_dir, camera_format, annotation_file)
        output_json_path = os.path.join(output_dir, annotation_file)
        print(output_json_path)
        write_json(output_json, output_json_path)        

if __name__ == "__main__":
    main()





