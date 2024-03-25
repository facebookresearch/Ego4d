from collections import defaultdict
import numpy as np
import json
import sys
import os
import submitit
import functools
def load_json(fname):
    with open(fname) as f:
        data = json.load(f)    
    return data

def write_json(data, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file)

def run_pipeline(params, annotation_file):
    output_dir = params["output_dir"]
    error_output_dir = params["error_output_dir"]
    release_dir = params["release_dir"]
        
    output_json_path = os.path.join(output_dir, annotation_file)
    error_json_path  = os.path.join(error_output_dir, annotation_file)
    release_json_path = os.path.join(release_dir, annotation_file)
    
    if not os.path.exists(output_json_path):        
        return

    error_json = load_json(error_json_path)    
    error_cams = 0
    for cam_name in error_json:
        cam_frames, cam_error = error_json[cam_name]['num_frames'], error_json[cam_name]['error']
        #print(annotation_file, cam_name, cam_frames, cam_error)
        if cam_frames>100 and cam_error>50:
            error_cams+=1

    if len(error_json) <=1:
        print(annotation_file, "Too few cameras")        
    elif error_cams>2:
        print(annotation_file, "Large erors")
    else:
        cmd= 'cp ' + output_json_path + ' ' + release_json_path
        os.system(cmd)
    


def main():    
    config_file = "halo_triangulate_config_new.json"
    
    config = load_json(config_file)
    annotation_type = config["annotation_type"]
    body_or_hand = config["body_or_hand"]
    camera_format = config["camera_format"]
    
    camera_dir = config["camera_dir"]
    annotation_dir = os.path.join(config["annotation_dir"], body_or_hand, annotation_type)
    output_dir =  os.path.join(config["output_dir"], camera_format, body_or_hand, annotation_type)
    error_output_dir =  os.path.join(config["error_output_dir"], camera_format, body_or_hand, annotation_type)
    release_dir =  os.path.join(config["release_dir"], body_or_hand, annotation_type)

    if not os.path.isdir(release_dir):
        cmd = 'mkdir -p ' + release_dir
        os.system(cmd)

    annotation_files = os.listdir(annotation_dir)    
    print('Num annotation files:', len(annotation_files))  

    params = dict()
    params["output_dir"] = output_dir
    params["error_output_dir"] = error_output_dir
    params["camera_dir"] = camera_dir
    params["annotation_dir"] = annotation_dir
    params["camera_format"] = camera_format
    params["release_dir"] = release_dir

    for idx, annotation_file in enumerate(annotation_files):
        #print(idx, len(annotation_files), annotation_file)        
        run_pipeline(params, annotation_file)
        
if __name__ == "__main__":
    main()





