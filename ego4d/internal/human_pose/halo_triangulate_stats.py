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
    output_json_path = os.path.join(output_dir, annotation_file)
    error_json_path  = os.path.join(error_output_dir, annotation_file)
    
    if not os.path.exists(output_json_path):                
        return 0, 0, 0

    output = load_json(output_json_path)    
    
    frame_numbers = output.keys()     
    num_3d, num_2d = 0, 0
    for frame_number in frame_numbers:             
        for annot in output[frame_number]:
            num_3d+=1
            num_2d+=len(annot['annotation2D'])

    return len(frame_numbers), num_3d, num_2d

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
    
    annotation_files = os.listdir(annotation_dir)    
    print('Num annotation files:', len(annotation_files))  

    params = dict()
    params["output_dir"] = output_dir
    params["error_output_dir"] = error_output_dir
    params["camera_dir"] = camera_dir
    params["annotation_dir"] = annotation_dir
    params["camera_format"] = camera_format    

    total_takes, total_frames, total_3d, total_2d = 0, 0, 0, 0
    for idx, annotation_file in enumerate(annotation_files):    
        num_frames, num_3d, num_2d = run_pipeline(params, annotation_file)
        print(idx, num_frames, num_3d, num_2d)
        if num_frames>0:
            total_takes+=1
            total_frames+=num_frames
            total_3d+=num_3d
            total_2d+=num_2d
    
    print(total_takes, total_frames, total_3d, total_2d)
    
if __name__ == "__main__":
    main()

# Hands
# Annotation: 458 61499 67779 340500
# Automatic: 976 5079777 4381700 21901592
    
# Body    
# Annotation: 1358 289219 376582 2006610
# Automatic: 

