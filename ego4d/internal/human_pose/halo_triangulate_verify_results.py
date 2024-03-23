# python halo_triangulate_verify_results.py
from collections import defaultdict
import numpy as np
import json
import sys
import os
import math
def load_json(fname):
    with open(fname) as f:
        data = json.load(f)    
    return data

def match_keypoints3d(orig, trans):
    errors = list()
    for kp_name in orig:
        if kp_name in trans:
            pt_orig  = orig[kp_name]
            pt_trans = trans[kp_name]            
            err = np.sum([np.abs(pt_orig[k]-pt_trans[k]) for k in ['x', 'y', 'z']])           
            errors.append(err)
    if len(errors)==0:
        return None
    
    mean_error = np.mean(np.array(errors))
    return mean_error
            
def compare_results(annotation_json_path, output_json_path):
    original = load_json(annotation_json_path)
    transfomred = load_json(output_json_path)
    frame_numbers = original.keys()
    all_errors = list()    
    for frame_number in frame_numbers:
        if len(original[frame_number])==1:
            original_annot = original[frame_number][0]
            transfomred_annot = transfomred[frame_number][0]
            mean_error = match_keypoints3d(original_annot['annotation3D'], transfomred_annot['annotation3D'])
            if mean_error is not None:
                all_errors.append(mean_error)
    

    final_error = np.mean(np.array(all_errors))
    if final_error>0.001:
        print(final_error)        
    
def main():    
    config_file = "halo_triangulate_config_old.json"    
    #config_file = "halo_triangulate_config_new.json"    
    config = load_json(config_file)
    annotation_type = config["annotation_type"]
    body_or_hand = config["body_or_hand"]
    camera_format = config["camera_format"]
    
    annotation_dir = os.path.join(config["annotation_dir"], body_or_hand, annotation_type)
    output_dir =  os.path.join(config["output_dir"], camera_format, body_or_hand, annotation_type)    
    annotation_files = os.listdir(annotation_dir)    
    print('Num annotation files:', len(annotation_files))    
    
    for idx, annotation_file in enumerate(annotation_files):        
        annotation_json_path = os.path.join(annotation_dir, annotation_file)
        output_json_path = os.path.join(output_dir, annotation_file)
        print(idx, annotation_file)        
        if os.path.exists(output_json_path):
            error = compare_results(annotation_json_path, output_json_path)
            
if __name__ == "__main__":
    main()





