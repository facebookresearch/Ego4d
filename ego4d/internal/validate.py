import logging
import os
from typing import Dict, List, Tuple
import csv
import tempfile
from pathlib import Path

from ego4d.cli.manifest import list_videos_in_manifest, VideoMetadata
from ego4d.cli.universities import BUCKET_TO_UNIV, UNIV_TO_BUCKET
from ffmpeg_utils import VideoInfo, create_presigned_url,get_video_info
from university_files import load_standard_metadata_files, load_university_files, validate_university_files, split_s3_path

import boto3
import tqdm
from collections import defaultdict


'''
 --validate: local path or an s3 path, local so someone can iterate on theirs files on their machine.

--all: check all of the latest files for each university
'''

control_file_location = (
    "ego4d_fair/tree/ingestion/configs/active_university_load_config_PRODUCTION_LATEST.csv"
)

# placeholder for standard metadata folder, will replace after uploading to S3
standard_metadata_folder = "./standard_metadata_v10"

def validate(path, s3):
    bucket, path = split_s3_path(path)
    
    # get access to metadata_folder 
    devices, component_types, scenarios = load_standard_metadata_files(
        standard_metadata_folder
    )

    # helper = s3.Bucket(bucket_name)
    (
        video_metadata_dict,
        video_components_dict,
        auxiliary_video_component_dict,
        participant_dict,
        synchronized_video_dict,
        physical_setting_dict,
        annotations_dict,
    ) = load_university_files(
        s3,
        bucket,
        path
    )

    validate_university_files(
        video_metadata_dict,
        video_components_dict,
        auxiliary_video_component_dict,
        participant_dict,
        synchronized_video_dict,
        physical_setting_dict,
        annotations_dict,
        devices,
        component_types,
        scenarios,
        bucket,
        s3
    )
