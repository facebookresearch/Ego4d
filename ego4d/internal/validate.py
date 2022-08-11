import csv
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ego4d.cli.manifest import list_videos_in_manifest, VideoMetadata
from ego4d.cli.universities import BUCKET_TO_UNIV, UNIV_TO_BUCKET
from ego4d.internal.ffmpeg_utils import get_video_info, VideoInfo
from ego4d.internal.university_files import *
import logging
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import boto3
import botocore
from botocore.exceptions import ClientError
from ego4d.internal.credential_s3 import S3Helper
from tqdm import tqdm

lock = Lock()

"""
 --validate: local path or an s3 path, local so someone can iterate on theirs files on their machine.

--all: check all of the latest files for each university
"""

control_file_location = "ego4d_fair/tree/ingestion/configs/active_university_load_config_PRODUCTION_LATEST.csv"

# placeholder for standard metadata folder, will replace after uploading to S3
standard_metadata_folder = "./standard_metadata_v10"


def _validate_video_components(
    s3,
    bucket_name,
    video_metadata_dict: Dict[str, VideoMetadata],
    video_components_dict: Dict[str, List[VideoComponentFile]],
    error_message: List[ErrorMessage],
) -> List[Tuple[str, str]]:
    """
    Args:
        video_components_dict: Dict[str, List[VideoComponentFile]]:
        mapping from university_video_id to a list of all VideoComponentFile
        objects that have equal values at university_video_id

        error_message: List[ErrorMessage]: a list to store ErrorMessage objects
        generated when validating video components.

    Returns:
        List[Tuple[str, str]] a list containing a tuple for each
        video component's university_video_id and key
    """
    video_info_param = []
    for video_id, components in video_components_dict.items():
        if video_id not in video_metadata_dict:
            error_message.append(
                ErrorMessage(
                    video_id,
                    "Error",
                    "video_component_file points to video_metadata at"
                    " non-existent university_video_id",
                )
            )
            continue
        elif video_metadata_dict[video_id].number_video_components != len(components):
            error_message.append(
                ErrorMessage(
                    video_id,
                    "Error",
                    f"video_metadata has {len(components)} components when it"
                    f" should have {video_metadata_dict[video_id].number_video_components}",
                )
            )
        components.sort(key=lambda x: x.component_index)
        # check component_index is incremental and starts at 0
        university_video_folder_path = video_metadata_dict[
            video_id
        ].university_video_folder_path
        for i in range(len(components)):
            if components[i].component_index != i:
                error_message.append(
                    ErrorMessage(
                        video_id,
                        "Error",
                        f"video_metadata has component index {components[i].component_index}"
                        f" when it should have {i}",
                    )
                )

            s3_path = f"{university_video_folder_path.rstrip('/')}/{components[i].video_component_relative_path}"
            bucket, key = split_s3_path(s3_path)
            if bucket != bucket_name:
                error_message.append(
                    ErrorMessage(
                        video_id, "Error", "video_metadata has wrong bucket name"
                    )
                )

            try:
                s3.head_object(Bucket=bucket, Key=key)
                video_info_param.append((video_id, key))
            except ClientError as e:
                error_message.append(
                    ErrorMessage(
                        video_id,
                        "Error",
                        f"video s3://{bucket}/{key} doesn't exist in bucket",
                    )
                )

    return video_info_param


def _get_videos(
    s3,
    bucket_name: str,
    video_info_param: List[Tuple[str, str]],
    error_message: List[ErrorMessage],
) -> Dict[str, List[VideoInfo]]:
    """
    Args:
        video_info_param: List[Tuple[str, str]]: a list containing tuples for each
        video component's university_video_id and key

    Returns:
        Dict[str, List[VideoInfo]]: mapping from university_video_id
        to a list of all VideoInfo objects that have equal values at university_video_id

    """
    video_info_dict = defaultdict(list)
    # check component mp4 exist in S3
    def thread_helper(x):
        video_info = get_video_info(s3, bucket_name, x, error_message)
        if video_info != None:
            lock.acquire()
            video_info_dict[x[0]].append(video_info)
            lock.release()

    with ThreadPoolExecutor(max_workers=15) as pool:
        tqdm(pool.map(thread_helper, video_info_param), total=len(video_info_param))
    return video_info_dict


def _validate_mp4(
    video_info_dict: Dict[str, List[VideoInfo]], error_message: List[ErrorMessage]
) -> None:
    """
    This function checks a set of MP4 files on S3 for the following properties:
    - null fps
    - consistent width/height
    - rotation consistent
    - video time base consistent
    - video codec consistent
    - audio codec consistent
    - mp4 duration too large or small
    - mp4 duration does not exist
    - video fps consistent
    - SAR consistent
    - FPS consistent

    Args:
        video_info_dict: Dict[str, List[VideoInfo]]: mapping from university_video_id
        to a list of all VideoInfo objects that have equal values at university_video_id

        error_message: List[ErrorMessage]: a list to store ErrorMessage objects
        generated when validating mp4 files in s3.
    """
    for video_id, video_infos in video_info_dict.items():
        total_vcodec = set()
        total_acodec = set()
        total_rotate = set()
        total_dimension = set()
        total_size = set()
        total_vtb = set()
        total_sar = set()
        total_fps = set()

        for i, video_info in enumerate(video_infos):
            # check consistent video codec
            if video_info.vcodec != None:
                total_vcodec.add(video_info.vcodec)

            # check consistent audio time base
            if video_info.acodec != None:
                total_acodec.add(video_info.acodec)

            # check consistent rotation
            if video_info.rotate != None:
                total_rotate.add(video_info.rotate)
            else:
                error_message.append(
                    ErrorMessage(
                        video_id,
                        "Warning",
                        f"component {i} has no rotation information",
                    )
                )

            # check consistent width and height
            if video_info.sample_width != None and video_info.sample_height != None:
                total_size.add((video_info.sample_width, video_info.sample_height))

                if (
                    video_info.sample_width < video_info.sample_height
                    and video_info.rotate == None
                ):
                    error_message.append(
                        ErrorMessage(
                            video_id,
                            "Error",
                            f"component {i} has width < height without rotation",
                        )
                    )

            # check consistent video time base
            if video_info.video_time_base != None:
                total_vtb.add(video_info.video_time_base)

            # check consistent sar
            if video_info.sar != None:
                total_sar.add(video_info.sar)

            # check null/inconsistent video fps
            if video_info.fps == None:
                error_message.append(
                    ErrorMessage(
                        video_id, "Warning", f"component {i} has null fps value"
                    )
                )
            else:
                total_fps.add(video_info.fps)

            # check null mp4 duration
            if video_info.mp4_duration == None:
                error_message.append(
                    ErrorMessage(
                        video_id, "Warning", f"component {i} has no mp4 duration"
                    )
                )
            else:
                video_length = (
                    video_info.vstart + video_info.vduration
                    if video_info.vstart != None
                    else video_info.vduration
                )
                audio_length = (
                    video_info.astart + video_info.aduration
                    if video_info.astart != None
                    else video_info.aduration
                )
                delta = abs(max(video_length, audio_length) - video_info.mp4_duration)
                if delta > 0.5:
                    error_message.append(
                        ErrorMessage(
                            video_id,
                            "Warning",
                            f"component {i} has mismatching mp4 duration",
                        )
                    )

        if len(total_vcodec) > 1:
            error_message.append(
                ErrorMessage(video_id, "Error", "inconsistent video codec")
            )
        if len(total_acodec) > 1:
            error_message.append(
                ErrorMessage(video_id, "Error", "inconsistent audio codec")
            )
        if len(total_rotate) > 1:
            error_message.append(
                ErrorMessage(video_id, "Error", "inconsistent rotation")
            )
        if len(total_size) > 1:
            error_message.append(
                ErrorMessage(
                    video_id, "Error", "components with inconsistent width x height"
                )
            )
        if len(total_vtb) > 1:
            error_message.append(
                ErrorMessage(video_id, "Error", "inconsistent video time base")
            )
        if len(total_sar) > 1:
            error_message.append(ErrorMessage(video_id, "Warning", "inconsistent sar"))
        if len(total_fps) > 1:
            error_message.append(
                ErrorMessage(video_id, "Warning", "inconsistent video fps")
            )


def _validate_synchronized_videos(
    video_metadata_dict: Dict[str, VideoMetadata],
    synchronized_video_dict: Dict[str, SynchronizedVideos],
    error_message: List[ErrorMessage],
) -> None:
    """
    Args:
        synchronized_video_dict: Dict[str, SynchronizedVideos]: mapping from
        video_grouping_id  to a list of all SynchronizedVideos objects that
        have equal values at video_grouping_id

        error_message: List[ErrorMessage]: a list to store ErrorMessage objects
        generated when validating synchronized_videos.csv.
    """
    if synchronized_video_dict:
        for video_grouping_id, components in synchronized_video_dict.items():
            for component in components:
                for video_id, num in component.associated_videos:
                    if video_id not in video_metadata_dict:
                        error_message.append(
                            ErrorMessage(
                                video_id,
                                "Error",
                                "synchronized_video_file points to video_metadata at non-existent university_video_id",
                            )
                        )


def _validate_auxilliary_videos(
    video_metadata_dict: Dict[str, VideoMetadata],
    auxiliary_video_component_dict: Dict[str, List[AuxiliaryVideoComponentDataFile]],
    component_types: Dict[str, ComponentType],
    error_message: List[ErrorMessage],
) -> None:
    """
    Args:
        auxiliary_video_component_dict: Dict[str, List[AuxiliaryVideoComponentDataFile]]:
        mapping from university_video_id to a list of all AuxiliaryVideoComponentDataFile
        objects that have equal values at university_video_id

        error_message: List[ErrorMessage]: a list to store ErrorMessage objects
        generated when validating auxilliary_videos.csv.
    """
    # Check ids in auxiliary_video_component_dict are in video_metadata_dict
    # and that the component_type is valid
    if auxiliary_video_component_dict:
        for video_id, components in auxiliary_video_component_dict.items():
            if video_id not in video_metadata_dict:
                error_message.append(
                    ErrorMessage(
                        video_id,
                        "Error",
                        "auxiliary_video_component_data_file points to video_metadata at"
                        " non-existent university_video_id",
                    )
                )
            elif video_metadata_dict[video_id].number_video_components != len(
                components
            ):
                error_message.append(
                    ErrorMessage(
                        video_id,
                        "Error",
                        f"video_metadata has {len(components)} auxiliary components when it"
                        f" should have {video_metadata_dict[video_id].number_video_components}",
                    )
                )
            components.sort(key=lambda x: x.component_index)
            for i in range(len(components)):
                component = components[i]
                if i != component.component_index:
                    error_message.append(
                        ErrorMessage(
                            video_id,
                            "Error",
                            f"video_metadata has auxiliary component index {component.component_index}"
                            f" when it should have {i}",
                        )
                    )
                if component.component_type_id not in component_types:
                    error_message.append(
                        ErrorMessage(
                            video_id,
                            "Error",
                            f"auxillary component's component_type_id: '{component.component_type_id} not exist in component_types'",
                        )
                    )


def _validate_participant(
    video_metadata_dict: Dict[str, VideoMetadata],
    participant_dict: Dict[str, Particpant],
    error_message: List[ErrorMessage],
) -> None:
    """
    Args:
        participant_dict: Dict[str, Particpant]: mapping from participant_id
        to Participant objects that have equal values at participant_id

        error_message: List[ErrorMessage]: a list to store ErrorMessage objects
        generated when validating participants.csv.
    """
    if participant_dict:
        for participant_id, components in participant_dict.items():
            if participant_id not in video_metadata_dict:
                error_message.append(
                    ErrorMessage(
                        video_id,
                        "Error",
                        f" participant '{participant_id}' not exist in "
                        " video metadata",
                    )
                )


def _validate_annotations(
    video_metadata_dict: Dict[str, VideoMetadata],
    annotations_dict: Dict[str, Annotations],
    error_message: List[ErrorMessage],
) -> None:
    """
    Args:
        annotations_dict: Dict[str, Annotations]: mapping from participant_id
        to Participant objects that have equal values at participant_id

        error_message: List[ErrorMessage]: a list to store ErrorMessage objects
        generated when validating participants.csv.
    """

    if annotations_dict:
        for video_id in annotations_dict:
            if video_id not in video_metadata_dict:
                error_message.append(
                    ErrorMessage(
                        video_id,
                        "annotations points to video_metadata at"
                        " non-existent university_video_id",
                    )
                )


def _validate_video_metadata(
    video_metadata_dict: Dict[str, VideoMetadata],
    video_components_dict: Dict[str, List[VideoComponentFile]],
    participant_dict: Dict[str, Particpant],
    physical_setting_dict: Dict[str, PhysicalSetting],
    devices: Dict[str, Device],
    component_types: Dict[str, ComponentType],
    scenarios: Dict[str, Scenario],
    error_message: List[ErrorMessage],
) -> None:
    """
    Args:
        video_metadata_dict: Dict[str, VideoMetadata]: mapping from
        university_video_id to a list of all VideoMetaData objects that
        have equal values at university_video_id

        video_components_dict: Dict[str, List[VideoComponentFile]]:
        mapping from university_video_id to a list of all VideoComponentFile
        objects that have equal values at university_video_id

        participant_dict: Dict[str, Particpant]: mapping from participant_id
        to Participant objects that have equal values at participant_id

        physical_setting_dict: Dict[str, PhysicalSetting],
        devices: Dict[str, Device],
        component_types: Dict[str, ComponentType],
        scenarios: Dict[str, Scenario],

        error_message: List[ErrorMessage]: a list to store ErrorMessage objects
        generated when validating video_metadata.csv.
    """
    # Check from video_metadata:
    video_ids = set()
    for video_id, video_metadata in video_metadata_dict.items():
        #   0. check no duplicate university_video_id
        if video_id in video_ids:
            error_message.append(
                ErrorMessage(video_id, "Error", f"duplicate video id in metadata")
            )
        video_ids.add(video_id)
        #   1. participant_ids are in participant_dict
        if participant_dict:
            if video_metadata.recording_participant_id not in participant_dict:
                error_message.append(
                    ErrorMessage(
                        video_id,
                        "Error",
                        f"recording_participant_id '{video_metadata.recording_participant_id}' not in participant_dict",
                    )
                )
            if video_metadata.recording_participant_id is None:
                error_message.append(
                    ErrorMessage(video_id, "Warning", "empty participant_id")
                )

        #   2. scenario_ids are in scenarios
        for scenario_id in video_metadata.video_scenario_ids:
            if scenario_id not in scenarios:
                error_message.append(
                    ErrorMessage(
                        video_id,
                        "Error",
                        f"video_scenario_id: '{scenario_id}' not in scenarios.csv",
                    )
                )

        #   3. device_ids are in devices
        if video_metadata.device_id and video_metadata.device_id not in devices:
            error_message.append(
                ErrorMessage(
                    video_id,
                    "Error",
                    f"device_id '{video_metadata.device_id}' not in devices.csv",
                )
            )

        #   4. physical_settings are in physical_setting
        if physical_setting_dict:
            if (
                video_metadata.physical_setting_id
                and video_metadata.physical_setting_id not in physical_setting_dict
            ):
                error_message.append(
                    ErrorMessage(
                        video_id,
                        "Error",
                        f"physical_setting_id '{video_metadata.physical_setting_id}' not in physical_setting.csv",
                    )
                )

        #   5. university_video_ids are in components
        if video_id not in video_components_dict:
            error_message.append(
                ErrorMessage(
                    video_id,
                    "Error",
                    f"Video: {video_id} points to component at"
                    " non-existent university_video_id: ",
                )
            )


def validate_university_files(  # noqa :C901
    video_metadata_dict: Dict[str, VideoMetadata],
    video_components_dict: Dict[str, List[VideoComponentFile]],
    auxiliary_video_component_dict: Dict[str, List[AuxiliaryVideoComponentDataFile]],
    participant_dict: Dict[str, Particpant],
    synchronized_video_dict: Dict[str, SynchronizedVideos],
    physical_setting_dict: Dict[str, PhysicalSetting],
    annotations_dict: Dict[str, Annotations],
    devices: Dict[str, Device],
    component_types: Dict[str, ComponentType],
    scenarios: Dict[str, Scenario],
    bucket_name: str,
    s3: botocore.client.BaseClient,
) -> List[ErrorMessage]:
    error_message = []
    # Check ids in video_components_dict are in video_metadata_dict
    # and the # of components is correct
    s3_bucket = S3Helper(s3, bucket_name)
    video_info_param = []

    video_info_param = _validate_video_components(
        s3, bucket_name, video_metadata_dict, video_components_dict, error_message
    )
    video_info_dict = _get_videos(s3, bucket_name, video_info_param, error_message)
    _validate_mp4(video_info_dict, error_message)
    _validate_synchronized_videos(
        video_metadata_dict, synchronized_video_dict, error_message
    )
    _validate_auxilliary_videos(
        video_metadata_dict,
        auxiliary_video_component_dict,
        component_types,
        error_message,
    )
    _validate_participant(video_metadata_dict, participant_dict, error_message)
    _validate_annotations(video_metadata_dict, annotations_dict, error_message)
    _validate_video_metadata(
        video_metadata_dict,
        component_types,
        participant_dict,
        scenarios,
        devices,
        physical_setting_dict,
        video_components_dict,
        error_message,
    )

    if error_message:
        print(error_message)

    fields = ["univeristy_video_id", "errorType", "description"]
    with open("error_message", "w") as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerow(fields)
        for e in error_message:
            write.writerow([e.uid, e.errorType, e.description])
    return error_message


def validate_all(path, s3):
    bucket, path = split_s3_path(path)
    print(bucket, path)

    # get access to metadata_folder
    devices, component_types, scenarios = load_standard_metadata_files(
        standard_metadata_folder
    )

    (
        video_metadata_dict,
        video_components_dict,
        auxiliary_video_component_dict,
        participant_dict,
        synchronized_video_dict,
        physical_setting_dict,
        annotations_dict,
    ) = load_university_files(s3, bucket, path)

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
        s3,
    )
