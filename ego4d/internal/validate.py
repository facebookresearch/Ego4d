import collections
import csv
import functools
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

import botocore
from botocore.exceptions import ClientError

from ego4d.cli.manifest import VideoMetadata
from ego4d.internal.ffmpeg_utils import get_video_info, VideoInfo
from ego4d.internal.university_files import (
    Annotations,
    AuxiliaryVideoComponentDataFile,
    ComponentType,
    Device,
    ErrorMessage,
    load_standard_metadata_files,
    load_university_files,
    Particpant,
    PhysicalSetting,
    Scenario,
    split_s3_path,
    SynchronizedVideos,
    VideoComponentFile,
)
from tqdm import tqdm

"""
-i: The S3 path where the metadata is stored
-a: check all of the latest files for each university
-mf: "./ego4d/internal/standard_metadata_v10"
-ed: "error_details"
-es: "error_summary"
"""


def _validate_vcs(
    video_id_component_pair: Tuple[str, List[VideoComponentFile]],
    video_metadata_dict: Dict[str, VideoMetadata],
    bucket_name: str,
    s3,
) -> Tuple[List[Tuple[str, str]], List[ErrorMessage]]:
    errors = []
    result = []

    video_id, components = video_id_component_pair
    components.sort(key=lambda x: x.component_index)

    if video_id not in video_metadata_dict:
        errors.append(
            ErrorMessage(
                video_id,
                "video_not_found_in_video_metadata_error",
                f"{video_id} in video_components_dict can't be found in video_metadata",
            )
        )
        return result, errors
    elif video_metadata_dict[video_id].number_video_components != len(components):
        errors.append(
            ErrorMessage(
                video_id,
                "video_component_length_inconsistent_error",
                f"the video has {len(components)} components when it"
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
            errors.append(
                ErrorMessage(
                    video_id,
                    "video_component_wrong_index_error",
                    f"the video component has index {components[i].component_index}"
                    f" when it should have {i}",
                )
            )
        if (
            not components[i].video_component_relative_path
            and not components[i].is_redacted
        ):
            errors.append(
                ErrorMessage(
                    video_id,
                    "empty_video_component_relative_path_error",
                    f"Found an empty relative path for the video_id {video_id}",
                )
            )
        else:
            if not components[i].is_redacted:
                s3_path = f"{university_video_folder_path.rstrip('/')}/{components[i].video_component_relative_path}"
                bucket, key = split_s3_path(s3_path)
                if bucket != bucket_name:
                    errors.append(
                        ErrorMessage(
                            video_id,
                            "bucket_name_inconsistent_error",
                            f"video has bucket_name {bucket_name}"
                            f" when it should have {bucket}",
                        )
                    )
                try:
                    s3.head_object(Bucket=bucket, Key=key)
                    result.append((video_id, key))
                except ClientError:
                    errors.append(
                        ErrorMessage(
                            video_id,
                            "path_does_not_exist_error",
                            f"video s3://{bucket}/{key} doesn't exist in bucket",
                        )
                    )

    return result, errors


def _validate_video_components(
    s3,
    bucket_name: str,
    video_metadata_dict: Dict[str, VideoMetadata],
    video_components_dict: Dict[str, List[VideoComponentFile]],
    error_message: List[ErrorMessage],
    num_workers: int,
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
    print("validating video components")
    map_fn = functools.partial(
        _validate_vcs,
        video_metadata_dict=video_metadata_dict,
        bucket_name=bucket_name,
        s3=s3,
    )
    video_info_param = []
    with ThreadPoolExecutor(num_workers) as pool:
        vals_to_map = list(video_components_dict.items())
        for res, errs in tqdm(pool.map(map_fn, vals_to_map), total=len(vals_to_map)):
            video_info_param.extend(res)
            error_message.extend(errs)

    return video_info_param


def _get_presigned_url(
    video_param: Tuple[str, str],
    s3_client: botocore.client.BaseClient,
    bucket_name: str,
    expiration: int,
):
    errors = []
    object_name = video_param[1]
    try:
        filename = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": object_name},
            ExpiresIn=expiration,
        )
        return filename, errors
    except ClientError as ex:
        errors.append(
            ErrorMessage(
                video_param[0],
                "video_does_not_exist_in_bucket_error",
                f"video s3://{bucket_name}/{object_name} doesn't exist in bucket - {ex}",
            )
        )
        return None, errors


def _check_video(x):
    video_info, errs = get_video_info(x[1], name=x[0][1])
    return x[0], video_info, errs


def _get_videos(
    s3,
    bucket_name: str,
    video_info_param: List[Tuple[str, str]],
    error_message: List[ErrorMessage],
    num_workers: int,
    expiry_time_sec: int,
) -> Dict[str, List[VideoInfo]]:
    """
    Args:
        video_info_param: List[Tuple[str, str]]: a list containing tuples for each
        video component's university_video_id and key

    Returns:
        Dict[str, List[VideoInfo]]: mapping from university_video_id
        to a list of all VideoInfo objects that have equal values at university_video_id

    """
    print("loading and validating videos")
    video_info_dict = defaultdict(list)

    print("Generating presigned urls", flush=True)
    # NOTE: s3 is not thread safe, doing this just incase
    fps = [
        (
            x,
            _get_presigned_url(
                video_param=x,
                s3_client=s3,
                bucket_name=bucket_name,
                expiration=expiry_time_sec,
            ),
        )
        for x in tqdm(video_info_param)
    ]
    for _, (_, errs) in fps:
        if len(errs) > 0:
            error_message.extend(errs)

    fps = [(x, filename) for x, (filename, errs) in fps if len(errs) == 0]

    print("Checking URLs", flush=True)
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for x, video_info, errs in tqdm(
            pool.map(_check_video, fps),
            total=len(fps),
        ):
            video_info_dict[x[0]].append(video_info)
            error_message.extend(errs)

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
    print("validating mp4")
    for video_id, video_infos in tqdm(video_info_dict.items()):
        total_vcodec = set()
        total_acodec = set()
        total_rotate = set()
        total_size = set()
        total_vtb = set()
        total_sar = set()
        total_fps = set()

        for i, video_info in enumerate(video_infos):
            if video_info is None:
                error_message.append(
                    ErrorMessage(
                        video_id,
                        "no_video_info_for_component",
                        f"component {i} has no metadata",
                    )
                )
                continue

            # check consistent video codec
            if video_info.vcodec is not None:
                total_vcodec.add(video_info.vcodec)

            # check consistent audio time base
            if video_info.acodec is not None:
                total_acodec.add(video_info.acodec)

            # check consistent rotation
            total_rotate.add(video_info.rotate)

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
                            "component_having_width_lt_height_error",
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
                        video_id,
                        "missing_fps_info_warning",
                        f"component {i} has null fps value",
                    )
                )
            else:
                total_fps.add(video_info.fps)

            # check null mp4 duration
            if video_info.mp4_duration is None:
                error_message.append(
                    ErrorMessage(
                        video_id,
                        "missing_mp4_duration_info_warning",
                        f"component {i} has no mp4 duration",
                    )
                )
            else:
                video_length = (
                    video_info.vstart + video_info.vduration
                    if video_info.vstart is not None
                    else video_info.vduration
                )
                audio_length = (
                    video_info.astart + video_info.aduration
                    if video_info.astart is not None
                    else video_info.aduration
                )
                if video_length is None and audio_length is None:
                    error_message.append(
                        ErrorMessage(
                            video_id,
                            "no_video_or_audio_stream_duration",
                            f"component {i} has no video or audio stream duration metadata",
                        )
                    )

                if video_length is None:
                    video_length = -1

                if audio_length is None:
                    audio_length = -1

                video_stream_length = max(video_length, audio_length)
                threshold = 2 / (video_info.fps or 30)
                delta = abs(video_stream_length - video_info.mp4_duration)
                if delta >= threshold:
                    error_message.append(
                        ErrorMessage(
                            video_id,
                            "mp4_duration_too_large_or_small_warning",
                            f"component {i}: mp4_duration={video_info.mp4_duration}, stream_duration={video_stream_length}, vsd={video_length}, asd={audio_length}",  # noqa
                        )
                    )

        if len(total_vcodec) > 1:
            error_message.append(
                ErrorMessage(
                    video_id,
                    "inconsistent_video_codec_error",
                    "inconsistent video codec",
                )
            )
        if len(total_acodec) > 1:
            error_message.append(
                ErrorMessage(
                    video_id,
                    "inconsistent_audio_codec_error",
                    "inconsistent audio codec",
                )
            )
        if len(total_rotate) > 1:
            error_message.append(
                ErrorMessage(
                    video_id, "inconsistent_rotation_error", "inconsistent rotation"
                )
            )
        if len(total_size) > 1:
            error_message.append(
                ErrorMessage(
                    video_id,
                    "inconsistent_width_height_pair_error",
                    "components with inconsistent width x height",
                )
            )
        if len(total_vtb) > 1:
            error_message.append(
                ErrorMessage(
                    video_id,
                    "inconsistent_video_time_base_error",
                    "inconsistent video time base",
                )
            )
        if len(total_sar) > 1:
            error_message.append(
                ErrorMessage(video_id, "inconsistent_sar_warning", "inconsistent sar")
            )
        if len(total_fps) > 1:
            error_message.append(
                ErrorMessage(
                    video_id, "inconsistent_video_fps_warning", "inconsistent video fps"
                )
            )


def _validate_synchronized_videos(
    video_metadata_dict: Dict[str, VideoMetadata],
    synchronized_video_dict: Dict[str, List[SynchronizedVideos]],
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
    print("validating syncrhonized videos")
    if synchronized_video_dict:
        for video_grouping_id, components in synchronized_video_dict.items():
            for component in components:
                for video_id, _ in component.associated_videos:
                    if video_id not in video_metadata_dict:
                        error_message.append(
                            ErrorMessage(
                                video_id,
                                "video_not_found_in_video_metadata_error",
                                f"{video_id} in synchronized_video_dict can't be found in video_metadata",
                            )
                        )


def _validate_auxilliary_videos(
    video_metadata_dict: Dict[str, VideoMetadata],
    video_components_dict: Dict[str, List[VideoComponentFile]],
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
    print("validating auxiliary videos")
    if auxiliary_video_component_dict:
        for video_id, aux_components in auxiliary_video_component_dict.items():
            if video_id not in video_metadata_dict:
                error_message.append(
                    ErrorMessage(
                        video_id,
                        "video_not_found_in_video_metadata_error",
                        f"{video_id} in auxiliary_video_component_dict can't be found in video_metadata",
                    )
                )

            else:
                redacted_video_components = [
                    component
                    for component in video_components_dict[video_id]
                    if component.is_redacted
                ]
                if video_metadata_dict[video_id].number_video_components - len(
                    redacted_video_components
                ) != len(aux_components):
                    error_message.append(
                        ErrorMessage(
                            video_id,
                            "video_component_length_inconsistent_error",
                            f"the video has {len(aux_components)} auxiliary components when it"
                            f" should have {video_metadata_dict[video_id].number_video_components}",  # noqa
                        )
                    )
                non_redacted_video_components = [
                    component.component_index
                    for component in video_components_dict[video_id]
                    if not component.is_redacted
                ]
                aux_components.sort(key=lambda x: x.component_index)
                for i in range(len(aux_components)):
                    component = aux_components[i]
                    if non_redacted_video_components[i] != component.component_index:
                        error_message.append(
                            ErrorMessage(
                                video_id,
                                "video_component_wrong_index_error",
                                f"the video component has auxiliary component index {component.component_index}"
                                f" when it should have {non_redacted_video_components[i]}",  # noqa
                            )
                        )
                    if component.component_type_id not in component_types:
                        error_message.append(
                            ErrorMessage(
                                video_id,
                                "component_type_id_not_found_error",
                                f"auxiliary component's component_type_id: '{component.component_type_id} does not exist in component_types'",  # noqa
                            )
                        )


def _validate_participant(
    video_metadata_dict: Dict[str, VideoMetadata],
    participant_dict: Dict[str, List[Particpant]],
    error_message: List[ErrorMessage],
) -> None:
    """
    Args:
        participant_dict: Dict[str, Particpant]: mapping from participant_id
        to Participant objects that have equal values at participant_id

        error_message: List[ErrorMessage]: a list to store ErrorMessage objects
        generated when validating participants.csv.
    """
    print("validating participants")
    if participant_dict:
        for participant_id, _ in participant_dict.items():
            if participant_id not in video_metadata_dict:
                error_message.append(
                    ErrorMessage(
                        participant_id,
                        "participant_not_found_error",
                        f"participant '{participant_id}' does not exist in video metadata",
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
    print("validating annotations")
    if annotations_dict:
        for video_id in annotations_dict:
            if video_id not in video_metadata_dict:
                error_message.append(
                    ErrorMessage(
                        video_id,
                        "video_not_found_in_video_metadata_error",
                        f"{video_id} in annotations_dict can't be found in video_metadata",
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
    print("validating video metadata")
    # Check from video_metadata:
    video_ids = set()
    for video_id, video_metadata in video_metadata_dict.items():
        #   0. check no duplicate university_video_id
        if video_id in video_ids:
            error_message.append(
                ErrorMessage(
                    video_id,
                    "duplicate_video_id_error",
                    f"duplicate video id in metadata",
                )
            )
        video_ids.add(video_id)
        #   1. participant_ids are in participant_dict
        if participant_dict:
            if video_metadata.recording_participant_id not in participant_dict:
                error_message.append(
                    ErrorMessage(
                        video_id,
                        "participant_id_not_found_error",
                        f"recording_participant_id '{video_metadata.recording_participant_id}' not in participant_dict",
                    )
                )
            if video_metadata.recording_participant_id is None:
                error_message.append(
                    ErrorMessage(
                        video_id, "null_participant_id_warning", "null participant_id"
                    )
                )

        #   2. scenario_ids are in scenarios
        for scenario_id in video_metadata.video_scenario_ids:
            if scenario_id not in scenarios:
                error_message.append(
                    ErrorMessage(
                        video_id,
                        "scenario_id_not_found_error",
                        f"video_scenario_id: '{scenario_id}' not in scenarios.csv",
                    )
                )

        #   3. device_ids are in devices
        if video_metadata.device_id and video_metadata.device_id not in devices:
            error_message.append(
                ErrorMessage(
                    video_id,
                    "device_id_not_found_error",
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
                        "physical_setting_id_not_found_error",
                        f"physical_setting_id '{video_metadata.physical_setting_id}' not in physical_setting.csv",
                    )
                )

        #   5. university_video_ids are in components
        if video_id not in video_components_dict:
            error_message.append(
                ErrorMessage(
                    video_id,
                    "video_not_found_in_video_components_error",
                    f"{video_id} in video_metadata can't be found in video_components_dict",
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
    error_details_path: str,
    error_summary_path: str,
    num_workers: int,
    expiry_time_sec: int,
) -> List[ErrorMessage]:
    error_message = []
    # Check ids in video_components_dict are in video_metadata_dict
    # and the # of components is correct
    video_info_param = []

    video_info_param = _validate_video_components(
        s3=s3,
        bucket_name=bucket_name,
        video_metadata_dict=video_metadata_dict,
        video_components_dict=video_components_dict,
        error_message=error_message,
        num_workers=num_workers,
    )
    video_info_dict = _get_videos(
        s3=s3,
        bucket_name=bucket_name,
        video_info_param=video_info_param,
        error_message=error_message,
        num_workers=num_workers,
        expiry_time_sec=expiry_time_sec,
    )
    _validate_mp4(video_info_dict, error_message)
    _validate_synchronized_videos(
        video_metadata_dict=video_metadata_dict,
        synchronized_video_dict=synchronized_video_dict,
        error_message=error_message,
    )
    _validate_auxilliary_videos(
        video_metadata_dict=video_metadata_dict,
        video_components_dict=video_components_dict,
        auxiliary_video_component_dict=auxiliary_video_component_dict,
        component_types=component_types,
        error_message=error_message,
    )
    _validate_participant(video_metadata_dict, participant_dict, error_message)
    _validate_annotations(video_metadata_dict, annotations_dict, error_message)
    _validate_video_metadata(
        video_metadata_dict=video_metadata_dict,
        video_components_dict=video_components_dict,
        participant_dict=participant_dict,
        scenarios=scenarios,
        devices=devices,
        physical_setting_dict=physical_setting_dict,
        component_types=component_types,
        error_message=error_message,
    )

    error_dict = collections.defaultdict(int)
    if error_message:
        for err in error_message:
            error_dict[err.errorType] += 1

    fields = ["univeristy_video_id", "errorType", "description"]
    with open(error_details_path, "w") as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerow(fields)
        for e in error_message:
            write.writerow([e.uid, e.errorType, e.description])
    with open(error_summary_path, "w") as f:
        write = csv.writer(f)
        write.writerow(["error_type", "num_of_occurrences"])
        for error_type, error_counts in error_dict.items():
            write.writerow([error_type, error_counts])
    return error_message


def validate_all(
    path,
    s3,
    standard_metadata_folder,
    error_details_path,
    error_summary_path,
    num_workers,
    expiry_time_sec,
):
    # get access to metadata_folder
    devices, component_types, scenarios = load_standard_metadata_files(
        s3, standard_metadata_folder
    )
    bucket, path = split_s3_path(path)

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
        error_details_path,
        error_summary_path,
        num_workers,
        expiry_time_sec=expiry_time_sec,
    )
