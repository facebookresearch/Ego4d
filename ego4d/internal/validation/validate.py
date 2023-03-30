# pyre-strict
import functools
import itertools
import os
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from ego4d.cli.manifest import VideoMetadata

from ego4d.cli.universities import BUCKET_TO_UNIV
from ego4d.internal.s3 import StreamPathMgr
from ego4d.internal.validation.ffmpeg_utils import get_video_info, VideoInfo
from ego4d.internal.validation.types import (
    Annotations,
    AuxiliaryVideoComponentDataFile,
    CaptureMetadataEgoExo,
    ComponentType,
    Error,
    ErrorLevel,
    load_egoexo_manifest,
    load_manifest,
    load_released_video_files,
    load_standard_metadata_files,
    load_standard_metadata_files_egoexo,
    Manifest,
    ManifestEgoExo,
    Particpant,
    StandardMetadata,
    StandardMetadataEgoExo,
    SynchronizedVideos,
    VideoComponentFile,
)

from iopath.common.file_io import PathManager
from iopath.common.s3 import S3PathHandler
from tqdm import tqdm

stream_path_mgr: Optional[StreamPathMgr] = None  # for streaming video
pathmgr = PathManager()  # for downloading files
pathmgr.register_handler(S3PathHandler(profile="default"))


def _split_s3_path(s3_path: str) -> Tuple[Optional[str], str]:
    """
    Splits a full s3_path of the form "s3://bucket/folder/.../file.ext"
    into a tuple of ("bucket", "folder/.../file.ext")

    If this doesn't start with s3:// it will assume that the there is no
    bucket and the path is just returned as the second output
    """
    if s3_path[:5] == "s3://":
        s3_path_components = s3_path[5:].split("/")
        return s3_path_components[0], "/".join(s3_path_components[1:])
    # pyre-fixme[7]: Expected `Tuple[str, str]` but got `Tuple[None, str]`.
    return None, s3_path


def _validate_vcs(
    video_id_component_pair: Tuple[str, List[VideoComponentFile]],
    videos: Dict[str, VideoMetadata],
    university: str,
) -> List[Error]:
    errors = []
    video_id, components = video_id_component_pair
    components.sort(key=lambda x: x.component_index)

    if video_id not in videos:
        errors.append(
            Error(
                ErrorLevel.ERROR,
                video_id,
                "video_not_found_in_video_metadata",
                f"{video_id} in video_components_dict can't be found in video_metadata",
            )
        )
        return errors
    elif videos[video_id].number_video_components != len(components):
        errors.append(
            Error(
                ErrorLevel.ERROR,
                video_id,
                "video_component_length_inconsistent",
                f"the video has {len(components)} components when it"
                f" should have {videos[video_id].number_video_components}",
            )
        )
    components.sort(key=lambda x: x.component_index)
    # check component_index is incremental and starts at 0
    university_video_folder_path = videos[video_id].university_video_folder_path
    for i in range(len(components)):
        if components[i].component_index != i:
            errors.append(
                Error(
                    ErrorLevel.ERROR,
                    video_id,
                    "video_component_wrong_index",
                    f"the video component has index {components[i].component_index}"
                    f" when it should have {i}",
                )
            )
        if (
            not components[i].video_component_relative_path
            and not components[i].is_redacted
        ):
            errors.append(
                Error(
                    ErrorLevel.ERROR,
                    video_id,
                    "empty_video_component_relative_path",
                    f"Found an empty relative path for the video_id {video_id}",
                )
            )
        else:
            if components[i].is_redacted:
                path = os.path.join(
                    university_video_folder_path,
                    components[i].video_component_relative_path,
                )
                if not path.startswith("s3://"):
                    errors.append(
                        Error(
                            ErrorLevel.WARN,
                            video_id,
                            "using_local_path",
                            f"{path} ",
                        )
                    )

                if "s3://" in path:
                    bucket = path.split("s3://")[1].split("/")[0]
                    assoc_uni = BUCKET_TO_UNIV[bucket]
                    if assoc_uni != university:
                        errors.append(
                            Error(
                                ErrorLevel.ERROR,
                                video_id,
                                "incorrect_bucket_name",
                                path,
                            )
                        )

                if not pathmgr.exists(path):
                    errors.append(
                        Error(
                            ErrorLevel.ERROR,
                            video_id,
                            "path_does_not_exist",
                            path,
                        )
                    )

    return errors


def validate_video_components(
    manifest: Manifest,
    university: str,
    num_workers: int,
) -> List[Error]:
    """
    This method in parallel (w.r.t num_workers) checks each video component file
    provided in the manifest.

    Refer to _validate_vcs for logic
    """
    map_fn = functools.partial(
        _validate_vcs,
        videos=manifest.videos,
        university=university,
    )
    errors = []
    with ThreadPoolExecutor(num_workers) as pool:
        vals_to_map = list(manifest.video_components.items())
        for errs in tqdm(pool.map(map_fn, vals_to_map), total=len(vals_to_map)):
            errors.extend(errs)

    return errors


def validate_mp4s(video_infos: Dict[str, List[VideoInfo]]) -> List[Error]:  # noqa
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
    """
    errors = []
    for video_id, vis in video_infos.items():
        total_vcodec = set()
        total_acodec = set()
        total_rotate = set()
        total_size = set()
        total_vtb = set()
        total_sar = set()
        total_fps = set()

        for i, video_info in enumerate(vis):
            if video_info is None:
                errors.append(
                    Error(
                        ErrorLevel.ERROR,
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
            if (
                video_info.sample_width is not None
                and video_info.sample_height is not None
            ):
                total_size.add((video_info.sample_width, video_info.sample_height))

                if (
                    video_info.sample_width < video_info.sample_height
                    and video_info.rotate is None
                ):
                    errors.append(
                        Error(
                            ErrorLevel.ERROR,
                            video_id,
                            "component_having_width_lt_height",
                            f"component {i} has width < height without rotation",
                        )
                    )

            # check consistent video time base
            if video_info.video_time_base is not None:
                total_vtb.add(video_info.video_time_base)

            # check consistent sar
            if video_info.sar is not None:
                total_sar.add(video_info.sar)

            # check null/inconsistent video fps
            if video_info.fps is None:
                errors.append(
                    Error(
                        ErrorLevel.WARN,
                        video_id,
                        "missing_fps_info",
                        f"component {i} has null fps value",
                    )
                )
            else:
                total_fps.add(video_info.fps)

            # check null mp4 duration
            if video_info.mp4_duration is None:
                errors.append(
                    Error(
                        ErrorLevel.WARN,
                        video_id,
                        "missing_mp4_duration_info",
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
                    errors.append(
                        Error(
                            ErrorLevel.WARN,
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
                    errors.append(
                        Error(
                            ErrorLevel.WARN,
                            video_id,
                            "mp4_duration_too_large_or_small",
                            f"component {i}: mp4_duration={video_info.mp4_duration}, stream_duration={video_stream_length}, vsd={video_length}, asd={audio_length}",  # noqa
                        )
                    )

        if len(total_vcodec) > 1:
            errors.append(
                Error(
                    ErrorLevel.ERROR,
                    video_id,
                    "inconsistent_video_codec",
                    "",
                )
            )
        if len(total_acodec) > 1:
            errors.append(
                Error(
                    ErrorLevel.WARN,
                    video_id,
                    "inconsistent_audio_codec",
                    "",
                )
            )
        if len(total_rotate) > 1:
            errors.append(
                Error(ErrorLevel.ERROR, video_id, "inconsistent_rotation", "")
            )
        if len(total_size) > 1:
            errors.append(
                Error(
                    ErrorLevel.ERROR,
                    video_id,
                    "inconsistent_width_height_pair",
                    "components with inconsistent width x height",
                )
            )
        if len(total_vtb) > 1:
            errors.append(
                Error(
                    ErrorLevel.ERROR,
                    video_id,
                    "inconsistent_video_time_base",
                    "",
                )
            )
        if len(total_sar) > 1:
            errors.append(Error(ErrorLevel.WARN, video_id, "inconsistent_sar", ""))
        if len(total_fps) > 1:
            errors.append(
                Error(ErrorLevel.WARN, video_id, "inconsistent_video_fps", "")
            )
    return errors


def validate_synchronized_metadata(
    video_metadata_dict: Dict[str, VideoMetadata],
    synchronized_video_dict: Dict[str, SynchronizedVideos],
) -> List[Error]:
    """
    Args:
        synchronized_video_dict: Dict[str, SynchronizedVideos]: mapping from
        video_grouping_id  to a list of all SynchronizedVideos objects that
        have equal values at video_grouping_id

        error_message: List[Error]: a list to store Error objects
        generated when validating synchronized_videos.csv.
    """
    errors = []
    if synchronized_video_dict:
        for video_grouping_id, component in synchronized_video_dict.items():
            for video_id, _ in component.associated_videos.items():
                if video_id not in video_metadata_dict:
                    errors.append(
                        Error(
                            ErrorLevel.ERROR,
                            video_id,
                            "video_not_found_in_video_metadata",
                            f"({video_id}, {video_grouping_id}) in synchronized_video_dict can't be found in video_metadata",  # noqa
                        )
                    )
    return errors


def validate_auxilliary_videos(
    video_metadata_dict: Dict[str, VideoMetadata],
    video_components_dict: Dict[str, List[VideoComponentFile]],
    auxiliary_video_component_dict: Dict[str, List[AuxiliaryVideoComponentDataFile]],
    component_types: Dict[str, ComponentType],
) -> List[Error]:
    """
    Args:
        auxiliary_video_component_dict: Dict[str, List[AuxiliaryVideoComponentDataFile]]:
        mapping from university_video_id to a list of all AuxiliaryVideoComponentDataFile
        objects that have equal values at university_video_id

        error_message: List[Error]: a list to store Error objects
        generated when validating auxilliary_videos.csv.
    """
    # Check ids in auxiliary_video_component_dict are in video_metadata_dict
    # and that the component_type is valid
    errors = []
    if auxiliary_video_component_dict:
        for video_id, aux_components in auxiliary_video_component_dict.items():
            if video_id not in video_metadata_dict:
                errors.append(
                    Error(
                        ErrorLevel.ERROR,
                        video_id,
                        "video_not_found_in_video_metadata",
                        f"{video_id} in auxiliary_video_component_dict can't be found in video_metadata",  # noqa
                    )
                )

            else:
                vcs = [
                    component.component_index
                    for component in video_components_dict[video_id]
                ]

                for component_type_id, xs in itertools.groupby(
                    aux_components, lambda x: x.component_type_id
                ):
                    xs = list(xs)
                    if video_metadata_dict[video_id].number_video_components != len(xs):
                        errors.append(
                            Error(
                                ErrorLevel.ERROR,
                                video_id,
                                "video_component_length_inconsistent",
                                f"the video has {len(aux_components)} auxiliary components when it"  # noqa
                                f" should have {video_metadata_dict[video_id].number_video_components}",  # noqa
                            )
                        )
                        continue

                    xs.sort(key=lambda x: x.component_index)  # pyre-ignore
                    for i, component in enumerate(xs):
                        if vcs[i] != component.component_index:
                            errors.append(
                                Error(
                                    ErrorLevel.ERROR,
                                    video_id,
                                    "video_component_wrong_index",
                                    "the video component has auxiliary component index",
                                    f"{component.component_index} (type_id={component_type_id})",  # noqa
                                    f" when it should have {vcs[i]}",  # noqa
                                )
                            )
                        if component.component_type_id not in component_types:
                            errors.append(
                                Error(
                                    ErrorLevel.ERROR,
                                    video_id,
                                    "component_type_id_not_found",
                                    f"auxiliary component's (type_id={component_type_id})",
                                    f"'{component.component_type_id}",
                                    "does not exist in component_types'",  # noqa
                                )
                            )
    return errors


def validate_participant(
    video_metadata_dict: Dict[str, VideoMetadata],
    participant_dict: Dict[str, List[Particpant]],
) -> List[Error]:
    """
    Args:
        participant_dict: Dict[str, Particpant]: mapping from participant_id
        to Participant objects that have equal values at participant_id

        error_message: List[Error]: a list to store Error objects
        generated when validating participants.csv.
    """
    errors = []
    if participant_dict:
        ps_in_video_metadata = {
            x.recording_participant_id for x in video_metadata_dict.values()
        }
        ps_in_meta = set(participant_dict.keys())

        wrong_ps = ps_in_video_metadata - ps_in_meta

        if len(wrong_ps) > 0:
            for participant_id in wrong_ps:
                errors.append(
                    Error(
                        ErrorLevel.ERROR,
                        participant_id,
                        "participant_metadata_not_found",
                        f"participant '{participant_id}' does not exist in participant metadata",  # noqa
                    )
                )

    else:
        errors.append(Error(ErrorLevel.WARN, "general", "no_participant_metadata", ""))
    return errors


def validate_annotations(
    video_metadata_dict: Dict[str, VideoMetadata],
    annotations_dict: Dict[str, Annotations],
) -> List[Error]:
    """
    Args:
        annotations_dict: Dict[str, Annotations]: mapping from participant_id
        to Participant objects that have equal values at participant_id

        error_message: List[Error]: a list to store Error objects
        generated when validating participants.csv.
    """
    print("validating annotations")
    errors = []
    if annotations_dict:
        for video_id in annotations_dict:
            if video_id not in video_metadata_dict:
                errors.append(
                    Error(
                        ErrorLevel.ERROR,
                        video_id,
                        "video_not_found_in_video_metadata",
                        f"{video_id} in annotations_dict can't be found in video_metadata",
                    )
                )
    return errors


def validate_video_metadata(
    manifest: Manifest,
    standard_metadata: StandardMetadata,
) -> List[Error]:
    # Check from video_metadata:
    errors = []
    video_ids = set()
    for video_id, video_metadata in manifest.videos.items():
        # 0. check no duplicate university_video_id
        if video_id in video_ids:
            errors.append(
                Error(
                    ErrorLevel.ERROR,
                    video_id,
                    "duplicate_video_id",
                    "duplicate video id in metadata",
                )
            )
        video_ids.add(video_id)
        # 1. participant_ids are in participant_dict
        if manifest.participants is not None:
            if video_metadata.recording_participant_id is None:
                errors.append(
                    Error(
                        ErrorLevel.ERROR,
                        video_id,
                        "null_participant_id",
                        "",
                    )
                )

        # 2. scenario_ids are in scenarios
        for scenario_id in standard_metadata.scenarios:
            if scenario_id not in standard_metadata.scenarios:
                errors.append(
                    Error(
                        ErrorLevel.ERROR,
                        video_id,
                        "scenario_id_not_found",
                        f"video_scenario_id: '{scenario_id}' not in scenarios.csv",
                    )
                )

        # 3. device_ids are in devices
        if video_metadata.device_id is None:
            errors.append(Error(ErrorLevel.WARN, video_id, "device_id_null", ""))
        elif video_metadata.device_id not in standard_metadata.devices:
            errors.append(
                Error(
                    ErrorLevel.ERROR,
                    video_id,
                    "device_id_not_found",
                    f"device_id '{video_metadata.device_id}' not in devices.csv",
                )
            )

        # 4. physical_settings are in physical_setting
        if (
            video_metadata.physical_setting_id
            and video_metadata.physical_setting_id not in manifest.physical_setting
        ):
            errors.append(
                Error(
                    ErrorLevel.ERROR,
                    video_id,
                    "physical_setting_id_not_found",
                    f"physical_setting_id '{video_metadata.physical_setting_id}' not in physical_setting.csv",  # noqa
                )
            )

        # 5. university_video_ids are in components
        if video_id not in manifest.video_components:
            errors.append(
                Error(
                    ErrorLevel.ERROR,
                    video_id,
                    "video_not_found_in_video_components",
                    f"{video_id} in video_metadata can't be found in video_components_dict",
                )
            )
    return errors


def _get_video_metadata_map_fn(
    uid_path_pair: Tuple[str, str],
) -> Tuple[str, Optional[VideoInfo], Optional[Error]]:
    assert stream_path_mgr is not None
    uid, path = uid_path_pair
    metadata, errs = get_video_info(path, name=uid)
    return uid, metadata, errs


def get_video_metadata(
    video_paths: Dict[Any, str],
    num_workers: int,
) -> Tuple[Dict[str, VideoMetadata], List[Error]]:
    errors = []
    metadata = {}

    # prefetch the presigned URLs
    paths_to_check = {}
    for k, path in video_paths.items():
        paths_to_check[k] = stream_path_mgr.open(path)

    iterables = list(paths_to_check.items())
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for uid, m, errs in tqdm(
            pool.map(_get_video_metadata_map_fn, iterables), total=len(iterables)
        ):
            metadata[uid] = m
            if errs is not None:
                errors.append(errs)
    return metadata, errors


def validate_ego4d_files(  # noqa :C901
    university: str,
    manifest: Manifest,
    metadata: StandardMetadata,
    num_workers: int,
) -> List[Error]:
    errors = []
    errors.extend(
        validate_video_components(
            manifest=manifest,
            university=university,
            num_workers=num_workers,
        )
    )

    vps = {
        (vc.university_video_id, vc.component_index): os.path.join(
            manifest.videos[video_id].university_video_folder_path,
            vc.video_component_relative_path,
        )
        for video_id, vcs in manifest.video_components.items()
        for vc in vcs
    }
    print("Obtaining MP4 metadata")
    video_infos, video_errs = get_video_metadata(vps, num_workers)
    errors.extend(video_errs)

    video_infos_by_video_id = defaultdict(list)
    for (v_uid, comp_idx), vi in video_infos.items():
        video_infos_by_video_id[v_uid].append((comp_idx, vi))

    for vs in video_infos_by_video_id.values():
        vs.sort(key=lambda x: x[0])

    print("Validating MP4 files")
    errors.extend(
        validate_mp4s(
            {k: [x[1] for x in vs] for k, vs in video_infos_by_video_id.items()}
        )
    )
    print("Validating synchronized video metadata")
    errors.extend(
        validate_synchronized_metadata(
            video_metadata_dict=manifest.videos,
            synchronized_video_dict=manifest.sync_videos,
        )
    )
    print("Validating auxiliary video metadata")
    errors.extend(
        validate_auxilliary_videos(
            video_metadata_dict=manifest.videos,
            video_components_dict=manifest.video_components,
            auxiliary_video_component_dict=manifest.aux_components,
            component_types=metadata.component_types,
        )
    )
    print("Validating participant metadata")
    errors.extend(
        validate_participant(
            manifest.videos,
            manifest.participants,
        )
    )
    print("validating annotations")
    errors.extend(
        validate_annotations(
            manifest.videos,
            manifest.annotations,
        )
    )
    print("validating video metadata")
    errors.extend(
        validate_video_metadata(
            manifest=manifest,
            standard_metadata=metadata,
        )
    )
    return errors


def summarize_errors(
    errors: List[Error],
    released_videos: Optional[Dict[str, List[str]]],
    university: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    error_dict_non_released = defaultdict(int)
    error_dict_released = defaultdict(int)
    if len(errors) > 0 and released_videos is not None:
        for err in errors:
            if err.uid not in released_videos[university]:
                error_dict_non_released[err.errorType] += 1
            else:
                error_dict_released[err.errorType] += 1

    errors_dict = {
        "uid": [],
        "error_type": [],
        "description": [],
        "is_released": [],
    }

    err_by_type_in_release = defaultdict(int)
    err_by_type_total = defaultdict(int)
    for e in errors:
        errors_dict["uid"].append(e.uid)
        errors_dict["error_type"].append(e.type)
        errors_dict["description"].append(e.description)
        errors_dict["is_released"].append(
            e.uid in released_videos if released_videos is not None else None
        )

        if released_videos is not None and e.uid in released_videos:
            err_by_type_in_release[e.type] += 1
        err_by_type_total[e.type] += 1

    summary_dict = {
        "error_type": [],
        "num_total": [],
        "num_in_release": [],
    }

    for et, t in err_by_type_total.items():
        summary_dict["error_type"].append(et)
        summary_dict["num_total"].append(t)
        summary_dict["num_in_release"].append(err_by_type_in_release.get(et, None))

    errors_df = pd.DataFrame(errors_dict)
    summary_df = pd.DataFrame(summary_dict)
    return errors_df, summary_df


def _check_associated_takes_metadata(
    manifest: ManifestEgoExo,
    metadata: StandardMetadataEgoExo,
    capture: CaptureMetadataEgoExo,
    capture_uid: str,
) -> List[Error]:
    ret = []
    if capture_uid not in manifest.takes:
        ret.append(
            Error(
                ErrorLevel.ERROR,
                capture_uid,
                "no_takes_for_capture",
                f"capture {capture_uid} has no associated takes",
            )
        )
        return ret

    takes = manifest.takes[capture_uid]
    if len(takes) != capture.number_takes:
        ret.append(
            Error(
                ErrorLevel.ERROR,
                capture_uid,
                "incorrect_number_takes_for_capture",
                f"capture {capture_uid} has {capture.number_takes} specified but only recieved {len(takes)} takes",
            )
        )

    for take in takes:
        if take.scenario_id not in metadata.scenarios:
            ret.append(
                Error(
                    ErrorLevel.ERROR,
                    take.take_id,
                    "take_has_incorrect_scenario_id",
                    f"take {take.take_id} for capture {capture_uid} has incorrect scenario ID {take.scenario_id}",
                )
            )
        if take.recording_participant_id is None:
            ret.append(
                Error(
                    ErrorLevel.WARN,
                    take.take_id,
                    "take_missing_participant",
                    f"take {take.take_id} for capture {capture_uid} has missing participant ID",
                )
            )

        if take.object_ids is None:
            ret.append(
                Error(
                    ErrorLevel.WARN,
                    take.take_id,
                    "take_missing_objects",
                    f"take {take.take_id} has no objects provided. Provide an empty list to silence warning",
                )
            )

        additional_objects = set(take.object_ids or {}) - set(manifest.objects or {})
        if len(additional_objects) > 0:
            ret.append(
                Error(
                    ErrorLevel.ERROR,
                    take.take_id,
                    "take_has_unspecified_objects",
                    f"take {take.take_id} for capture {capture_uid} has unspecified objects: {additional_objects}",
                )
            )

    return ret


def _check_capture_metadata(
    manifest: ManifestEgoExo,
    metadata: StandardMetadataEgoExo,
    capture: CaptureMetadataEgoExo,
    capture_uid: str,
) -> List[Error]:
    ret = []
    capture = manifest.captures[capture_uid]
    if capture.post_surveys_relative_path is None:
        ret.append(
            Error(
                ErrorLevel.ERROR,
                capture_uid,
                "capture_missing_post_surveys",
                f"capture {capture_uid} has no post survey data",
            )
        )

    if capture.start_date_recorded_utc is None:
        ret.append(
            Error(
                ErrorLevel.WARN,
                capture_uid,
                "capture_missing_recording_date",
                f"capture {capture_uid} has no recording date",
            )
        )

    return ret


def _check_colmap(
    manifest: ManifestEgoExo,
) -> List[Error]:
    all_colmap = manifest.colmap
    assert all_colmap is not None

    ret = []
    for capture_uid, colmap in all_colmap.items():
        for c in colmap:
            if capture_uid not in manifest.captures:
                ret.append(
                    Error(
                        ErrorLevel.ERROR,
                        c.colmap_configuration_id,
                        "colmap_configuration_invalid_capture_uid_fk",
                        f"colmap configuration links to non-existent capture_uid: {capture_uid}",
                    )
                )

            if c.colmap_ran and not c.was_inspected:
                ret.append(
                    Error(
                        ErrorLevel.ERROR,
                        c.colmap_configuration_id,
                        "colmap_ran_but_not_inspected",
                        f"colmap configuration for {c.colmap_configuration_id} was not inspected but ran",
                    )
                )

        vers_for_capture = set(c.version for c in colmap)
        if len(vers_for_capture) != len(colmap):
            ret.append(
                Error(
                    ErrorLevel.ERROR,
                    capture_uid,
                    "colmap_duplicate_versions",
                    f"Duplicate colmap versions found for capture uid: {capture_uid}",
                )
            )

    return ret


def _check_objects(manifest: ManifestEgoExo) -> List[Error]:
    ret = []
    physical_setting_ids = (
        set(manifest.physical_setting) if manifest.physical_setting is not None else {}
    )
    assert manifest.objects is not None
    for object_id, obj in manifest.objects.items():
        if obj.physical_setting_id not in physical_setting_ids:
            ret.append(
                Error(
                    ErrorLevel.ERROR,
                    object_id,
                    "object_links_to_invalid_physical_setting_id",
                    f"phyiscal setting {obj.physical_setting_id} does not exist",
                )
            )
    return ret


def _check_participants(
    manifest: ManifestEgoExo, metadata: StandardMetadataEgoExo
) -> List[Error]:
    ret = []
    assert manifest.participants is not None
    for participant_id, p in manifest.participants.items():
        if p.scenario_id not in metadata.scenarios:
            ret.append(
                Error(
                    ErrorLevel.ERROR,
                    participant_id,
                    "participant_links_to_invalid_scenario",
                    "scenario {p.scenario_id} does not exist",
                )
            )
        if len(p.pre_survey_data) == 0:
            ret.append(
                Error(
                    ErrorLevel.ERROR,
                    participant_id,
                    "participant_has_empty_pre_survey_data",
                    "participants should have pre survey data populated",
                )
            )
        else:
            ks = {
                "recording_location",
                "scenario_num_iterations",
                "scenario_frequency",
                "scenario_experience_years",
                "has_taught_scenario",
                "has_recorded_scenario_howto",
                "typical_time_to_complete_scenario_minutes",
            }
            given_ks = set(p.pre_survey_data.keys())

            missing_ks = ks - given_ks
            aux_ks = given_ks - ks
            if len(missing_ks) > 0 or len(aux_ks) > 0:
                ret.append(
                    Error(
                        ErrorLevel.WARN,
                        participant_id,
                        "participant_pre_survey_data_contraints",
                        f"auxiliary keys: {aux_ks}, missing keys: {missing_ks}",
                    )
                )

            if "recording_location" in given_ks:
                v = p.pre_survey_data["recording_location"]
                valid_vs = {
                    "typical",
                    "familiar",
                    "unfamiliar",
                }
                if v not in valid_vs:
                    ret.append(
                        Error(
                            ErrorLevel.WARN,
                            participant_id,
                            "participant_pre_survey_data_contraints",
                            f"recording_location scenario_num_iterations {v} not one of: {valid_vs}",
                        )
                    )

            if "scenario_num_iterations" in given_ks:
                v = p.pre_survey_data["scenario_num_iterations"]
                valid_vs = {
                    "1-10",
                    "10-100",
                    "100-500",
                    "500-1000",
                    "1000+",
                }
                if v not in valid_vs:
                    ret.append(
                        Error(
                            ErrorLevel.WARN,
                            participant_id,
                            "participant_pre_survey_data_contraints",
                            f"scenario_num_iterations {v} not one of: {valid_vs}",
                        )
                    )

            if "scenario_frequency" in given_ks:
                v = p.pre_survey_data["scenario_frequency"]
                valid_vs = {
                    "daily",
                    "weekly",
                    "monthly",
                    "rarely",
                    "never",
                }
                if v not in valid_vs:
                    ret.append(
                        Error(
                            ErrorLevel.WARN,
                            participant_id,
                            "participant_pre_survey_data_contraints",
                            f"scenario_frequency {v} not one of: {valid_vs}",
                        )
                    )

            if "scenario_experience_years" in given_ks:
                v = p.pre_survey_data["scenario_experience_years"]
                valid_vs = {
                    "1 year",
                    "1-3 years",
                    "3-5 years",
                    "5-10 years",
                    "10+ years",
                }
                if v not in valid_vs:
                    ret.append(
                        Error(
                            ErrorLevel.WARN,
                            participant_id,
                            "participant_pre_survey_data_contraints",
                            f"scenario_experience_years {v} not one of: {valid_vs}",
                        )
                    )
            if "has_taught_scenario" in given_ks:
                v = p.pre_survey_data["has_taught_scenario"]
                try:
                    _ = bool(v)
                except Exception:
                    ret.append(
                        Error(
                            ErrorLevel.WARN,
                            participant_id,
                            "participant_pre_survey_data_contraints",
                            f"has_taught_scenario could not be converted to boolean: {traceback.format_exc()}",
                        )
                    )
            if "has_recorded_scenario_howto" in given_ks:
                v = p.pre_survey_data["has_recorded_scenario_howto"]
                try:
                    _ = bool(v)
                except Exception:
                    ret.append(
                        Error(
                            ErrorLevel.WARN,
                            participant_id,
                            "participant_pre_survey_data_contraints",
                            f"has_recorded_scenario_howto could not be converted to boolean: {traceback.format_exc()}",
                        )
                    )

            if "typical_time_to_complete_scenario_minutes" in given_ks:
                v = p.pre_survey_data["typical_time_to_complete_scenario_minutes"]
                try:
                    _ = int(v)
                except Exception:
                    ret.append(
                        Error(
                            ErrorLevel.WARN,
                            participant_id,
                            "participant_pre_survey_data_contraints",
                            f"typical_time_to_complete_scenario_minutes could not be converted to integer: {traceback.format_exc()}",
                        )
                    )

        if len(p.participant_metadata) == 0:
            ret.append(
                Error(
                    ErrorLevel.ERROR,
                    participant_id,
                    "participant_empty_metadata",
                    "participant has empty metadata for demographics, etc.",
                )
            )
        else:
            ks = {
                "age_range",
                "gender",
                "race_ethnicity",
                "country_born",
                "native_language",
                "home_language",
                "education_completed",
                "current_student",
                "field",
            }
            given_ks = set(p.participant_metadata.keys())
            missing_ks = ks - given_ks
            aux_ks = given_ks - ks
            if len(missing_ks) > 0 or len(aux_ks) > 0:
                ret.append(
                    Error(
                        ErrorLevel.WARN,
                        participant_id,
                        "participant_metadata_constraints",
                        f"auxiliary keys: {aux_ks}, missing keys: {missing_ks}",
                    )
                )

            if "gender" in p.participant_metadata:
                v = p.participant_metadata["gender"]
                valid_vs = {
                    "female",
                    "male",
                    "non_binary",
                    "prefer_not_to_disclose",
                    "prefer_to_self_describe",
                }
                if v not in valid_vs:
                    ret.append(
                        Error(
                            ErrorLevel.WARN,
                            participant_id,
                            "participant_metadata_constraints",
                            f"gender '{v}' not one of: {valid_vs}",
                        )
                    )

    return ret


def _check_video_metadata(
    manifest: ManifestEgoExo, metadata: StandardMetadataEgoExo
) -> List[Error]:
    ret = []
    for capture_uid, videos in manifest.videos.items():
        any_has_walkaround = any(v.has_walkaround for v in videos)
        any_is_ego = any(v.is_ego for v in videos)
        if not any_has_walkaround:
            ret.append(
                Error(
                    ErrorLevel.ERROR,
                    capture_uid,
                    "capture_missing_walkaround",
                    "capture does not have walkaround (from video metadata)",
                )
            )

        if not any_is_ego:
            ret.append(
                Error(
                    ErrorLevel.ERROR,
                    capture_uid,
                    "capture_missing_ego",
                    "capture does not have ego view",
                )
            )

        seen_devices = set()
        for video in videos:
            video_uid = (capture_uid, video.university_video_id)
            video_uid_str = (
                f"capture_uid={capture_uid},video_id={video.university_video_id}"
            )
            if capture_uid not in manifest.captures:
                ret.append(
                    Error(
                        ErrorLevel.ERROR,
                        video_uid_str,
                        "video_invalid_capture_uid_fk",
                        f"video links to invalid capture_uid {capture_uid}",
                    )
                )
            if video_uid not in manifest.video_components:
                ret.append(
                    Error(
                        ErrorLevel.ERROR,
                        video_uid_str,
                        "video_has_no_linked_components",
                        "video has no associated video components",
                    )
                )
            else:
                vcs = manifest.video_components[video_uid]  # pyre-ignore
                if len(vcs) != video.number_video_components:
                    ret.append(
                        Error(
                            ErrorLevel.ERROR,
                            video_uid_str,
                            "video_has_incorrect_num_components",
                            f"video specified {video.number_video_components} linked components but has {len(vcs)} in metadata",
                        )
                    )

                all_comps_red = all(vc.is_redacted for vc in vcs)
                if video.is_redacted and not all_comps_red:
                    ret.append(
                        Error(
                            ErrorLevel.ERROR,
                            video_uid_str,
                            "video_associated_components_not_redacted",
                            f"video flagged as redacted but components are not all redacted",
                        )
                    )

            if video.device_type not in metadata.devices:
                ret.append(
                    Error(
                        ErrorLevel.ERROR,
                        video_uid_str,
                        "video_invalid_device",
                        f"video has invalid device_type: {video.device_type}",
                    )
                )
            device_uid = (video.device_type, video.device_id)
            if device_uid in seen_devices:
                ret.append(
                    Error(
                        ErrorLevel.ERROR,
                        video_uid_str,
                        "video_duplicate_device",
                        f"video has duplicate device type and instance: device_type={video.device_type}, device_id (instance)={video.device_id}",
                    )
                )
            seen_devices.add(device_uid)
    return ret


def _check_video_components(manifest: ManifestEgoExo) -> List[Error]:
    ret = []
    all_video_ids = set(
        v.university_video_id
        for vs in manifest.videos.values()
        for v in vs
        if v.number_video_components > 0
    )
    for (capture_uid, video_id), vcs in manifest.video_components.items():
        next_expected_idx = 0
        for vc in sorted(vcs, key=lambda x: x.component_index):
            if video_id not in all_video_ids:
                ret.append(
                    Error(
                        ErrorLevel.ERROR,
                        vc.video_component_relative_path,
                        "video_component_wrong_video_fk",
                        f"video component has incorrect video id: {video_id}",
                    )
                )
                continue

            if capture_uid not in manifest.captures:
                assert capture_uid == vc.university_capture_id
                ret.append(
                    Error(
                        ErrorLevel.ERROR,
                        vc.video_component_relative_path,
                        "video_component_wrong_capture_uid_fk",
                        f"video component has incorrect university_capture_id: {vc.university_capture_id}",
                    )
                )
                continue

            if vc.component_index != next_expected_idx:
                ret.append(
                    Error(
                        ErrorLevel.ERROR,
                        vc.video_component_relative_path,
                        "video_component_incorrect_index",
                        f"expected index: {next_expected_idx}, got {vc.component_index}",
                    )
                )
            next_expected_idx += 1
    return ret


def validate_egoexo_files(
    university: str,
    manifest: ManifestEgoExo,
    metadata: StandardMetadataEgoExo,
    num_workers: int,
) -> List[Error]:
    ret = []

    for capture_uid, capture in manifest.captures.items():
        ret.extend(_check_capture_metadata(manifest, metadata, capture, capture_uid))
        ret.extend(
            _check_associated_takes_metadata(manifest, metadata, capture, capture_uid)
        )

    if not manifest.colmap:
        ret.append(
            Error(
                ErrorLevel.WARN,
                "colmap",
                "no_colmap_metadata_provided",
                "missing colmap metadata (null or empty)",
            )
        )
    else:
        ret.extend(_check_colmap(manifest))

    if not manifest.participants:
        ret.append(
            Error(
                ErrorLevel.WARN,
                "participants",
                "no_participant_metadata",
                "missing participant metadata (null or empty)",
            )
        )
    else:
        ret.extend(_check_participants(manifest, metadata))

    if not manifest.physical_setting:
        ret.append(
            Error(
                ErrorLevel.WARN,
                "physical_setting",
                "no_physical_setting_metadata",
                "missing physical setting metadata (null or empty)",
            )
        )

    if not manifest.objects:
        ret.append(
            Error(
                ErrorLevel.WARN,
                "objects",
                "no_objects_metadata",
                "missing object metadata (null or empty)",
            )
        )
    else:
        ret.extend(_check_objects(manifest))

    provided_capture_uids = set(manifest.captures.keys())
    for capture_uid, takes in manifest.takes.items():
        take = takes[0]
        if capture_uid not in provided_capture_uids:
            ret.append(
                Error(
                    ErrorLevel.ERROR,
                    take.take_id,
                    "take_has_incorrect_capture_uid",
                    f"take {take.take_id} has incorrect capture uid: {capture_uid} (typo or misisng capture in metadata?)",
                )
            )

    ret.extend(_check_video_metadata(manifest, metadata))
    ret.extend(_check_video_components(manifest))
    # TODO ret.extend(_check_files_exist(manfiest))
    return ret


def run_validation(
    manifest_dir: str,
    standard_metadata_folder: str,
    input_university: str,
    num_workers: int,
    expiry_time_sec: int,
    released_video_path: str,
    version: str,
    output_dir: str,
):
    global stream_path_mgr
    if stream_path_mgr is not None:
        raise AssertionError(
            "Don't use this method in multi-threaded contexts. Use processes instead."
        )

    stream_path_mgr = StreamPathMgr(expiration_sec=expiry_time_sec)

    # get access to metadata_folder
    if version.lower() == "egoexo":
        metadata = load_standard_metadata_files_egoexo(standard_metadata_folder)
        manifest = load_egoexo_manifest(manifest_dir)
        released_videos = None  # TODO: load ingested videos here

        errors = validate_egoexo_files(
            university=input_university,
            manifest=manifest,
            metadata=metadata,
            num_workers=num_workers,
        )
    else:
        assert version.lower() == "ego4d", "expected ego4d as version"
        metadata = load_standard_metadata_files(standard_metadata_folder)
        manifest = load_manifest(manifest_dir)
        released_videos = load_released_video_files(released_video_path)

        errors = validate_ego4d_files(
            university=input_university,
            manifest=manifest,
            metadata=metadata,
            num_workers=num_workers,
        )

    errors_df, summary_df = summarize_errors(
        errors=errors,
        released_videos=released_videos,
        university=input_university,
    )
    if len(summary_df) > 0:
        summary_df.to_csv("/dev/stdout", index=False)
    else:
        print("No errors")

    print(f"Writing to directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    errors_df.to_csv(
        pathmgr.open(os.path.join(output_dir, "errors.csv"), "w"), index=False
    )
    summary_df.to_csv(
        pathmgr.open(os.path.join(output_dir, "summary.csv"), "w"), index=False
    )

    # NOTE: this is not the best solution, but we're likely going to be running
    # this once per python interpreter context
    stream_path_mgr = None
