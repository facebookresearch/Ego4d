from types import AuxiliaryVideoComponentDataFile, VideoComponentFile, VideoMetadata

import boto3
import validate
from ffmpeg_utils import VideoInfo

bucket_name = "ego4d-universityaf"
broken_obj = "Participant 4/0249/CMUA_GH010249_04_GOPRO8_16_KNITTING_7_7_2021.MP4"
normal_obj = "Participant 4/0278/CMUA_GH010278_04_GOPRO8_16_KNITTING_7_7_2021.MP4"
uid1 = "123"
uid2 = "456"

metadata = VideoMetadata(
    university_video_id=uid1,
    university_video_folder_path=f"s3://{bucket_name}/Participant 4/0278/",
    number_video_components=2,
    start_date_recorded_utc=None,
    recording_participant_id=0,
    device_id=0,
    video_device_settings={},
    physical_setting_id="0",
    video_scenario_ids=[],
)

component1 = VideoComponentFile(
    university_video_id=uid1,
    video_component_relative_path="CMUA_GH010278_04_GOPRO8_16_KNITTING_7_7_2021.MP4",
    component_index=0,
    is_redacted=False,
    start_date_recorded_utc=None,
    compression_settings={},
    includes_audio=False,
    component_metadata={},
    deidentification_metadata={},
)

component2 = VideoComponentFile(
    university_video_id=uid1,
    video_component_relative_path="2.mp4",
    component_index=2,
    is_redacted=False,
    start_date_recorded_utc=None,
    compression_settings={},
    includes_audio=False,
    component_metadata={},
    deidentification_metadata={},
)

aux1 = AuxiliaryVideoComponentDataFile(
    university_video_id=uid1,
    video_component_relative_path="s3://ego4d-georgiatech/object/key/123/1.mp4",
    component_index=2,
    component_type_id=0,
)

aux2 = AuxiliaryVideoComponentDataFile(
    university_video_id=uid1,
    video_component_relative_path="s3://ego4d-georgiatech/object/key/123/2.mp4",
    component_index=2,
    component_type_id=0,
)

s3 = boto3.client("s3")

aux_dict = {uid1: [aux1, aux2]}


def test_validate_video_components():
    meta_dict = {uid1: metadata}
    component_dict = {uid1: [component1]}

    err = []
    validate._validate_video_components(s3, bucket_name, meta_dict, component_dict, err)
    assert err[0].description == "video_metadata has 1 components when it should have 2"

    component_dict[uid1].append(component2)
    validate._validate_video_components(s3, bucket_name, meta_dict, component_dict, err)
    assert (
        err[1].description
        == "video_metadata has component index 2 when it should have 1"
    )
    assert (
        err[2].description
        == "video s3://ego4d-universityaf/Participant 4/0278/2.mp4 doesn't exist in bucket"
    )

    component_dict = {uid2: [component2, component1]}
    validate._validate_video_components(s3, bucket_name, meta_dict, component_dict, err)
    assert (
        err[3].description
        == "video_component_file points to video_metadata at non-existent university_video_id"
    )


def test_get_videos():
    err = []
    video_info_param = [(uid1, normal_obj), (uid2, broken_obj)]
    video_info_dict = validate._get_videos(s3, bucket_name, video_info_param, err)

    assert (
        err[0].description
        == f"video s3://{bucket_name}/{broken_obj} can't be read by FFMPEG"
    )
    assert len(err) == 1
    assert len(video_info_dict) == 1

    return video_info_dict


def test_validate_mp4(video_info_dict):
    err = []
    validate._validate_mp4(video_info_dict, err)
    err = []
    vinfo = video_info_dict[uid1][0]
    v1 = VideoInfo(
        sample_height=vinfo.sample_height,
        sample_width=vinfo.sample_height + 10,
        vcodec="wrong",
        acodec="wrong",
        sar=vinfo.sar / 2,
        fps=None,
        dar=vinfo.dar / 2,
        mp4_duration=None,
        video_time_base=vinfo.video_time_base / 2,
        vstart=vinfo.vstart,
        astart=vinfo.astart,
        vduration=vinfo.vduration,
        aduration=vinfo.aduration,
    )

    v2 = VideoInfo(
        sample_height=vinfo.sample_height,
        sample_width=vinfo.sample_width,
        vcodec=vinfo.vcodec,
        acodec=vinfo.acodec,
        sar=vinfo.sar,
        fps=vinfo.fps + 1,
        dar=vinfo.dar,
        mp4_duration=0,
        video_time_base=vinfo.video_time_base,
        vstart=vinfo.vstart + 1,
        astart=vinfo.astart,
        vduration=vinfo.vduration,
        aduration=vinfo.aduration,
        rotate=True,
    )

    video_info_dict[uid1][1] = v1
    validate._validate_mp4(video_info_dict, err)

    assert err[0].description == "component 0 has no rotation information"
    assert err[1].description == "component 0 has width < height without rotation"
    assert err[2].description == "component 1 has no rotation information"
    assert err[3].description == "component 1 has null fps value"
    assert err[4].description == "component 1 has no mp4 duration"
    assert err[5].description == "inconsistent video codec"
    assert err[6].description == "components with inconsistent width x height"
    assert err[7].description == "inconsistent video time base"
    assert err[8].description == "inconsistent sar"

    err = []
    video_info_dict[uid1][1] = v2
    validate._validate_mp4(video_info_dict, err)

    assert err[0].description == "component 0 has no rotation information"
    assert err[1].description == "component 0 has width < height without rotation"
    assert err[2].description == "component 1 has mismatching mp4 duration"
    assert err[3].description == "inconsistent video fps"


if __name__ == "__main__":
    test_validate_video_components()
    video_info_dict = test_get_videos()
    video_info_dict[uid1].append(video_info_dict[uid1][0])
    test_validate_mp4(video_info_dict)
