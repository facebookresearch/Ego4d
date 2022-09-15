# How to use the validation script

## Run the below command in cli to validate
```
python -m ego4d.internal.cli \
       -i "<s3_video_metadata_bucket>" \
       -mf "<s3_other_metadata_bucket>" \
       -ed "<error_details_path>" \
       -es "<error_summary_path>" \
```
- s3_video_metadata_bucket: the s3 bucket where files like video_metadata.csv, auxiliary_video_component_data_file.csv, video_component_file.csv, video_metadata.csv were stored
- s3_other_metadata_bucket: the s3 bucket where files like component_type.csv, device.csv, scenarios.csv were stored
- error_details_path: the path where you want to save the validation error details to
- error_summary_path: the path where you want to save the validation error summary to

## Example
```
python -m ego4d.internal.cli \
       -i "s3://ego4d-unict/metadata_v7" \
       -mf "s3://ego4d-consortium-sharing/internal/standard_metadata_v10" \
       -ed "error_details" \
       -es "error_summary" \
```

# Types of validation errors
### `validate_video_components`, `validate_synchronized_videos`, `validate_auxilliary_videos`, `validate_annotations`
- video_not_found_in_video_metadata_error: provided video_id in one of [video_component_dict, synchronized_videos.csv, auxiliary_video_component_dict, annotations_dict], can’t be found in video_metadata
- video_component_length_inconsistent_error: length of video components is different in video_metadata_dict vs. the video provided (in [video_component_dict, auxiliary_video_component_dict]
- video_component_wrong_index_error: component_index isn’t conforming the rule (incremental and starts at 0)
- bucket_name_inconsistent_error: video_metadata has a different bucket name than video_component_dict
- path_does_not_exist_error: can’t find the relevant file in S3 given the video components
- component_type_id_not_found_error: auxiliary component's component_type_id does not exist in component_types
- empty_video_component_relative_path_error: an entry for the video_id has empty video component relative path
### `validate_mp4`
- no_video_info_for_component: there’s no metadata provided for the component
- no_video_or_audio_stream_duration: the component has no video or audio stream duration metadata
- component_having_width_lt_height_error: the video component has width < height without rotation
- missing_fps_info_warning: the video component doesn’t have fps info
- missing_mp4_duration_info_warning: the video component doesn’t have mp4 duration info
- mp4_duration_too_large_or_small_warning: the video component has an mp4 duration that's too large or small
- inconsistent_video_codec_error
- inconsistent_audio_codec_error
- inconsistent_rotation_error
- inconsistent_width_height_pair_error
- inconsistent_video_time_base_error
- inconsistent_sar_warning
- inconsistent_video_fps_warning
- participant_not_found_error: participant_id not found in the video metadata
### `validata_video_metadata`
- duplicate_video_id_error: duplicate video_id found in metadata
participant_id_not_found_error: participant_id in video metadata not found in participant_id dict
- null_participant_id_warning: participant_id in video metadata is null
- scenario_id_not_found_error: scenario_id in video metadata not found in scenarios.csv
- device_id_not_found_error: device_id in video metadata not found in devices.csv
- physical_setting_id_not_found_error: physical_setting_id in video metadata not found in physical_setting.csv
- video_not_found_in_video_components_error: provided video_id in video_metadata can’t  be found in video_component_dict
get_videos
- ffmpeg_cannot_read_error: FFMPEG cannot read the file given the S3 path
video_does_not_exist_in_bucket_error: video path provided can’t be found in bucket
