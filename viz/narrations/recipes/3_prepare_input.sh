#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

INPUT_FILE=${INPUT_FILE:-./narrations_v2_7-27-21.json}
VID_ROOT=${VID_ROOT:-~/ego4d}
MODE="${1:ALL}"
BASE_PATH=/assets/

if [[ $MODE = "ALL" ]]
then
  jq -c --arg BASE_PATH "$BASE_PATH" 'to_entries[] | {file: ($BASE_PATH+"full_scale/"+.key+".mp4"), img: ($BASE_PATH+"viz/"+.key+"_small.jpg"), uid: .key, info: { type: "TIME_SEGMENTATION", role: "RESULT", payload: .value.narration_pass_1.narrations | map({start_time: .timestamp_sec, end_time: .timestamp_sec, label: .narration_text, id: .annotation_uid}) } }' $INPUT_FILE
else
  xargs -n 1 -I {} jq --arg BASE_PATH "$BASE_PATH" --arg ID "{}" '{info: { payload: .[$ID].narration_pass_1.narrations , type: "TIME_SEGMENTATION", role: "RESULT" }, file: ($BASE_PATH+"full_scale/"+$ID+".mp4"), img: ($BASE_PATH+"viz/"+$ID+"_small.jpg"), uid: $ID }' $INPUT_FILE  | jq -c '.info.payload |= map({start_time: .timestamp_sec, end_time: .timestamp_sec, label: .narration_text, id: .annotation_uid})'
fi

# jq -c "to_entries[]" ../narrations_v2_7-27-21.json
# jq -c '{file: ("/assets"+.key+".mp4"), uid: .key, info: { type: "TIME_SEGMENTATION", role: "RESULT", payload: .value.narration_pass_1.narrations | map({start_time: .timestamp_sec, end_time: .timestamp_sec, label: .narration_text, id: .annotation_uid}) } }' ./to_entries.json > final.json

# jq -c 'to_entries[] | {file: ("/assets/"+.key+".mp4"), uid: .key, info: { type: "TIME_SEGMENTATION", role: "RESULT", payload: .value.narration_pass_1.narrations | map({start_time: .timestamp_sec, end_time: .timestamp_sec, label: .narration_text, id: .annotation_uid}) } }' $INPUT_FILE
