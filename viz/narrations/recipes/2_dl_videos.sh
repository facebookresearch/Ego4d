#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

VID_ROOT=${VID_ROOT:-~/ego4d}
OUTPUT=${1:LOG}

if [[ $OUTPUT = "LOG" ]]
then
  xargs -n 1 -I {} python -m ego4d.cli.cli --yes --datasets full_scale viz --output_directory $VID_ROOT --video_uids {}
else
  cat
  xargs -n 1 -I {} python -m ego4d.cli.cli --yes --datasets full_scale viz --output_directory $VID_ROOT --video_uids {} > /dev/null
fi
