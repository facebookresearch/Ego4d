#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

INPUT_FILE=${INPUT_FILE:-./narrations_v2_7-27-21.json}
FIRST_X="${1:-2}"

if [[ $FIRST_X = "ALL" ]]
then
  jq -r 'keys | @csv' $INPUT_FILE | tr -d '"' | tr , ' '
else
  jq -r --arg FIRST_X "$FIRST_X"  'keys | .[0:($FIRST_X|tonumber)] | @csv' $INPUT_FILE | tr -d '"' | tr , ' '
fi
