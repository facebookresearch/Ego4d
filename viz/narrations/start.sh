#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
set -e

# Configuration options:
VID_ROOT=${VID_ROOT:-~/ego4d/}
REVIEW_DIR=${REVIEW_DIR:-./review/build/}
PORT=${PORT:-3030}

# This is the preprocessed data that will drive the review interface:
INPUT_FILE=${INPUT_FILE:-$VID_ROOT/v1/viz/preprocessed_narrations_input.jsonl}

# Check that Mephisto is installed:
if ! command -v mephisto &> /dev/null
then
    echo "Mephisto could not be found. Install with: pip install mephisto";
    exit
fi

# Check that the preprocessed data exists, if not attempt to download it:
if [ -f "$INPUT_FILE" ]; then
    echo "Preprocessed file found, using $INPUT_FILE"
    true
else
    # If the file cannot be found within the ego4d viz dataset location,
    # we will need to download the dataset first:
    echo "Preprocessed file not found, downloading the 'viz' dataset using the Ego4D CLI to $VID_ROOT..."
    python -m ego4d.cli.cli --yes --datasets viz --output_directory $VID_ROOT
fi

if [ -f "$INPUT_FILE" ]; then
    cat $INPUT_FILE | VID_ROOT=$VID_ROOT REVIEW_DIR=$REVIEW_DIR PORT=$PORT ./recipes/4_review.sh
else
    # If the file still cannot be found, it's an error
    echo "Error: $INPUT_FILE does not exist."
    exit 1
fi
