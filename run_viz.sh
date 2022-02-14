#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
set -e

# Help Screen
HelpScreen()
{
   # Display Help
   echo "Downloads and Starts Mephisto Visualization of Ego4D Data"
   echo
   echo "usage: ./run_viz.sh [options]"
   echo "  options:"
   echo "  -h  Print help screen"
   echo "  -o <dir>  Overrides video root dir (default ~/ego4d_data/)"
   echo "  -r <dir>  Overrides review build dir (default ./viz/narrations/review/build/)"
   echo "  -p <dir>  Run server on a different port (default 3030)"
   echo
}

# Handle CMD Options
while getopts ":ho:r:p:" option; do
   case $option in
      h) # display help
         HelpScreen
         exit;;
      o) # Enter a vid root dir
         VID_ROOT=$OPTARG;;
      r) # Enter a review build dir
         REVIEW_DIR=$OPTARG;;
      p) # Pick an alternate port
         PORT=$OPTARG;;
     \?) # Invalid option
         echo "Error: Invalid option, please check the usage:"
         HelpScreen
         exit;;
   esac
done

# Configuration options:
VID_ROOT=${VID_ROOT:-~/ego4d_data/}
REVIEW_DIR=${REVIEW_DIR:-./viz/narrations/review/build/}
PORT=${PORT:-3030}

# If there's a ~/ starting any paths, they need to be expanded for file checks to work
VID_ROOT=${VID_ROOT/#~\//$HOME\/}
REVIEW_DIR=${REVIEW_DIR/#~\//$HOME\/}

# This is the preprocessed data that will drive the review interface:
INPUT_FILE=${INPUT_FILE:-$VID_ROOT/v1/viz/preprocessed_narrations_input.jsonl}

# Check that Mephisto is installed:
if ! command -v mephisto &> /dev/null
then
    echo "Mephisto could not be found. Install with: pip install mephisto";
    exit
fi

# Check that the preprocessed data exists, if not attempt to download it:
if [ -f $INPUT_FILE ]; then
    echo "Preprocessed file found, using $INPUT_FILE"
    true
else
    # If the file cannot be found within the ego4d viz dataset location,
    # we will need to download the dataset first:
    echo "Preprocessed file not found, downloading the 'viz' dataset using the Ego4D CLI to $VID_ROOT..."
    python -m ego4d.cli.cli --yes --datasets viz --output_directory $VID_ROOT
fi

if [ -f $INPUT_FILE ]; then
    cat $INPUT_FILE | VID_ROOT=$VID_ROOT REVIEW_DIR=$REVIEW_DIR PORT=$PORT ./viz/narrations/recipes/4_review.sh
else
    # If the file still cannot be found, it's an error
    echo "Error: $INPUT_FILE does not exist."
    exit 1
fi
