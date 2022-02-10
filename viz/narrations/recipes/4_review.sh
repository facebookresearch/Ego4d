#!/bin/bash

VID_ROOT=${VID_ROOT:-~/ego4d}
REVIEW_DIR=${REVIEW_DIR:-../review/build/}
PORT=${PORT:-3030}

mephisto review $REVIEW_DIR --json --stdout --assets $VID_ROOT/v1/ --all --port $PORT
