# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import sys
import time
from threading import Thread

import pyark.datatools as datatools


def since(timestampSec):
    return time.time() - timestampSec


def read_records(data_provider, viewer):
    start = time.time()
    [current_timestamp_sec, fastest_nominal_rate_hz] = viewer.initDataStreams()
    wait_time_sec = (1.0 / fastest_nominal_rate_hz) / 10
    while not data_provider.atLastRecords():
        if viewer.readData(current_timestamp_sec):
            current_timestamp_sec += wait_time_sec
            this_wait_time_sec = wait_time_sec - since(start)
            if this_wait_time_sec > 0:
                time.sleep(this_wait_time_sec / viewer.getPlaybackSpeedFactor())
            start = time.time()
    print("Finished reading records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualization example.")
    parser.add_argument(
        "--vrs_path", type=str, help="Path to the input VRS file or folder."
    )
    parser.add_argument(
        "--pose_path",
        type=str,
        help="Optional path to the input pose file.",
        default="",
    )
    parser.add_argument(
        "--eyetracking_path",
        type=str,
        help="Optional path to the input eyetracking file.",
        default="",
    )
    parser.add_argument(
        "--speechtotext_path",
        type=str,
        help="Optional path to the input speech-to-text file.",
        default="",
    )
    args = parser.parse_args()
    data_provider = datatools.dataprovider.AriaVrsDataProvider()
    if data_provider.open(
        args.vrs_path, args.pose_path, args.eyetracking_path, args.speechtotext_path
    ):
        print("Opened the VRS file successfully")
    else:
        print("Couldn't open the VRS file")
        sys.exit()

    viewer = datatools.visualization.AriaViewer(data_provider, 700, 800)
    reader_thread = Thread(target=read_records, args=(data_provider, viewer))
    reader_thread.start()

    viewer.run()
    reader_thread.join()
