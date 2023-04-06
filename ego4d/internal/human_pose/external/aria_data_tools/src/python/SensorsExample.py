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

import numpy as np

from pyark.datatools import sensors


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vrs",
        dest="vrs_path",
        type=str,
        required=True,
        help="path to vrs file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    #
    # Read calibration data from a VRS file
    #
    print("Attempting to read calibration data from: ", args.vrs_path)
    calib_str = sensors.getCalibStrFromFile(args.vrs_path)

    device = sensors.DeviceModel.fromJson(calib_str)
    print(f"Cameras: {device.getCameraLabels()}")
    print(f"IMUs: {device.getImuLabels()}")
    print(f"Magnetometers: {device.getMagnetometerLabels()}")
    print(f"Barometers: {device.getBarometerLabels()}")
    print(f"Microphones: {device.getMicrophoneLabels()}")

    #
    # Demonstrate how to use camera model

    # Create a 3D points and project and unproject it with a given camera
    camLabel = "camera-slam-left"
    p_slamLeft = np.array([3.0, 2.0, 1.0])
    uv_slamLeft = device.getCameraCalib(camLabel).projectionModel.project(p_slamLeft)
    print(
        f"Projecting 3D point {p_slamLeft} to image space of {camLabel}: "
        + f"{uv_slamLeft}."
    )
    p_slamLeft_convertBack = device.getCameraCalib(camLabel).projectionModel.unproject(
        uv_slamLeft
    )
    print(
        f"Unprojecting 2D pixel {uv_slamLeft} to 3D space in "
        + f"the frame of {camLabel}: {p_slamLeft_convertBack}."
    )

    # Transform points between sensor frames.
    imuLabel = "imu-left"
    p_imuLeft = device.transform(p_slamLeft, camLabel, imuLabel)
    print(
        f"Transforming {p_slamLeft} from {camLabel} frame to {imuLabel} "
        + f"frame: {p_imuLeft}"
    )

    # Rectifying points with the IMU accelerometer model.
    p_imuLeft_rect = device.getImuCalib(
        imuLabel
    ).accel.compensateForSystematicErrorFromMeasurement(p_imuLeft)

    print(
        f"Point {p_imuLeft} is rectified by the accelerometer model "
        + f"of {imuLabel} as: {p_imuLeft_rect}"
    )
