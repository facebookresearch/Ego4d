/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cereal/external/rapidjson/document.h>
#include <array>
#include <fstream>
#include <iostream>

#ifndef CSV_IO_NO_THREAD
#define CSV_IO_NO_THREAD
#endif
#include "eyeGazeReader.h"
#include "fast-cpp-csv-parser/csv.h"

namespace ark::datatools {

constexpr std::array<const char*, 5> EyeGazeColumns =
    {"tracking_timestamp_us", "gaze_vector_x", "gaze_vector_y", "gaze_vector_z", "uncertainty"};

TemporalEyeGazeData readEyeGaze(const std::string& path) {
  io::CSVReader<EyeGazeColumns.size()> csv(path);
  // Read in the CSV header
  const auto readHeader = [&](auto&&... args) { csv.read_header(io::ignore_no_column, args...); };
  std::apply(readHeader, EyeGazeColumns);

  TemporalEyeGazeData eyeGazeSequence;
  // Read each row and populate the trajectory with each recording
  EyeGaze eyeGaze;

  std::int64_t tracking_timestamp_us;

  while (csv.read_row(
      tracking_timestamp_us,
      eyeGaze.gaze_vector.x(),
      eyeGaze.gaze_vector.y(),
      eyeGaze.gaze_vector.z(),
      eyeGaze.uncertainty)) {
    eyeGazeSequence[std::chrono::microseconds(tracking_timestamp_us)] = eyeGaze;
    eyeGaze = {}; // reset gaze data for the next reading
  }
  std::cout << "Loaded #eyegaze records: " << eyeGazeSequence.size() << std::endl;
  return eyeGazeSequence;
}

void ProjectEyeGazeRayInCamera(
    const std::string& cameraString,
    const sensors::DeviceModel& deviceModel,
    const Eigen::Vector3d& eyeGazeVector,
    const double minDepth,
    const double maxDepth,
    const double samples,
    std::map<double, Eigen::Vector2d>& camProjectionPerDepth,
    const int cameraWidth,
    const int cameraHeight) {
  // Compute projection of the EyeGaze vector in chosen camera
  camProjectionPerDepth.clear();

  const bool projectionInCPF = true;
  // Since EyeGaze results is defined in CPF you can project in Aria cameras:
  // 1. by projecting directly for the CAD camera calib
  // Or
  // 2. by computing the device to CPF compensation (relative transform
  // between the CAD and your own glasses calibration)

  if (!deviceModel.getCameraCalib(cameraString)) {
    return;
  }

  if (maxDepth < minDepth) {
    return;
  }

  // Get back camera poses of interest
  const auto vrsCamera = deviceModel.getCameraCalib(cameraString);
  const auto cadCamera = deviceModel.getCADSensorPose(cameraString);

  // If you want to compensate for the CPF
  const auto T_device_cpf = vrsCamera->T_Device_Camera * cadCamera->inverse();
  const auto T_cpf_cam = T_device_cpf.inverse() * vrsCamera->T_Device_Camera;

  // Build the list of 3D points we will have to project in the camera
  // 1. list depth we need to use
  // 2. Sample the depth along the ray
  // 3. Project in the camera

  // 1.
  std::vector<double> depthSamples;
  for (double depth = minDepth; depth <= maxDepth; depth += ((maxDepth - minDepth) / samples)) {
    depthSamples.push_back(depth);
  }

  // 2.
  for (const auto itDepth : depthSamples) {
    const Eigen::Vector3d sampledDepth = itDepth * eyeGazeVector;

    const auto usedCameraTransform = projectionInCPF ? *cadCamera : T_cpf_cam;
    auto projection =
        vrsCamera->projectionModel.project(usedCameraTransform.inverse() * sampledDepth);

    // Is projection inside the image or not
    if (cameraWidth > 0 && cameraHeight > 0) {
      if (!(projection.x() > 0 && projection.x() < cameraWidth && projection.y() > 0 &&
            projection.y() < cameraHeight)) {
        // Projection is outside image, we continue to the next point
        continue;
      }
    }
    camProjectionPerDepth[itDepth] = projection;
  }
}

} // namespace ark::datatools
