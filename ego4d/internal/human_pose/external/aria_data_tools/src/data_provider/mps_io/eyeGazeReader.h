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

#pragma once

#include <sophus/se3.hpp>
#include <Eigen/Core>
#include <chrono>
#include <filesystem>
#include <map>
#include <optional>
#include "models/DeviceModel.h"

namespace ark::datatools {

struct EyeGaze {
  // EyeGaze data is defined in CPF (Central Pupil Frame).
  // Coordinates of a unit vector along the gaze direction.
  Eigen::Vector3d gaze_vector = {0., 0., 0.};

  // Confidence in the output [0, 1]
  // Smaller values indicate higher confidence
  float uncertainty = 0.f;
};

// Timestamp is aligned with device image capture
using TemporalEyeGazeData = std::map<std::chrono::microseconds, EyeGaze>;

TemporalEyeGazeData readEyeGaze(const std::string& path);

void ProjectEyeGazeRayInCamera(
    const std::string& cameraString,
    const sensors::DeviceModel& deviceModel,
    const Eigen::Vector3d& eyeGazeVector,
    const double minDepth,
    const double maxDepth,
    const double samples,
    std::map<double, Eigen::Vector2d>& camProjectionPerDepth,
    const int cameraWidth = -1,
    const int cameraHeight = -1);

} // namespace ark::datatools
