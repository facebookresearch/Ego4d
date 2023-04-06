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
#include <chrono>
#include <filesystem>
#include <optional>
#include <string>

namespace ark::datatools {

struct TrajectoryPose {
  // Timestamp of the measurement
  std::chrono::microseconds tracking_timestamp_us;

  // UTC Timestamp of the device image capture
  std::chrono::nanoseconds utcTimestamp;

  // Transformation from this device to some arbitrary world coordinate frame
  Sophus::SE3d T_world_device;

  // Translational velocity of the device with respect to the world in device frame
  Eigen::Vector3d deviceLinearVelocity_device = {0., 0., 0.};

  // Angular velocity with respect to the world in device frame
  Eigen::Vector3d angularVelocity_device = {0., 0., 0.};

  // Quality score: float between [0, 1] which describes how good the pose and dynamics are.
  //
  // qualityScore = 1: we are confident the pose and dynamics "good"
  // qualityScore = 0: we have no confidence on the pose and dynamics quality
  std::optional<float> qualityScore;

  // Estimated gravity direction in device frame
  std::optional<Eigen::Vector3d> gravity_odometry;

  // The Unique Identifier of the graph-connected component this pose belongs to
  std::optional<std::string> graphUid;
};

// A Trajectory is a list of Trajectory poses
using Trajectory = std::vector<TrajectoryPose>;

Trajectory readOpenLoop(const std::string& path);
Trajectory readCloseLoop(const std::string& path);

} // namespace ark::datatools
