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

#include <array>
#include <iostream>

#ifndef CSV_IO_NO_THREAD
#define CSV_IO_NO_THREAD
#endif
#include "fast-cpp-csv-parser/csv.h"
#include "trajectoryReader.h"

namespace ark::datatools {

constexpr std::array<const char*, 19> OpenLoopTrajectoryColumns = {
    "tracking_timestamp_us",
    "utc_timestamp_ns",
    "session_uid",
    "tx_odometry_device",
    "ty_odometry_device",
    "tz_odometry_device",
    "qx_odometry_device",
    "qy_odometry_device",
    "qz_odometry_device",
    "qw_odometry_device",
    "device_linear_velocity_x_odometry",
    "device_linear_velocity_y_odometry",
    "device_linear_velocity_z_odometry",
    "angular_velocity_x_device",
    "angular_velocity_y_device",
    "angular_velocity_z_device",
    "gravity_x_odometry",
    "gravity_y_odometry",
    "gravity_z_odometry"};

Trajectory readOpenLoop(const std::string& path) {
  io::CSVReader<OpenLoopTrajectoryColumns.size()> csv(path);
  // Read in the CSV header
  const auto readHeader = [&](auto&&... args) { csv.read_header(io::ignore_no_column, args...); };
  std::apply(readHeader, OpenLoopTrajectoryColumns);

  Trajectory trajectory;
  // Read each row and populate the trajectory with each recording
  TrajectoryPose pose;

  std::string session_uid;
  std::int64_t tracking_timestamp_us;
  std::int64_t utc_timestamp_ns;
  Eigen::Vector3d t_device;
  Eigen::Quaterniond q_device;
  Eigen::Vector3d gravity_odometry;

  while (csv.read_row(
      tracking_timestamp_us,
      utc_timestamp_ns,
      session_uid,
      t_device.x(),
      t_device.y(),
      t_device.z(),
      q_device.x(),
      q_device.y(),
      q_device.z(),
      q_device.w(),
      pose.deviceLinearVelocity_device.x(),
      pose.deviceLinearVelocity_device.y(),
      pose.deviceLinearVelocity_device.z(),
      pose.angularVelocity_device.x(),
      pose.angularVelocity_device.y(),
      pose.angularVelocity_device.z(),
      gravity_odometry.x(),
      gravity_odometry.y(),
      gravity_odometry.z())) {
    // Update compound object, and optional values
    pose.graphUid = session_uid;
    pose.T_world_device = Sophus::SE3d(q_device, t_device);
    pose.gravity_odometry = gravity_odometry;
    pose.tracking_timestamp_us = std::chrono::microseconds(tracking_timestamp_us);
    pose.utcTimestamp = std::chrono::nanoseconds(utc_timestamp_ns);
    trajectory.push_back(pose);
    pose = {}; // reset pose for the next reading
  }
  std::cout << "Loaded #poses records: " << trajectory.size() << std::endl;
  return trajectory;
}

constexpr std::array<const char*, 17> CloseLoopTrajectoryColumns = {
    "graph_uid",
    "tracking_timestamp_us",
    "utc_timestamp_ns",
    "tx_world_device",
    "ty_world_device",
    "tz_world_device",
    "qx_world_device",
    "qy_world_device",
    "qz_world_device",
    "qw_world_device",
    "device_linear_velocity_x_device",
    "device_linear_velocity_y_device",
    "device_linear_velocity_z_device",
    "angular_velocity_x_device",
    "angular_velocity_y_device",
    "angular_velocity_z_device",
    "quality_score"};

Trajectory readCloseLoop(const std::string& path) {
  io::CSVReader<CloseLoopTrajectoryColumns.size()> csv(path);
  // Read in the CSV header
  const auto readHeader = [&](auto&&... args) { csv.read_header(io::ignore_no_column, args...); };
  std::apply(readHeader, CloseLoopTrajectoryColumns);

  Trajectory trajectory;
  // Read each row and populate the trajectory with each recording
  TrajectoryPose pose;

  std::string graphUid;
  std::int64_t tracking_timestamp_us;
  std::int64_t utc_timestamp_ns;
  Eigen::Vector3d t_device;
  Eigen::Quaterniond q_device;
  float qualityScore;

  while (csv.read_row(
      graphUid,
      tracking_timestamp_us,
      utc_timestamp_ns,
      t_device.x(),
      t_device.y(),
      t_device.z(),
      q_device.x(),
      q_device.y(),
      q_device.z(),
      q_device.w(),
      pose.deviceLinearVelocity_device.x(),
      pose.deviceLinearVelocity_device.y(),
      pose.deviceLinearVelocity_device.z(),
      pose.angularVelocity_device.x(),
      pose.angularVelocity_device.y(),
      pose.angularVelocity_device.z(),
      qualityScore)) {
    auto& trajectoryPose = trajectory.emplace_back();
    trajectoryPose = pose;
    // Update compound object, and optional values
    pose.graphUid = graphUid;
    pose.qualityScore = qualityScore;
    trajectoryPose.T_world_device = Sophus::SE3d(q_device, t_device);
    trajectoryPose.tracking_timestamp_us = std::chrono::microseconds(tracking_timestamp_us);
    trajectoryPose.utcTimestamp = std::chrono::nanoseconds(utc_timestamp_ns);

    pose = {}; // reset pose for the next reading
  }
  std::cout << "Loaded #poses records: " << trajectory.size() << std::endl;
  return trajectory;
}

} // namespace ark::datatools
