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

#include <deque>
#include "data_provider/AriaVrsDataProvider.h"
#include "eyeGazeReader.h"
#include "visualization/AriaViewerBase.h"

namespace ark::datatools::visualization {

class AriaViewer : public AriaViewerBase {
 public:
  AriaViewer(
      datatools::dataprovider::AriaDataProvider* dataProvider,
      int width,
      int height,
      const std::string& eyeTrackingFilepath,
      const std::string& name = "AriaEyeGazeViewer");
  ~AriaViewer() override = default;
  void run() override;

  bool readData(double currentTimestampSec) override;

 public:
  // Interface to store temporally sorted EyeGaze data record
  using EyeGazeDataRecords = std::map<std::chrono::microseconds, std::pair<Eigen::Vector3d, float>>;

 private:
  // Store all record for search (timestamp sorted)
  EyeGazeDataRecords eyeGazeData_;
  // Last current valid EyeGaze recording
  EyeGazeDataRecords::mapped_type lastEyeGazeRecord_;
  // A rolling buffer history of EyeGaze yaw, pitch recordings
  std::deque<Eigen::Vector2d> eyeGazeHistory_;

  // Store current Timestamp relative to the Aria sequence we are at
  std::int64_t currentTimestamp_ = 0;
};

} // namespace ark::datatools::visualization
