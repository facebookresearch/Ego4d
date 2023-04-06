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
#include "AriaVrsDataProvider.h"
#include "speech_to_text_datum.h"

namespace ark::datatools::dataprovider {

// PilotDatasetProvider provides the necessary APIs to load and access timesync data from the Aria
// CVPR Pilot dataset
// 3 kind of metadata is available:
// - Poses (based on IMULeft)
// - EyeTracking
// - SpeechToText
class PilotDatasetProvider : public AriaVrsDataProvider {
 public:
  // Initialize Additional data, VRS will still be load using the open(X) function
  PilotDatasetProvider(
      const std::string& posePath = "",
      const std::string& eyetrackingPath = "",
      const std::string& speechToTextPath = "");

  //
  // Time aligned serving of Dataset metadata
  //

  // eyetracking data time-aligned serving
  std::optional<Eigen::Vector2f> getEyetracksOnRgbImage() const;
  // speechToText data time-aligned serving
  std::optional<SpeechToTextDatum> getSpeechToText() const;
  // aria pose side-loading time-aligned serving
  std::optional<Sophus::SE3d> getPose() const;

  // return pose aligned with the current SLAM camera timestamp
  std::optional<Sophus::SE3d> getLatestPoseOfStream(const vrs::StreamId& streamId);
  std::optional<Sophus::SE3d> getPoseOfStreamAtTimestampNs(
      const vrs::StreamId& streamId,
      const int64_t timestampNs);

 protected:
  std::map<int64_t, Sophus::SE3d> imuLeftPoses_;
  std::map<int64_t, Eigen::Vector2f> eyetracksOnRgbImage_;
  std::map<int64_t, SpeechToTextDatum> speechToText_;

  bool hasEyetracks_ = false;
  bool hasSpeechToText_ = false;
  bool hasPoses_ = false;
};

} // namespace ark::datatools::dataprovider
