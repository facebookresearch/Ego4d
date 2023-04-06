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

#include "AriaViewerBase.h"

#include "PilotDataset.h"

namespace ark {
namespace datatools {
namespace visualization {

class PilotDatasetViewer : public AriaViewerBase {
 public:
  PilotDatasetViewer(
      dataprovider::PilotDatasetProvider* dataProvider,
      int width,
      int height,
      const std::string& name = "PilotDatasetViewer",
      int id = 0);

  ~PilotDatasetViewer() override = default;

  void run() override;

  std::pair<double, double> initDataStreams(
      const std::vector<vrs::StreamId>& kImageStreamIds,
      const std::vector<vrs::StreamId>&,
      const std::vector<vrs::StreamId>&) override;

 private:
  void setPose(const std::optional<Sophus::SE3d>& T_World_ImuLeft);

  void drawTraj();
  void drawRigs(
      bool showRig3D = true,
      bool showLeftCam3D = true,
      bool showRightCam3D = true,
      bool showRgbCam3D = true,
      int camSparsity = 4);

  // Save transformation to move data to IMU Left (Aria Pilot dataset Reference system)
  std::unordered_map<vrs::RecordableTypeId, std::unordered_map<uint16_t, Sophus::SE3d>>
      T_ImuLeft_cameraMap_;

  std::vector<Sophus::SE3d> T_World_Camera_; // Position of the Aria device
  Sophus::SE3d T_Viewer_World_; // Setup initial position

  const dataprovider::PilotDatasetProvider* dataProvider_;
};
} // namespace visualization
} // namespace datatools
} // namespace ark
