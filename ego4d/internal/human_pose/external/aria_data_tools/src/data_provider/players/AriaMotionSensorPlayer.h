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

#include <data_layout/aria/MotionSensorMetadata.h>
#include <vrs/RecordFormatStreamPlayer.h>

namespace ark {
namespace datatools {
namespace dataprovider {

using MotionCallback =
    std::function<bool(const vrs::CurrentRecord& r, vrs::DataLayout& dataLayout, bool verbose)>;

struct AriaMotionConfigRecord {
  uint32_t streamIndex;
  std::string deviceType;
  std::string deviceVersion;
  std::string deviceSerial;
  uint64_t deviceId;
  std::string sensorModel;
  double nominalRateHz;
  bool hasAccelerometer;
  bool hasGyroscope;
  bool hasMagnetometer;
  std::string factoryCalibration;
  std::string onlineCalibration;
  std::string description;
};

struct AriaMotionDataRecord {
  bool accelValid;
  bool gyroValid;
  bool magValid;
  double temperature;
  int64_t captureTimestampNs;
  int64_t arrivalTimestampNs;
  std::vector<float> accelMSec2;
  std::vector<float> gyroRadSec;
  std::vector<float> magTesla;
};

class AriaMotionSensorPlayer : public vrs::RecordFormatStreamPlayer {
 public:
  explicit AriaMotionSensorPlayer(vrs::StreamId streamId) : streamId_(streamId) {}
  AriaMotionSensorPlayer(const AriaMotionSensorPlayer&) = delete;
  AriaMotionSensorPlayer& operator=(const AriaMotionSensorPlayer&) = delete;
  AriaMotionSensorPlayer(AriaMotionSensorPlayer&&) = default;

  void setCallback(MotionCallback callback) {
    callback_ = callback;
  }

  const AriaMotionConfigRecord& getConfigRecord() const {
    return configRecord_;
  }

  const AriaMotionDataRecord& getDataRecord() const {
    return dataRecord_;
  }

  const vrs::StreamId& getStreamId() const {
    return streamId_;
  }

  double getNextTimestampSec() const {
    return nextTimestampSec_;
  }

  void setVerbose(bool verbose) {
    verbose_ = verbose;
  }

 private:
  bool onDataLayoutRead(const vrs::CurrentRecord& r, size_t blockIndex, vrs::DataLayout& dl)
      override;

  const vrs::StreamId streamId_;
  MotionCallback callback_ = [](const vrs::CurrentRecord&, vrs::DataLayout&, bool) { return true; };

  AriaMotionConfigRecord configRecord_;
  AriaMotionDataRecord dataRecord_;

  double nextTimestampSec_ = 0;
  bool verbose_ = false;
};

} // namespace dataprovider
} // namespace datatools
} // namespace ark
