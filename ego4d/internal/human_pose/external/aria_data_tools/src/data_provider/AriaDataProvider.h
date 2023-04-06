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

#include <vrs/StreamId.h>
#include <map>
#include <optional>
#include <set>
#include "models/DeviceModel.h"

namespace ark {
namespace datatools {
namespace dataprovider {

class AriaDataProvider {
 protected:
  using optional_audio_reference_vector =
      std::optional<std::reference_wrapper<const std::vector<int32_t>>>;
  using optional_img_buffer_reference_vector =
      std::optional<std::reference_wrapper<const std::vector<uint8_t>>>;

 public:
  AriaDataProvider() = default;
  virtual ~AriaDataProvider() = default;
  virtual bool open(const std::string& sourcePath) = 0;
  virtual void setStreamPlayer(const vrs::StreamId& streamId) = 0;
  virtual bool tryFetchNextData(
      const vrs::StreamId& streamId,
      double currentTimestampSec = std::numeric_limits<double>::max()) = 0;
  virtual void* getImageBuffer(const vrs::StreamId& streamId) const = 0;
  virtual optional_img_buffer_reference_vector getImageBufferVector(
      const vrs::StreamId& streamId) const = 0;
  virtual uint32_t getImageWidth(const vrs::StreamId& streamId) const = 0;
  virtual uint32_t getImageHeight(const vrs::StreamId& streamId) const = 0;
  virtual double getFastestNominalRateHz() = 0;
  virtual double getFirstTimestampSec() = 0;
  // imu data
  virtual Eigen::Vector3f getMotionAccelData(const vrs::StreamId& streamId) const = 0;
  virtual Eigen::Vector3f getMotionGyroData(const vrs::StreamId& streamId) const = 0;
  // barometer data
  virtual double getBarometerPressure() const = 0;
  virtual double getBarometerTemperature() const = 0;
  // magnetometer data
  virtual Eigen::Vector3f getMagnetometerData() const = 0;
  // audio data
  virtual optional_audio_reference_vector getAudioData() const = 0;
  virtual uint8_t getAudioNumChannels() const = 0;

  virtual bool atLastRecords() = 0;
  virtual bool loadDeviceModel() = 0;
  virtual bool streamExistsInSource(const vrs::StreamId& streamId) = 0;

  virtual void setImagePlayerVerbose(const vrs::StreamId& streamId, bool verbose) = 0;
  virtual void setMotionPlayerVerbose(const vrs::StreamId& streamId, bool verbose) = 0;
  virtual void setWifiBeaconPlayerVerbose(bool verbose) = 0;
  virtual void setAudioPlayerVerbose(bool verbose) = 0;
  virtual void setBluetoothBeaconPlayerVerbose(bool verbose) = 0;
  virtual void setGpsPlayerVerbose(bool verbose) = 0;
  virtual void setBarometerPlayerVerbose(bool verbose) = 0;
  virtual void setTimeSyncPlayerVerbose(bool verbose) = 0;

  const datatools::sensors::DeviceModel& getDeviceModel() const {
    return deviceModel_;
  }

 protected:
  std::set<vrs::StreamId> providerStreamIds_;
  datatools::sensors::DeviceModel deviceModel_;

  std::string sourcePath_;
};
} // namespace dataprovider
} // namespace datatools
} // namespace ark
