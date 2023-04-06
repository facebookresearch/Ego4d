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

#include <data_layout/aria/BluetoothBeaconMetadata.h>
#include <vrs/RecordFormatStreamPlayer.h>

namespace ark {
namespace datatools {
namespace dataprovider {

using BluetoothBeaconCallback =
    std::function<bool(const vrs::CurrentRecord& r, vrs::DataLayout& dataLayout, bool verbose)>;

struct AriaBluetoothBeaconConfigRecord {
  uint32_t streamId;
  double sampleRateHz;
};

struct AriaBluetoothBeaconDataRecord {
  int64_t systemTimestampNs;
  int64_t boardTimestampNs;
  int64_t boardScanRequestStartTimestampNs;
  int64_t boardScanRequestCompleteTimestampNs;
  std::string uniqueId;
  float txPower;
  float rssi;
  float freqMhz;
};

class AriaBluetoothBeaconPlayer : public vrs::RecordFormatStreamPlayer {
 public:
  explicit AriaBluetoothBeaconPlayer(vrs::StreamId streamId) : streamId_(streamId) {}
  AriaBluetoothBeaconPlayer(const AriaBluetoothBeaconPlayer&) = delete;
  AriaBluetoothBeaconPlayer& operator=(const AriaBluetoothBeaconPlayer&) = delete;
  AriaBluetoothBeaconPlayer(AriaBluetoothBeaconPlayer&&) = default;

  void setCallback(BluetoothBeaconCallback callback) {
    callback_ = callback;
  }

  const AriaBluetoothBeaconConfigRecord& getConfigRecord() const {
    return configRecord_;
  }

  const AriaBluetoothBeaconDataRecord& getDataRecord() const {
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
  BluetoothBeaconCallback callback_ = [](const vrs::CurrentRecord&, vrs::DataLayout&, bool) {
    return true;
  };

  AriaBluetoothBeaconConfigRecord configRecord_;
  AriaBluetoothBeaconDataRecord dataRecord_;

  double nextTimestampSec_ = 0;
  bool verbose_ = false;
};

} // namespace dataprovider
} // namespace datatools
} // namespace ark
