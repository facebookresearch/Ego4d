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

#include <data_layout/aria/WifiBeaconMetadata.h>
#include <vrs/RecordFormatStreamPlayer.h>

namespace ark {
namespace datatools {
namespace dataprovider {

using WifiBeaconCallback =
    std::function<bool(const vrs::CurrentRecord& r, vrs::DataLayout& dataLayout, bool verbose)>;

struct AriaWifiBeaconConfigRecord {
  uint32_t streamId;
};

struct AriaWifiBeaconDataRecord {
  int64_t systemTimestampNs;
  int64_t boardTimestampNs;
  int64_t boardScanRequestStartTimestampNs;
  int64_t boardScanRequestCompleteTimestampNs;
  std::string ssid;
  std::string bssidMac;
  float rssi;
  float freqMhz;
  std::vector<float> rssiPerAntenna;
};

class AriaWifiBeaconPlayer : public vrs::RecordFormatStreamPlayer {
 public:
  explicit AriaWifiBeaconPlayer(vrs::StreamId streamId) : streamId_(streamId) {}
  AriaWifiBeaconPlayer(const AriaWifiBeaconPlayer&) = delete;
  AriaWifiBeaconPlayer& operator=(const AriaWifiBeaconPlayer&) = delete;
  AriaWifiBeaconPlayer(AriaWifiBeaconPlayer&&) = default;

  void setCallback(WifiBeaconCallback callback) {
    callback_ = callback;
  }

  const AriaWifiBeaconConfigRecord& getConfigRecord() const {
    return configRecord_;
  }

  const AriaWifiBeaconDataRecord& getDataRecord() const {
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
  WifiBeaconCallback callback_ = [](const vrs::CurrentRecord&, vrs::DataLayout&, bool) {
    return true;
  };

  AriaWifiBeaconConfigRecord configRecord_;
  AriaWifiBeaconDataRecord dataRecord_;

  double nextTimestampSec_ = 0;
  bool verbose_ = false;
};

} // namespace dataprovider
} // namespace datatools
} // namespace ark
