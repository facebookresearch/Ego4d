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

#include <data_layout/aria/GpsMetadata.h>
#include <vrs/RecordFormatStreamPlayer.h>

namespace ark {
namespace datatools {
namespace dataprovider {

using GpsCallback =
    std::function<bool(const vrs::CurrentRecord& r, vrs::DataLayout& dataLayout, bool verbose)>;

struct AriaGpsConfigRecord {
  uint32_t streamId;
  double sampleRateHz;
};

struct AriaGpsDataRecord {
  int64_t captureTimestampNs;
  int64_t utcTimeMs;
  std::string provider;
  float latitude;
  float longitude;
  float altitude;
  float accuracy;
  float speed;
  std::vector<std::string> rawData;
};

class AriaGpsPlayer : public vrs::RecordFormatStreamPlayer {
 public:
  explicit AriaGpsPlayer(vrs::StreamId streamId) : streamId_(streamId) {}
  AriaGpsPlayer(const AriaGpsPlayer&) = delete;
  AriaGpsPlayer& operator=(const AriaGpsPlayer&) = delete;
  AriaGpsPlayer(AriaGpsPlayer&&) = default;

  void setCallback(GpsCallback callback) {
    callback_ = callback;
  }

  const AriaGpsConfigRecord& getConfigRecord() const {
    return configRecord_;
  }

  const AriaGpsDataRecord& getDataRecord() const {
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
  GpsCallback callback_ = [](const vrs::CurrentRecord&, vrs::DataLayout&, bool) { return true; };

  AriaGpsConfigRecord configRecord_;
  AriaGpsDataRecord dataRecord_;

  double nextTimestampSec_ = 0;
  bool verbose_ = false;
};

} // namespace dataprovider
} // namespace datatools
} // namespace ark
