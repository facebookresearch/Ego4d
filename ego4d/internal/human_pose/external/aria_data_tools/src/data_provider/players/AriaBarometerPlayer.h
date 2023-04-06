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

#include <data_layout/aria/BarometerMetadata.h>
#include <vrs/RecordFormatStreamPlayer.h>

namespace ark {
namespace datatools {
namespace dataprovider {

using BarometerCallback =
    std::function<bool(const vrs::CurrentRecord& r, vrs::DataLayout& dataLayout, bool verbose)>;

struct AriaBarometerConfigRecord {
  uint32_t streamId;
  std::string sensorModelName;
  double sampleRate;
};

struct AriaBarometerDataRecord {
  int64_t captureTimestampNs;
  double temperature; // in degrees Celsius
  double pressure; // in Pascal
  double altitude;
};

class AriaBarometerPlayer : public vrs::RecordFormatStreamPlayer {
 public:
  explicit AriaBarometerPlayer(vrs::StreamId streamId) : streamId_(streamId) {}
  AriaBarometerPlayer(const AriaBarometerPlayer&) = delete;
  AriaBarometerPlayer& operator=(const AriaBarometerPlayer&) = delete;
  AriaBarometerPlayer(AriaBarometerPlayer&&) = default;

  void setCallback(BarometerCallback callback) {
    callback_ = callback;
  }

  const AriaBarometerConfigRecord& getConfigRecord() const {
    return configRecord_;
  }

  const AriaBarometerDataRecord& getDataRecord() const {
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
  BarometerCallback callback_ = [](const vrs::CurrentRecord&, vrs::DataLayout&, bool) {
    return true;
  };

  AriaBarometerConfigRecord configRecord_;
  AriaBarometerDataRecord dataRecord_;

  double nextTimestampSec_ = 0;
  bool verbose_ = false;
};

} // namespace dataprovider
} // namespace datatools
} // namespace ark
