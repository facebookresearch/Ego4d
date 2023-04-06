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

#include <data_layout/aria/PoseMetadata.h>
#include <vrs/RecordFormatStreamPlayer.h>

namespace ark {
namespace datatools {
namespace dataprovider {

using PoseCallback =
    std::function<bool(const vrs::CurrentRecord& r, vrs::DataLayout& dataLayout, bool verbose)>;

struct AriaPoseConfigRecord {
  uint32_t streamId;
};

struct AriaPoseDataRecord {
  int64_t captureTimestampNs;
  std::vector<float> T_World_ImuLeft_translation = {0, 0, 0};
  std::vector<float> T_World_ImuLeft_quaternion = {0, 0, 0, 1};
};

class AriaPosePlayer : public vrs::RecordFormatStreamPlayer {
 public:
  explicit AriaPosePlayer(vrs::StreamId streamId) : streamId_(streamId) {}
  AriaPosePlayer(const AriaPosePlayer&) = delete;
  AriaPosePlayer& operator=(const AriaPosePlayer&) = delete;
  AriaPosePlayer(AriaPosePlayer&&) = default;

  void setCallback(PoseCallback callback) {
    callback_ = callback;
  }

  const AriaPoseConfigRecord& getConfigRecord() const {
    return configRecord_;
  }

  const AriaPoseDataRecord& getDataRecord() const {
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
  PoseCallback callback_ = [](const vrs::CurrentRecord&, vrs::DataLayout&, bool) { return true; };

  AriaPoseConfigRecord configRecord_;
  AriaPoseDataRecord dataRecord_;

  double nextTimestampSec_ = 0;
  bool verbose_ = false;
};

} // namespace dataprovider
} // namespace datatools
} // namespace ark
