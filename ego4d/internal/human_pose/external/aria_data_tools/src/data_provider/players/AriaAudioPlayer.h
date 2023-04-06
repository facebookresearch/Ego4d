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

#include <data_layout/aria/AudioMetadata.h>
#include <vrs/RecordFormatStreamPlayer.h>

namespace ark {
namespace datatools {
namespace dataprovider {

using AudioCallback =
    std::function<bool(const vrs::CurrentRecord& r, std::vector<int32_t>& data, bool verbose)>;

struct AriaAudioData {
  std::vector<int32_t> data;
};

struct AriaAudioConfigRecord {
  uint32_t streamId;
  uint8_t numChannels;
  uint32_t sampleRate;
  uint8_t sampleFormat;
};

struct AriaAudioDataRecord {
  std::vector<int64_t> captureTimestampsNs;
  uint8_t audioMuted;
};

class AriaAudioPlayer : public vrs::RecordFormatStreamPlayer {
 public:
  explicit AriaAudioPlayer(vrs::StreamId streamId) : streamId_(streamId) {}
  AriaAudioPlayer(const AriaAudioPlayer&) = delete;
  AriaAudioPlayer& operator=(const AriaAudioPlayer&) = delete;
  AriaAudioPlayer(AriaAudioPlayer&&) = default;

  void setCallback(AudioCallback callback) {
    callback_ = callback;
  }

  const AriaAudioData& getData() const {
    return data_;
  }

  const AriaAudioConfigRecord& getConfigRecord() const {
    return configRecord_;
  }

  const AriaAudioDataRecord& getDataRecord() const {
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
  bool onAudioRead(const vrs::CurrentRecord& r, size_t blockIdx, const vrs::ContentBlock& cb)
      override;

  const vrs::StreamId streamId_;
  AudioCallback callback_ = [](const vrs::CurrentRecord&, std::vector<int32_t>&, bool) {
    return true;
  };
  AriaAudioData data_;
  AriaAudioConfigRecord configRecord_;
  AriaAudioDataRecord dataRecord_;

  double nextTimestampSec_ = 0;
  bool verbose_ = false;
};

} // namespace dataprovider
} // namespace datatools
} // namespace ark
