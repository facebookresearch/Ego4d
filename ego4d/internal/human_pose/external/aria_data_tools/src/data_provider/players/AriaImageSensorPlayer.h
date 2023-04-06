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

#include <data_layout/aria/ImageSensorMetadata.h>
#include <vrs/RecordFormatStreamPlayer.h>
#include <vrs/utils/PixelFrame.h>

namespace ark {
namespace datatools {
namespace dataprovider {

using ImageCallback =
    std::function<bool(const vrs::CurrentRecord& r, std::vector<uint8_t>& data, bool verbose)>;

struct AriaImageData {
  std::shared_ptr<vrs::utils::PixelFrame> pixelFrame;
};

struct AriaImageConfigRecord {
  std::string deviceType;
  std::string deviceVersion;
  std::string deviceSerial;
  uint32_t cameraId;
  std::string sensorModel;
  std::string sensorSerial;
  double nominalRateHz;
  uint32_t imageWidth;
  uint32_t imageHeight;
  uint32_t imageStride;
  uint32_t pixelFormat;
  double exposureDurationMin;
  double exposureDurationMax;
  double gainMin;
  double gainMax;
  double gammaFactor;
  std::string factoryCalibration;
  std::string onlineCalibration;
  std::string description;
};

struct AriaImageDataRecord {
  uint64_t groupId;
  uint64_t groupMask;
  uint64_t frameNumber;
  double exposureDuration;
  double gain;
  int64_t captureTimestampNs;
  int64_t arrivalTimestampNs;
  double temperature;
};

class AriaImageSensorPlayer : public vrs::RecordFormatStreamPlayer {
 public:
  explicit AriaImageSensorPlayer(vrs::StreamId streamId) : streamId_(streamId) {}
  AriaImageSensorPlayer(const AriaImageSensorPlayer&) = delete;
  AriaImageSensorPlayer& operator=(const AriaImageSensorPlayer&) = delete;
  AriaImageSensorPlayer(AriaImageSensorPlayer&&) = default;

  void setCallback(ImageCallback callback) {
    callback_ = callback;
  }

  const AriaImageData& getData() const {
    return data_;
  }

  const AriaImageConfigRecord& getConfigRecord() const {
    return configRecord_;
  }

  const AriaImageDataRecord& getDataRecord() const {
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
  bool onImageRead(const vrs::CurrentRecord& r, size_t /*idx*/, const vrs::ContentBlock& cb)
      override;

  const vrs::StreamId streamId_;
  ImageCallback callback_ = [](const vrs::CurrentRecord&, std::vector<uint8_t>&, bool) {
    return true;
  };
  AriaImageData data_;
  AriaImageConfigRecord configRecord_;
  AriaImageDataRecord dataRecord_;

  double nextTimestampSec_ = 0;
  bool verbose_ = false;
};

} // namespace dataprovider
} // namespace datatools
} // namespace ark
