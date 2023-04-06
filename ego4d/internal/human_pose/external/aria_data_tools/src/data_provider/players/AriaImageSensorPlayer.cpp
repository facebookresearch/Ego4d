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

#include <cmath>

#include "AriaImageSensorPlayer.h"

#include <vrs/MultiRecordFileReader.h>
#include <vrs/RecordFormat.h>

namespace ark {
namespace datatools {
namespace dataprovider {

bool AriaImageSensorPlayer::onDataLayoutRead(
    const vrs::CurrentRecord& r,
    size_t blockIndex,
    vrs::DataLayout& dl) {
  if (r.recordType == vrs::Record::Type::CONFIGURATION) {
    auto& config = getExpectedLayout<aria::ImageSensorConfigRecordMetadata>(dl, blockIndex);
    configRecord_.deviceType = config.deviceType.get();
    configRecord_.deviceVersion = config.deviceVersion.get();
    configRecord_.deviceSerial = config.deviceSerial.get();
    configRecord_.cameraId = config.cameraId.get();
    configRecord_.sensorModel = config.sensorModel.get();
    configRecord_.sensorSerial = config.sensorSerial.get();
    configRecord_.nominalRateHz = config.nominalRateHz.get();
    configRecord_.imageWidth = config.imageWidth.get();
    configRecord_.imageHeight = config.imageHeight.get();
    configRecord_.imageStride = config.imageStride.get();
    configRecord_.pixelFormat = config.pixelFormat.get();
    configRecord_.exposureDurationMin = config.exposureDurationMin.get();
    configRecord_.exposureDurationMax = config.exposureDurationMax.get();
    configRecord_.gainMin = config.gainMin.get();
    configRecord_.gainMax = config.gainMax.get();
    configRecord_.gammaFactor = config.gammaFactor.get();
    configRecord_.factoryCalibration = config.factoryCalibration.get();
    configRecord_.onlineCalibration = config.onlineCalibration.get();
    configRecord_.description = config.description.get();
  } else if (r.recordType == vrs::Record::Type::DATA) {
    auto& data = getExpectedLayout<aria::ImageSensorDataRecordMetadata>(dl, blockIndex);
    dataRecord_.groupId = data.groupId.get();
    dataRecord_.groupMask = data.groupMask.get();
    dataRecord_.frameNumber = data.frameNumber.get();
    dataRecord_.exposureDuration = data.exposureDuration.get();
    dataRecord_.gain = data.gain.get();
    dataRecord_.captureTimestampNs = data.captureTimestampNs.get();
    dataRecord_.arrivalTimestampNs = data.arrivalTimestampNs.get();
    nextTimestampSec_ = std::nextafter(r.timestamp, std::numeric_limits<double>::max());
  }
  return true;
}
bool AriaImageSensorPlayer::onImageRead(
    const vrs::CurrentRecord& r,
    size_t /*idx*/,
    const vrs::ContentBlock& cb) {
  // the image data was not read yet: allocate your own buffer & read!
  auto& imageSpec = cb.image();
  size_t blockSize = cb.getBlockSize();
  // Synchronously read the image data, which is jpg compressed with Aria
  if (imageSpec.getImageFormat() == vrs::ImageFormat::JPG) {
    vrs::utils::PixelFrame::readJpegFrame(data_.pixelFrame, r.reader, cb.getBlockSize());
    callback_(r, data_.pixelFrame->getBuffer(), verbose_);
  } else if (imageSpec.getImageFormat() == vrs::ImageFormat::RAW) {
    vrs::utils::PixelFrame::readRawFrame(data_.pixelFrame, r.reader, imageSpec);
    callback_(r, data_.pixelFrame->getBuffer(), verbose_);
  }
  if (verbose_) {
    fmt::print(
        "{:.3f} {} [{}]: {}, {} bytes.\n",
        r.timestamp,
        r.streamId.getName(),
        r.streamId.getNumericName(),
        imageSpec.asString(),
        blockSize);
  }
  return true; // read next blocks, if any
}

} // namespace dataprovider
} // namespace datatools
} // namespace ark
