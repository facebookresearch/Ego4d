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

#include "AriaGpsPlayer.h"

namespace ark {
namespace datatools {
namespace dataprovider {

bool AriaGpsPlayer::onDataLayoutRead(
    const vrs::CurrentRecord& r,
    size_t blockIndex,
    vrs::DataLayout& dl) {
  if (r.recordType == vrs::Record::Type::CONFIGURATION) {
    auto& config = getExpectedLayout<aria::GpsConfigRecordMetadata>(dl, blockIndex);
    configRecord_.streamId = config.streamId.get();
    configRecord_.sampleRateHz = config.sampleRateHz.get();
  } else if (r.recordType == vrs::Record::Type::DATA) {
    auto& data = getExpectedLayout<aria::GpsDataRecordMetadata>(dl, blockIndex);
    dataRecord_.captureTimestampNs = data.captureTimestampNs.get();
    dataRecord_.utcTimeMs = data.utcTimeMs.get();
    dataRecord_.provider = data.provider.get();
    dataRecord_.latitude = data.latitude.get();
    dataRecord_.longitude = data.longitude.get();
    dataRecord_.altitude = data.altitude.get();
    dataRecord_.accuracy = data.accuracy.get();
    dataRecord_.speed = data.speed.get();
    data.rawData.get(dataRecord_.rawData);
    nextTimestampSec_ = std::nextafter(r.timestamp, std::numeric_limits<double>::max());
    callback_(r, data, verbose_);
  }
  return true;
}

} // namespace dataprovider
} // namespace datatools
} // namespace ark
