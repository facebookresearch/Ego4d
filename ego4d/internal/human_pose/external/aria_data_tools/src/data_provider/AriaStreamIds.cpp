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

#include <AriaStreamIds.h>

namespace ark::datatools::dataprovider {
const vrs::StreamId kEyeCameraStreamId =
    vrs::StreamId(vrs::RecordableTypeId::EyeCameraRecordableClass, 1);
const vrs::StreamId kRgbCameraStreamId =
    vrs::StreamId(vrs::RecordableTypeId::RgbCameraRecordableClass, 1);
const vrs::StreamId kSlamLeftCameraStreamId =
    vrs::StreamId(vrs::RecordableTypeId::SlamCameraData, 1);
const vrs::StreamId kSlamRightCameraStreamId =
    vrs::StreamId(vrs::RecordableTypeId::SlamCameraData, 2);
const vrs::StreamId kImuRightStreamId = vrs::StreamId(vrs::RecordableTypeId::SlamImuData, 1);
const vrs::StreamId kImuLeftStreamId = vrs::StreamId(vrs::RecordableTypeId::SlamImuData, 2);
const vrs::StreamId kMagnetometerStreamId =
    vrs::StreamId(vrs::RecordableTypeId::SlamMagnetometerData, 1);
const vrs::StreamId kBarometerStreamId =
    vrs::StreamId(vrs::RecordableTypeId::BarometerRecordableClass, 1);
const vrs::StreamId kAudioStreamId =
    vrs::StreamId(vrs::RecordableTypeId::StereoAudioRecordableClass, 1);
const vrs::StreamId kWifiStreamId =
    vrs::StreamId(vrs::RecordableTypeId::WifiBeaconRecordableClass, 1);
const vrs::StreamId kBluetoothStreamId =
    vrs::StreamId(vrs::RecordableTypeId::BluetoothBeaconRecordableClass, 1);
const vrs::StreamId kGpsStreamId = vrs::StreamId(vrs::RecordableTypeId::GpsRecordableClass, 1);
const vrs::StreamId kTimeSyncStreamId =
    vrs::StreamId(vrs::RecordableTypeId::TimeRecordableClass, 1);
} // namespace ark::datatools::dataprovider
