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

#include <players/AriaAudioPlayer.h>
#include <players/AriaBarometerPlayer.h>
#include <players/AriaBluetoothBeaconPlayer.h>
#include <players/AriaGpsPlayer.h>
#include <players/AriaImageSensorPlayer.h>
#include <players/AriaMotionSensorPlayer.h>
#include <players/AriaTimeSyncPlayer.h>
#include <players/AriaWifiBeaconPlayer.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace ark {
namespace datatools {
namespace dataprovider {

namespace py = pybind11;

void exportPlayers(py::module& m) {
  py::class_<vrs::StreamId>(m, "StreamId")
      .def(py::init([](uint16_t recordableTypeId, uint16_t instanceId) {
        return vrs::StreamId(static_cast<vrs::RecordableTypeId>(recordableTypeId), instanceId);
      }));
  py::class_<vrs::utils::PixelFrame, std::shared_ptr<vrs::utils::PixelFrame>>(m, "PixelFrame")
      .def("getBuffer", &vrs::utils::PixelFrame::getBuffer);

  py::class_<AriaImageSensorPlayer>(m, "AriaImageSensorPlayer")
      .def("getData", &AriaImageSensorPlayer::getData)
      .def("getConfigRecord", &AriaImageSensorPlayer::getConfigRecord)
      .def("getDataRecord", &AriaImageSensorPlayer::getDataRecord)
      .def("getStreamId", &AriaImageSensorPlayer::getStreamId)
      .def("getNextTimestampSec", &AriaImageSensorPlayer::getNextTimestampSec);
  py::class_<AriaImageData>(m, "AriaImageData")
      .def(py::init<>())
      .def_readwrite("pixelFrame", &AriaImageData::pixelFrame, py::return_value_policy::reference);
  py::class_<AriaImageConfigRecord>(m, "AriaImageConfigRecord")
      .def(py::init<>())
      .def_readwrite("deviceType", &AriaImageConfigRecord::deviceType)
      .def_readwrite("deviceVersion", &AriaImageConfigRecord::deviceVersion)
      .def_readwrite("deviceSerial", &AriaImageConfigRecord::deviceSerial)
      .def_readwrite("cameraId", &AriaImageConfigRecord::cameraId)
      .def_readwrite("sensorModel", &AriaImageConfigRecord::sensorModel)
      .def_readwrite("sensorSerial", &AriaImageConfigRecord::sensorSerial)
      .def_readwrite("nominalRateHz", &AriaImageConfigRecord::nominalRateHz)
      .def_readwrite("imageWidth", &AriaImageConfigRecord::imageWidth)
      .def_readwrite("imageHeight", &AriaImageConfigRecord::imageHeight)
      .def_readwrite("imageStride", &AriaImageConfigRecord::imageStride)
      .def_readwrite("pixelFormat", &AriaImageConfigRecord::pixelFormat)
      .def_readwrite("exposureDurationMin", &AriaImageConfigRecord::exposureDurationMin)
      .def_readwrite("exposureDurationMax", &AriaImageConfigRecord::exposureDurationMax)
      .def_readwrite("gainMin", &AriaImageConfigRecord::gainMin)
      .def_readwrite("gainMax", &AriaImageConfigRecord::gainMax)
      .def_readwrite("gammaFactor", &AriaImageConfigRecord::gammaFactor)
      .def_readwrite("factoryCalibration", &AriaImageConfigRecord::factoryCalibration)
      .def_readwrite("onlineCalibration", &AriaImageConfigRecord::onlineCalibration)
      .def_readwrite("description", &AriaImageConfigRecord::description);
  py::class_<AriaImageDataRecord>(m, "AriaImageDataRecord")
      .def(py::init<>())
      .def_readwrite("groupId", &AriaImageDataRecord::groupId)
      .def_readwrite("groupMask", &AriaImageDataRecord::groupMask)
      .def_readwrite("frameNumber", &AriaImageDataRecord::frameNumber)
      .def_readwrite("exposureDuration", &AriaImageDataRecord::exposureDuration)
      .def_readwrite("gain", &AriaImageDataRecord::gain)
      .def_readwrite("captureTimestampNs", &AriaImageDataRecord::captureTimestampNs)
      .def_readwrite("arrivalTimestampNs", &AriaImageDataRecord::arrivalTimestampNs)
      .def_readwrite("temperature", &AriaImageDataRecord::temperature);

  py::class_<AriaMotionSensorPlayer>(m, "AriaMotionSensorPlayer")
      .def("getConfigRecord", &AriaMotionSensorPlayer::getConfigRecord)
      .def("getDataRecord", &AriaMotionSensorPlayer::getDataRecord)
      .def("getStreamId", &AriaMotionSensorPlayer::getStreamId)
      .def("getNextTimestampSec", &AriaMotionSensorPlayer::getNextTimestampSec);
  py::class_<AriaMotionConfigRecord>(m, "AriaMotionConfigRecord")
      .def(py::init<>())
      .def_readwrite("streamIndex", &AriaMotionConfigRecord::streamIndex)
      .def_readwrite("deviceType", &AriaMotionConfigRecord::deviceType)
      .def_readwrite("deviceSerial", &AriaMotionConfigRecord::deviceSerial)
      .def_readwrite("deviceId", &AriaMotionConfigRecord::deviceId)
      .def_readwrite("sensorModel", &AriaMotionConfigRecord::sensorModel)
      .def_readwrite("nominalRateHz", &AriaMotionConfigRecord::nominalRateHz)
      .def_readwrite("hasAccelerometer", &AriaMotionConfigRecord::hasAccelerometer)
      .def_readwrite("hasGyroscope", &AriaMotionConfigRecord::hasGyroscope)
      .def_readwrite("hasMagnetometer", &AriaMotionConfigRecord::hasMagnetometer)
      .def_readwrite("factoryCalibration", &AriaMotionConfigRecord::factoryCalibration)
      .def_readwrite("onlineCalibration", &AriaMotionConfigRecord::onlineCalibration)
      .def_readwrite("description", &AriaMotionConfigRecord::description);
  py::class_<AriaMotionDataRecord>(m, "AriaMotionDataRecord")
      .def(py::init<>())
      .def_readwrite("accelValid", &AriaMotionDataRecord::accelValid)
      .def_readwrite("gyroValid", &AriaMotionDataRecord::gyroValid)
      .def_readwrite("magValid", &AriaMotionDataRecord::magValid)
      .def_readwrite("temperature", &AriaMotionDataRecord::temperature)
      .def_readwrite("captureTimestampNs", &AriaMotionDataRecord::captureTimestampNs)
      .def_readwrite("arrivalTimestampNs", &AriaMotionDataRecord::arrivalTimestampNs)
      .def_readwrite("accelMSec2", &AriaMotionDataRecord::accelMSec2)
      .def_readwrite("gyroRadSec", &AriaMotionDataRecord::gyroRadSec)
      .def_readwrite("magTesla", &AriaMotionDataRecord::magTesla);

  py::class_<AriaWifiBeaconPlayer>(m, "AriaWifiBeaconPlayer")
      .def("getConfigRecord", &AriaWifiBeaconPlayer::getConfigRecord)
      .def("getDataRecord", &AriaWifiBeaconPlayer::getDataRecord)
      .def("getStreamId", &AriaWifiBeaconPlayer::getStreamId)
      .def("getNextTimestampSec", &AriaWifiBeaconPlayer::getNextTimestampSec);
  py::class_<AriaWifiBeaconConfigRecord>(m, "AriaWifiBeaconConfigRecord")
      .def(py::init<>())
      .def_readwrite("streamId", &AriaWifiBeaconConfigRecord::streamId);
  py::class_<AriaWifiBeaconDataRecord>(m, "AriaWifiBeaconDataRecord")
      .def(py::init<>())
      .def_readwrite("systemTimestampNs", &AriaWifiBeaconDataRecord::systemTimestampNs)
      .def_readwrite("boardTimestampNs", &AriaWifiBeaconDataRecord::boardTimestampNs)
      .def_readwrite(
          "boardScanRequestStartTimestampNs",
          &AriaWifiBeaconDataRecord::boardScanRequestStartTimestampNs)
      .def_readwrite(
          "boardScanRequestCompleteTimestampNs",
          &AriaWifiBeaconDataRecord::boardScanRequestCompleteTimestampNs)
      .def_readwrite("ssid", &AriaWifiBeaconDataRecord::ssid)
      .def_readwrite("bssidMac", &AriaWifiBeaconDataRecord::bssidMac)
      .def_readwrite("rssi", &AriaWifiBeaconDataRecord::rssi)
      .def_readwrite("gyrfreqMhzoRadSec", &AriaWifiBeaconDataRecord::freqMhz)
      .def_readwrite("rssiPerAntenna", &AriaWifiBeaconDataRecord::rssiPerAntenna);

  py::class_<AriaAudioPlayer>(m, "AriaAudioPlayer")
      .def("getData", &AriaAudioPlayer::getData)
      .def("getConfigRecord", &AriaAudioPlayer::getConfigRecord)
      .def("getDataRecord", &AriaAudioPlayer::getDataRecord)
      .def("getStreamId", &AriaAudioPlayer::getStreamId)
      .def("getNextTimestampSec", &AriaAudioPlayer::getNextTimestampSec);
  py::class_<AriaAudioData>(m, "AriaAudioData")
      .def(py::init<>())
      .def_readwrite("data", &AriaAudioData::data);
  py::class_<AriaAudioConfigRecord>(m, "AriaAudioConfigRecord")
      .def(py::init<>())
      .def_readwrite("streamId", &AriaAudioConfigRecord::streamId)
      .def_readwrite("numChannels", &AriaAudioConfigRecord::numChannels)
      .def_readwrite("sampleRate", &AriaAudioConfigRecord::sampleRate)
      .def_readwrite("sampleFormat", &AriaAudioConfigRecord::sampleFormat);
  py::class_<AriaAudioDataRecord>(m, "AriaAudioDataRecord")
      .def(py::init<>())
      .def_readwrite("captureTimestampsNs", &AriaAudioDataRecord::captureTimestampsNs)
      .def_readwrite("audioMuted", &AriaAudioDataRecord::audioMuted);

  py::class_<AriaBluetoothBeaconPlayer>(m, "AriaBluetoothBeaconPlayer")
      .def("getConfigRecord", &AriaBluetoothBeaconPlayer::getConfigRecord)
      .def("getDataRecord", &AriaBluetoothBeaconPlayer::getDataRecord)
      .def("getStreamId", &AriaBluetoothBeaconPlayer::getStreamId)
      .def("getNextTimestampSec", &AriaBluetoothBeaconPlayer::getNextTimestampSec);
  py::class_<AriaBluetoothBeaconConfigRecord>(m, "AriaBluetoothBeaconConfigRecord")
      .def(py::init<>())
      .def_readwrite("streamId", &AriaBluetoothBeaconConfigRecord::streamId)
      .def_readwrite("sampleRateHz", &AriaBluetoothBeaconConfigRecord::sampleRateHz);
  py::class_<AriaBluetoothBeaconDataRecord>(m, "AriaBluetoothBeaconDataRecord")
      .def(py::init<>())
      .def_readwrite("systemTimestampNs", &AriaBluetoothBeaconDataRecord::systemTimestampNs)
      .def_readwrite("boardTimestampNs", &AriaBluetoothBeaconDataRecord::boardTimestampNs)
      .def_readwrite(
          "boardScanRequestStartTimestampNs",
          &AriaBluetoothBeaconDataRecord::boardScanRequestStartTimestampNs)
      .def_readwrite(
          "boardScanRequestCompleteTimestampNs",
          &AriaBluetoothBeaconDataRecord::boardScanRequestCompleteTimestampNs)
      .def_readwrite("uniqueId", &AriaBluetoothBeaconDataRecord::uniqueId)
      .def_readwrite("txPower", &AriaBluetoothBeaconDataRecord::txPower)
      .def_readwrite("rssi", &AriaBluetoothBeaconDataRecord::rssi)
      .def_readwrite("freqMhz", &AriaBluetoothBeaconDataRecord::freqMhz);

  py::class_<AriaGpsPlayer>(m, "AriaGpsPlayer")
      .def("getConfigRecord", &AriaGpsPlayer::getConfigRecord)
      .def("getDataRecord", &AriaGpsPlayer::getDataRecord)
      .def("getStreamId", &AriaGpsPlayer::getStreamId)
      .def("getNextTimestampSec", &AriaGpsPlayer::getNextTimestampSec);
  py::class_<AriaGpsConfigRecord>(m, "AriaGpsConfigRecord")
      .def(py::init<>())
      .def_readwrite("streamId", &AriaGpsConfigRecord::streamId)
      .def_readwrite("sampleRateHz", &AriaGpsConfigRecord::sampleRateHz);
  py::class_<AriaGpsDataRecord>(m, "AriaGpsDataRecord")
      .def(py::init<>())
      .def_readwrite("captureTimestampNs", &AriaGpsDataRecord::captureTimestampNs)
      .def_readwrite("utcTimeMs", &AriaGpsDataRecord::utcTimeMs)
      .def_readwrite("provider", &AriaGpsDataRecord::provider)
      .def_readwrite("latitude", &AriaGpsDataRecord::latitude)
      .def_readwrite("longitude", &AriaGpsDataRecord::longitude)
      .def_readwrite("altitude", &AriaGpsDataRecord::altitude)
      .def_readwrite("accuracy", &AriaGpsDataRecord::accuracy)
      .def_readwrite("speed", &AriaGpsDataRecord::speed)
      .def_readwrite("rawData", &AriaGpsDataRecord::rawData);

  py::class_<AriaBarometerPlayer>(m, "AriaBarometerPlayer")
      .def("getConfigRecord", &AriaBarometerPlayer::getConfigRecord)
      .def("getDataRecord", &AriaBarometerPlayer::getDataRecord)
      .def("getStreamId", &AriaBarometerPlayer::getStreamId)
      .def("getNextTimestampSec", &AriaBarometerPlayer::getNextTimestampSec);
  py::class_<AriaBarometerConfigRecord>(m, "AriaBarometerConfigRecord")
      .def(py::init<>())
      .def_readwrite("streamId", &AriaBarometerConfigRecord::streamId)
      .def_readwrite("sensorModelName", &AriaBarometerConfigRecord::sensorModelName)
      .def_readwrite("sampleRate", &AriaBarometerConfigRecord::sampleRate);
  py::class_<AriaBarometerDataRecord>(m, "AriaBarometerDataRecord")
      .def(py::init<>())
      .def_readwrite("captureTimestampNs", &AriaBarometerDataRecord::captureTimestampNs)
      .def_readwrite("temperature", &AriaBarometerDataRecord::temperature)
      .def_readwrite("pressure", &AriaBarometerDataRecord::pressure)
      .def_readwrite("altitude", &AriaBarometerDataRecord::altitude);

  py::class_<AriaTimeSyncPlayer>(m, "AriaTimeSyncPlayer")
      .def("getConfigRecord", &AriaTimeSyncPlayer::getConfigRecord)
      .def("getDataRecord", &AriaTimeSyncPlayer::getDataRecord)
      .def("getStreamId", &AriaTimeSyncPlayer::getStreamId)
      .def("getNextTimestampSec", &AriaTimeSyncPlayer::getNextTimestampSec);
  py::class_<AriaTimeSyncConfigRecord>(m, "AriaTimeSyncConfigRecord")
      .def(py::init<>())
      .def_readwrite("streamId", &AriaTimeSyncConfigRecord::streamId)
      .def_readwrite("sampleRateHz", &AriaTimeSyncConfigRecord::sampleRateHz);
  py::class_<AriaTimeSyncDataRecord>(m, "AriaTimeSyncDataRecord")
      .def(py::init<>())
      .def_readwrite("monotonicTimestampNs", &AriaTimeSyncDataRecord::monotonicTimestampNs)
      .def_readwrite("realTimestampNs", &AriaTimeSyncDataRecord::realTimestampNs);
}

} // namespace dataprovider
} // namespace datatools
} // namespace ark
