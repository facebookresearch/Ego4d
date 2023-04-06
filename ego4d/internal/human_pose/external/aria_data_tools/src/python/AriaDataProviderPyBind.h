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

#include <data_provider/AriaDataProvider.h>
#include <data_provider/AriaVrsDataProvider.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace ark {
namespace datatools {
namespace dataprovider {

namespace py = pybind11;

void exportDataProvider(py::module& m) {
  m.doc() = "A pybind11 binding for Aria Data Tools data provider.";
  py::class_<vrs::IndexRecord::RecordInfo>(m, "RecordInfo")
      .def_readwrite("timestamp", &vrs::IndexRecord::RecordInfo::timestamp);

  // AriaDataProvider bindings
  py::class_<AriaDataProvider>(m, "AriaDataProvider")
      .def("getDeviceModel", &AriaDataProvider::getDeviceModel, py::return_value_policy::reference)
      .def("streamExistsInSource", &AriaDataProvider::streamExistsInSource, py::arg("streamId"));

  // AriaVrsDataProvider bindings
  py::class_<AriaVrsDataProvider, AriaDataProvider>(m, "AriaVrsDataProvider")
      .def(py::init<>())
      .def("openFile", &AriaVrsDataProvider::openFile, py::arg("vrsFilePath"))
      .def("getStreamsInFile", &AriaVrsDataProvider::getStreamsInFile)
      .def("readAllRecords", &AriaVrsDataProvider::readAllRecords)
      .def(
          "readFirstConfigurationRecord",
          &AriaVrsDataProvider::readFirstConfigurationRecord,
          py::arg("streamId"))
      .def(
          "getFirstTimestampSec",
          py::overload_cast<const vrs::StreamId&, const vrs::Record::Type&>(
              &AriaVrsDataProvider::getFirstTimestampSec),
          py::arg("streamId"),
          py::arg("type"))
      .def(
          "getRecordByTime",
          &AriaVrsDataProvider::getRecordByTime,
          py::return_value_policy::reference,
          py::arg("streamId"),
          py::arg("type"),
          py::arg("timestampSec"))
      .def(
          "readRecordsByTime",
          &AriaVrsDataProvider::readRecordsByTime,
          py::arg("type"),
          py::arg("timestampSec"))
      .def(
          "readRecordByTime",
          &AriaVrsDataProvider::readRecordByTime,
          py::arg("streamId"),
          py::arg("type"),
          py::arg("timestampSec"))
      .def("readRecord", &AriaVrsDataProvider::readRecord, py::arg("record"))
      .def(
          "getLastRecord",
          &AriaVrsDataProvider::getLastRecord,
          py::return_value_policy::reference,
          py::arg("streamId"),
          py::arg("type"))

      .def(
          "getFirstDataRecordTimestampSec",
          &AriaVrsDataProvider::getFirstDataRecordTimestampSec,
          py::arg("streamId"))
      .def(
          "getDataRecordByTime",
          &AriaVrsDataProvider::getDataRecordByTime,
          py::return_value_policy::reference,
          py::arg("streamId"),
          py::arg("timestampSec"))
      .def(
          "readDataRecordsByTime",
          &AriaVrsDataProvider::readDataRecordsByTime,
          py::arg("timestampSec"))
      .def(
          "readDataRecordByTime",
          &AriaVrsDataProvider::readDataRecordByTime,
          py::arg("streamId"),
          py::arg("timestampSec"))
      .def(
          "getLastDataRecord",
          &AriaVrsDataProvider::getLastDataRecord,
          py::return_value_policy::reference,
          py::arg("streamId"))

      .def("setSlamLeftCameraPlayer", &AriaVrsDataProvider::setSlamLeftCameraPlayer)
      .def("setSlamRightCameraPlayer", &AriaVrsDataProvider::setSlamRightCameraPlayer)
      .def("setRgbCameraPlayer", &AriaVrsDataProvider::setRgbCameraPlayer)
      .def("setEyeCameraPlayer", &AriaVrsDataProvider::setEyeCameraPlayer)
      .def("setImuRightPlayer", &AriaVrsDataProvider::setImuRightPlayer)
      .def("setImuLeftPlayer", &AriaVrsDataProvider::setImuLeftPlayer)
      .def("setMagnetometerPlayer", &AriaVrsDataProvider::setMagnetometerPlayer)
      .def("setWifiBeaconPlayer", &AriaVrsDataProvider::setWifiBeaconPlayer)
      .def("setBluetoothBeaconPlayer", &AriaVrsDataProvider::setBluetoothBeaconPlayer)
      .def("setAudioPlayer", &AriaVrsDataProvider::setAudioPlayer)
      .def("setGpsPlayer", &AriaVrsDataProvider::setGpsPlayer)
      .def("setBarometerPlayer", &AriaVrsDataProvider::setBarometerPlayer)
      .def("setTimeSyncPlayer", &AriaVrsDataProvider::setTimeSyncPlayer)
      .def("setStreamPlayer", &AriaVrsDataProvider::setStreamPlayer, py::arg("streamId"))

      .def(
          "getSlamLeftCameraPlayer",
          &AriaVrsDataProvider::getSlamLeftCameraPlayer,
          py::return_value_policy::reference)
      .def(
          "getSlamRightCameraPlayer",
          &AriaVrsDataProvider::getSlamRightCameraPlayer,
          py::return_value_policy::reference)
      .def(
          "getRgbCameraPlayer",
          &AriaVrsDataProvider::getRgbCameraPlayer,
          py::return_value_policy::reference)
      .def(
          "getEyeCameraPlayer",
          &AriaVrsDataProvider::getEyeCameraPlayer,
          py::return_value_policy::reference)
      .def(
          "getImuRightPlayer",
          &AriaVrsDataProvider::getImuRightPlayer,
          py::return_value_policy::reference)
      .def(
          "getImuLeftPlayer",
          &AriaVrsDataProvider::getImuLeftPlayer,
          py::return_value_policy::reference)
      .def(
          "getMagnetometerPlayer",
          &AriaVrsDataProvider::getMagnetometerPlayer,
          py::return_value_policy::reference)
      .def(
          "getWifiBeaconPlayer",
          &AriaVrsDataProvider::getWifiBeaconPlayer,
          py::return_value_policy::reference)
      .def(
          "getAudioPlayer",
          &AriaVrsDataProvider::getAudioPlayer,
          py::return_value_policy::reference)
      .def("getGpsPlayer", &AriaVrsDataProvider::getGpsPlayer, py::return_value_policy::reference)
      .def(
          "getBarometerPlayer",
          &AriaVrsDataProvider::getBarometerPlayer,
          py::return_value_policy::reference)
      .def(
          "getTimeSyncPlayer",
          &AriaVrsDataProvider::getTimeSyncPlayer,
          py::return_value_policy::reference)

      .def("setVerbose", &AriaVrsDataProvider::setVerbose, py::arg("verbose"))

      // AriaDataProvider override functions
      .def("open", &AriaVrsDataProvider::open, py::arg("vrsPath"))
      .def("setStreamPlayer", &AriaVrsDataProvider::setStreamPlayer, py::arg("streamId"))
      .def(
          "tryFetchNextData",
          &AriaVrsDataProvider::tryFetchNextData,
          py::arg("streamId"),
          py::arg("currentTimestampSec"))
      .def(
          "getImageBuffer",
          &AriaVrsDataProvider::getImageBuffer,
          py::return_value_policy::reference,
          py::arg("streamId"))
      .def("getImageWidth", &AriaVrsDataProvider::getImageWidth, py::arg("streamId"))
      .def("getImageHeight", &AriaVrsDataProvider::getImageHeight, py::arg("streamId"))
      .def("getFastestNominalRateHz", &AriaVrsDataProvider::getFastestNominalRateHz)
      .def("getFirstTimestampSec", py::overload_cast<>(&AriaVrsDataProvider::getFirstTimestampSec))
      .def("atLastRecords", &AriaVrsDataProvider::atLastRecords)
      .def("loadDeviceModel", &AriaVrsDataProvider::loadDeviceModel)
      .def(
          "getDeviceModel",
          &AriaVrsDataProvider::getDeviceModel,
          py::return_value_policy::reference)
      .def("streamExistsInSource", &AriaVrsDataProvider::streamExistsInSource, py::arg("streamId"));
}

} // namespace dataprovider
} // namespace datatools
} // namespace ark
