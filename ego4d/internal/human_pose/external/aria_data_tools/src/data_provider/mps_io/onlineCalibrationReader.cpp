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

#include <fstream>
#include <iostream>

#include "onlineCalibrationReader.h"

namespace ark::datatools {

TemporalDeviceModels readOnlineCalibration(const std::filesystem::path& filepath) {
  std::ifstream infile(filepath);
  if (infile) {
    std::string jsonCalibrationString = "";
    TemporalDeviceModels temporalDeviceModels;
    while (std::getline(infile, jsonCalibrationString)) {
      sensors::DeviceModel deviceModel = sensors::DeviceModel::fromJson(jsonCalibrationString);
      // Read timestamps
      fb_rapidjson::Document doc;
      doc.Parse(jsonCalibrationString.c_str());
      const auto timestamp_us = std::stoul(doc["tracking_timestamp_us"].GetString());
      // FYI you can also use "utc_timestamp_ns" for hashing if you need to.
      temporalDeviceModels[std::chrono::microseconds(timestamp_us)] = std::move(deviceModel);
    }
    std::cout << "Loaded #deviceModels records: " << temporalDeviceModels.size() << std::endl;
    return temporalDeviceModels;
  } else {
    std::cerr << "[readOnlineCalibration] Can't open the provided file path." << std::endl;
  }
  return {};
}

} // namespace ark::datatools
