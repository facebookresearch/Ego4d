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

#include <utility/VrsUtils.h>

#include <filesystem>
#include <fstream>
#include <sstream>

namespace ark::datatools::sensors {

namespace fs = std::filesystem;

std::string getCalibStrFromFile(const std::string& filePath) {
  auto ext = fs::path(filePath).extension();
  if (ext == ".json") {
    std::ifstream fin(filePath);
    if (!fin.is_open()) {
      std::cerr << "Unable to find file: " << filePath << std::endl;
    }
    std::ostringstream sstr;
    sstr << fin.rdbuf();
    fin.close();
    return sstr.str();
  } else if (ext == ".vrs") {
    vrs::RecordFileReader reader;
    reader.openFile(filePath);
    return sensors::getCalibrationFromVrsFile(reader);
  } else {
    std::cerr << "Unsupported file type: " << ext << std::endl;
  }
  // return {} - no calibration string was found
  return {};
}

std::string getCalibrationFromVrsFile(const vrs::RecordFileReader& reader) {
  return reader.getTag("calib_json");
}

} // namespace ark::datatools::sensors
