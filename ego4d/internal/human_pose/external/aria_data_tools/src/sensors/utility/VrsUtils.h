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

#include <vrs/RecordFileReader.h>

#include <string>

namespace ark::datatools::sensors {

// Get calibration string either from VRS or JSON file on disk
std::string getCalibStrFromFile(const std::string& filePath);

// Get calibration string from VRS RecordFileReader
std::string getCalibrationFromVrsFile(const vrs::RecordFileReader& reader);

} // namespace ark::datatools::sensors
