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

#include <sophus/se3.hpp>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>

namespace ark {
namespace datatools {
namespace dataprovider {

// Table to map Aria numeric StreamID to calibration label string.
// Please check our documentation for more details.
const std::unordered_map<std::string, std::string> kDeviceNumericIdToLabel = {
    {"1201-1", "camera-slam-left"},
    {"1201-2", "camera-slam-right"},
    {"1202-1", "imu-right"},
    {"1202-2", "imu-left"},
    {"214-1", "camera-rgb"}};

void getDirContent(const std::string& dirPath, std::vector<std::string>& dirContent);
std::vector<std::string> strSplit(const std::string& s, const char delimiter);

std::string getTimeSyncPath(const std::string& vrsPath);

std::map<int64_t, int64_t> readTimeSyncCsv(const std::string& inputTimeSyncCsv);

} // namespace dataprovider
} // namespace datatools
} // namespace ark
