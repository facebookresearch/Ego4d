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

#include <chrono>
#include <filesystem>
#include <map>
#include "models/DeviceModel.h"

namespace ark::datatools {

// A time sorted list of DeviceModels data
using TemporalDeviceModels = std::map<std::chrono::microseconds, sensors::DeviceModel>;

// Read Online Calibration data from a file
TemporalDeviceModels readOnlineCalibration(const std::filesystem::path& filepath);

} // namespace ark::datatools
