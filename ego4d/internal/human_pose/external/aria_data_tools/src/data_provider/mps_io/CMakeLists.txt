# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_library(mps_io
    trajectoryReader.h trajectoryReader.cpp
    eyeGazeReader.h eyeGazeReader.cpp
    onlineCalibrationReader.h onlineCalibrationReader.cpp)
target_link_libraries(mps_io PUBLIC Sophus::Sophus sensors)
target_include_directories(mps_io
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR} "../"
    PRIVATE
        "../../third_party/" #fast-cpp-csv-parser
)
