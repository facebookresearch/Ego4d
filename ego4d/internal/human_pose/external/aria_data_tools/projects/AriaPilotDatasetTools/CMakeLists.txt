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

message("--- Compiling AriaPilotDatasetTools.")

cmake_minimum_required(VERSION 3.12)

project(AriaPilotDatasetTools
    VERSION 0.1
    DESCRIPTION "Data provider and utilities for Project Aria Pilot Dataset (CVPR2022)."
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Build in Release mode by default
if (NOT CMAKE_BUILD_TYPE AND NOT MSVC)
  set(CMAKE_BUILD_TYPE "Release")
endif (NOT CMAKE_BUILD_TYPE AND NOT MSVC)

# This folder is "Project" using aria data tools core libraries
add_subdirectory(../../src ariaDataToolsCore EXCLUDE_FROM_ALL)

add_library(pilotDatasetProvider_lib
    PilotDataset.cpp
    PilotDataset.h
    speech_to_text_datum.h)
target_link_libraries(pilotDatasetProvider_lib
    PUBLIC
        vrsdataprovider
        Sophus::Sophus
        )
target_include_directories(pilotDatasetProvider_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
        "../../src/third_party/"#fast-cpp-csv-parser
        )

find_package(Pangolin QUIET)
if (Pangolin_FOUND)
    set(BUILD_WITH_PANGOLIN ON)
else()
    set(BUILD_WITH_PANGOLIN OFF)
endif()

if (BUILD_WITH_PANGOLIN)
    message("--- Pangolin library has been found. Visualizer will be built")
    find_package(Threads REQUIRED)
    add_executable(pilotDataset_viewer
        ./viewer/main.cpp
        ./viewer/PilotDatasetViewer.cpp
        ./viewer/PilotDatasetViewer.h)
    target_link_libraries(pilotDataset_viewer
        PRIVATE
        pilotDatasetProvider_lib
        aria_viewer_lib
        ${Pangolin_LIBRARIES}
        Threads::Threads)
else()
    message("--- Pangolin (optional) library dependency has NOT been found. The dataset visualizer will NOT be built")
endif (BUILD_WITH_PANGOLIN)
