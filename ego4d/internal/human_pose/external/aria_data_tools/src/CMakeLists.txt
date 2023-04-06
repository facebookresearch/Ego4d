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

cmake_minimum_required(VERSION 3.12)

project(ark
    VERSION 0.1
    DESCRIPTION "SDK and utilities for Project Aria datasets."
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Build in Release mode by default
if (NOT CMAKE_BUILD_TYPE AND NOT MSVC)
  set(CMAKE_BUILD_TYPE "Release")
endif (NOT CMAKE_BUILD_TYPE AND NOT MSVC)

# Handle shared 3rd party library (all as submodules)
option(UNIT_TESTS OFF)
option(BUILD_SAMPLES OFF)
option(BUILD_TOOLS OFF)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/third_party/vrs" EXCLUDE_FROM_ALL)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen" EXCLUDE_FROM_ALL)
option(SKIP_PERFORMANCE_COMPARISON OFF)
option(BUILD_SANDBOX OFF)
option(BUILD_DOC OFF)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/third_party/cereal")
option(BUILD_SOPHUS_TESTS OFF)
option(BUILD_SOPHUS_EXAMPLES OFF)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/third_party/Sophus")
add_definitions("-DSOPHUS_DISABLE_ENSURES")

# Project CMAKE options
option(BUILD_PYARK "Build python binding." OFF)
option(BUILD_ARK_TESTS "Build tests." ON)
option(BUILD_ARK_EXAMPLES "Build examples." ON)
option(BUILD_WITH_PANGOLIN "Build Pangolin related samples" OFF)

if(BUILD_ARK_TESTS)
    enable_testing()
endif(BUILD_ARK_TESTS)

if(BUILD_PYARK)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG master
    )
    FetchContent_MakeAvailable(pybind11)

    add_subdirectory(${pybind11_SOURCE_DIR})
    pybind11_add_module(pyark ${CMAKE_CURRENT_SOURCE_DIR}/python/bindings.cpp)
    target_link_libraries(pyark PUBLIC dataprovider sensors mps_io vrslib)
    if (BUILD_WITH_PANGOLIN)
        target_compile_definitions(pyark PUBLIC BUILD_WITH_PANGOLIN=1)
        target_link_libraries(pyark PUBLIC aria_viewer_lib)
    endif(BUILD_WITH_PANGOLIN)
endif(BUILD_PYARK)


add_subdirectory(data_provider)
add_subdirectory(sensors)

if (BUILD_ARK_EXAMPLES)
    add_subdirectory(experimental)
endif(BUILD_ARK_EXAMPLES)

if (BUILD_WITH_PANGOLIN)
    add_subdirectory(visualization)
endif(BUILD_WITH_PANGOLIN)
