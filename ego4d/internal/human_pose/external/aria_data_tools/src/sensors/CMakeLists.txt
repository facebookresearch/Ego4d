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

set(SENSORS_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

# Build sensors lib.
file(GLOB_RECURSE SENSORS_SRCS
    camera/*.h
    camera/*.cpp
    models/*.h
    models/*.cpp
    utility/*.cpp
    utility/*.cpp)
add_library(sensors STATIC ${SENSORS_SRCS})

target_include_directories(sensors
    PUBLIC
        ${SENSORS_SOURCE_DIR}
)
target_link_libraries(sensors
    PUBLIC
        Eigen3::Eigen
        vrslib
        Sophus::Sophus
        cereal::cereal
)

# Build tests.
if(BUILD_ARK_TESTS)
    add_subdirectory(tests)
endif(BUILD_ARK_TESTS)

# Build examples.
if (BUILD_ARK_EXAMPLES)
    add_subdirectory(examples)
endif(BUILD_ARK_EXAMPLES)
