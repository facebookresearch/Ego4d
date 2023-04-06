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

add_library(dataprovider AriaDataProvider.h AriaStreamIds.h AriaStreamIds.cpp utils.h utils.cpp)
target_link_libraries(dataprovider
    PUBLIC
        players
        sensors
        vrsdataprovider
        Sophus::Sophus)
target_include_directories(dataprovider
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR} "../"
    PRIVATE
        "../third_party/"#fast-cpp-csv-parser"
        )

add_library(vrsdataprovider AriaVrsDataProvider.h AriaVrsDataProvider.cpp)
target_link_libraries(vrsdataprovider PUBLIC dataprovider players)
target_include_directories(vrsdataprovider PUBLIC dataprovider ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(data_layout)
add_subdirectory(examples)
add_subdirectory(mps_io)

if (BUILD_WITH_PANGOLIN)
    add_subdirectory(mps_visualization)
endif(BUILD_WITH_PANGOLIN)

add_subdirectory(players)
