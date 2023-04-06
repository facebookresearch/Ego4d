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

#include <chrono>
#include <filesystem>
#include <string>
#include <thread>
#include "AriaStreamIds.h"
#include "PilotDatasetViewer.h"
#include "utils.h"

using namespace ark::datatools;

namespace {
template <
    class result_t = std::chrono::milliseconds,
    class clock_t = std::chrono::steady_clock,
    class duration_t = std::chrono::milliseconds>
auto since(std::chrono::time_point<clock_t, duration_t> const& start) {
  return std::chrono::duration_cast<result_t>(clock_t::now() - start);
};

const std::vector<vrs::StreamId> kImageStreamIds = {
    dataprovider::kSlamLeftCameraStreamId,
    dataprovider::kSlamRightCameraStreamId,
    dataprovider::kRgbCameraStreamId,
    dataprovider::kEyeCameraStreamId};
} // namespace

int main(int argc, const char* argv[]) {
  if (argc < 2) {
    std::cerr << "VRS file path must be provided as the argument, exiting." << std::endl;
    return 0;
  }

  const std::string vrsPath = argv[1];
  const std::string vrsFolder = std::filesystem::path(vrsPath).remove_filename();
  // override default pose and eyetracking paths from commandline. If left empty
  // strings they will be automatically set according to the folder layout.
  std::string posePath = vrsFolder;
  std::string eyetrackingPath = vrsFolder;
  std::string speechToTextPath = vrsFolder;
  if (argc >= 3) {
    posePath = argv[2];
  }
  if (argc >= 4) {
    eyetrackingPath = argv[3];
  }
  if (argc >= 5) {
    speechToTextPath = argv[4];
  }
  // get and open data provider
  auto dataProvider = std::make_shared<dataprovider::PilotDatasetProvider>(
      posePath, eyetrackingPath, speechToTextPath);
  if (!dataProvider->open(vrsPath)) {
    std::cerr << "Failed to open '" << vrsPath << "'" << std::endl;
    return 0;
  }
  std::cout << "Opened: '" << vrsPath << "'" << std::endl;
  // start viewer with dataprovider
  auto viewer = std::make_shared<visualization::PilotDatasetViewer>(dataProvider.get(), 1280, 800);
  // initialize and setup datastreams
  auto initDataStreams = viewer->initDataStreams(kImageStreamIds, {}, {});
  double currentTimestampSec = initDataStreams.first;
  double fastestNominalRateHz = initDataStreams.second;
  // read and visualize datastreams at desired speed
  const double waitTimeSec = (1. / fastestNominalRateHz) / 10.;
  std::thread readerThread([&viewer, &dataProvider, &currentTimestampSec, &waitTimeSec]() {
    auto start = std::chrono::steady_clock::now();
    while (!dataProvider->atLastRecords()) {
      if (viewer->readData(currentTimestampSec)) {
        currentTimestampSec += waitTimeSec;
        // subtract time it took to load data from wait time
        double thisWaitTimeSec =
            waitTimeSec - since<std::chrono::microseconds>(start).count() * 1e-6;
        if (thisWaitTimeSec > 0.) {
          std::this_thread::sleep_for(std::chrono::nanoseconds(
              static_cast<int64_t>(thisWaitTimeSec * 1e9 / viewer->getPlaybackSpeedFactor())));
        }
        start = std::chrono::steady_clock::now();
      }
    }
    std::cout << "Finished reading records" << std::endl;
  });
  viewer->run();
  readerThread.join();
  return 0;
}
