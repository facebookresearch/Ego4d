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
#include "AriaViewer.h"
#include "utils.h"

namespace {

template <
    class result_t = std::chrono::milliseconds,
    class clock_t = std::chrono::steady_clock,
    class duration_t = std::chrono::milliseconds>
auto since(std::chrono::time_point<clock_t, duration_t> const& start) {
  return std::chrono::duration_cast<result_t>(clock_t::now() - start);
};

using namespace ark::datatools::dataprovider;

const std::vector<vrs::StreamId> kImageStreamIds = {
    kSlamLeftCameraStreamId,
    kSlamRightCameraStreamId,
    kRgbCameraStreamId};
const std::vector<vrs::StreamId> kImuStreamIds = {kImuRightStreamId, kImuLeftStreamId};
const std::vector<vrs::StreamId> kDataStreams = {
    kMagnetometerStreamId,
    kBarometerStreamId,
    kAudioStreamId,
    kWifiStreamId,
    kBluetoothStreamId,
    kGpsStreamId};

} // namespace

using namespace ark::datatools;

int main(int argc, const char* argv[]) {
  if (argc < 2) {
    fmt::print(stderr, "VRS file path must be provided as the argument, exiting.\n");
    return 0;
  }

  std::vector<std::shared_ptr<visualization::AriaViewer>> viewers;
  std::vector<std::map<int64_t, int64_t>> timeSyncToTimeRecordings;
  std::vector<std::unique_ptr<dataprovider::AriaVrsDataProvider>> dataProviders;
  for (int argi = 1; argi < argc; ++argi) {
    std::string vrsPath = argv[argi];
    dataProviders.emplace_back(std::make_unique<dataprovider::AriaVrsDataProvider>());
    auto& dataProvider = dataProviders.back();
    if (!dataProvider->open(vrsPath)) {
      fmt::print(stderr, "Failed to open '{}'.\n", vrsPath);
      return 0;
    }
    fmt::print(stdout, "opened '{}'.\n", vrsPath);
    timeSyncToTimeRecordings.emplace_back(
        dataprovider::readTimeSyncCsv(dataprovider::getTimeSyncPath(vrsPath)));
    // start viewer with dataprovider
    viewers.emplace_back(std::make_shared<visualization::AriaViewer>(
        dataProvider.get(), 1280, 800, "AriaViewer", argi - 1));
    // initialize and setup datastreams
    viewers.back()->initDataStreams(kImageStreamIds, kImuStreamIds, kDataStreams);
  }
  // get joint synced time table across all recordings
  std::set<int64_t> baseTimeSet;
  for (const auto& timeSyncToTimeRecording : timeSyncToTimeRecordings) {
    for (const auto& [tSync, tRecord] : timeSyncToTimeRecording) {
      baseTimeSet.insert(tSync);
    }
  }
  // the sorted set of all sync time stamps across the recordings.
  std::vector<int64_t> baseTime(baseTimeSet.begin(), baseTimeSet.end());

  // start viewer threads
  std::vector<std::thread> threads;
  for (auto& viewer : viewers) {
    threads.emplace_back(viewer->runInThread());
  }
  // start data reading thread
  threads.emplace_back([&viewers, &timeSyncToTimeRecordings, &baseTime]() {
    for (size_t t = 1; t < baseTime.size(); ++t) {
      int64_t baseTimeNs = baseTime[t];
      int64_t prevBaseTimeNs = baseTime[t - 1];
      double waitTimeSec = (baseTimeNs - prevBaseTimeNs) * 1e-9;
      auto start = std::chrono::steady_clock::now();
      // check that all players are in playing mode
      bool allPlaying = false;
      while (!allPlaying) {
        allPlaying = true;
        for (const auto& viewer : viewers) {
          allPlaying &= viewer->isPlaying();
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
      }
      for (size_t i = 0; i < viewers.size(); ++i) {
        if (timeSyncToTimeRecordings[i].find(baseTimeNs) == timeSyncToTimeRecordings[i].end()) {
          std::cout << "recording " << i
                    << ": cannot find corresponding recording time skipping base time "
                    << baseTimeNs << std::endl;
          continue;
        }
        double currentTimestampSec = timeSyncToTimeRecordings[i][baseTimeNs] * 1.e-9;
        viewers[i]->readData(currentTimestampSec);
      }
      // subtract time it took to load data from wait time
      double thisWaitTimeSec = waitTimeSec - since<std::chrono::microseconds>(start).count() * 1e-6;
      if (thisWaitTimeSec > 0.) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(
            static_cast<int64_t>(thisWaitTimeSec * 1e9 / viewers[0]->getPlaybackSpeedFactor())));
      }
    }
    std::cout << "Finished reading records" << std::endl;
  });
  // join all threads
  for (auto& thread : threads) {
    thread.join();
  }
  return 0;
}
