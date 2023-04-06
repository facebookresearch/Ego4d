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

#include "utils.h"
#include <filesystem>
#include <iostream>
#include <set>
#ifndef CSV_IO_NO_THREAD
#define CSV_IO_NO_THREAD
#endif
#include "fast-cpp-csv-parser/csv.h"

namespace ark {
namespace datatools {
namespace dataprovider {

void getDirContent(const std::string& dirPath, std::vector<std::string>& dirContent) {
  std::set<std::string> pathSet;
  for (auto& p : std::filesystem::directory_iterator(dirPath)) {
    pathSet.insert(p.path());
  }
  std::copy(pathSet.begin(), pathSet.end(), std::back_inserter(dirContent));
}

constexpr const char* kTimeSyncPathSuffix = "synchronization/timestamp_map.csv";

std::string getTimeSyncPath(const std::string& vrsPath) {
  std::string timeSyncCsvPath;
  auto pathSplitted = ark::datatools::dataprovider::strSplit(vrsPath, '/');
  pathSplitted.pop_back();
  for (auto& subfolder : pathSplitted) {
    timeSyncCsvPath += subfolder + "/";
  }
  timeSyncCsvPath += kTimeSyncPathSuffix;
  return timeSyncCsvPath;
}

std::map<int64_t, int64_t> readTimeSyncCsv(const std::string& inputTimeSyncCsv) {
  std::map<int64_t, int64_t> timeSyncToTimeRecording;
  if (!inputTimeSyncCsv.empty()) {
    io::CSVReader<2> in(inputTimeSyncCsv);
    in.read_header(io::ignore_extra_column, "deviceTimestampNs", "syncedTimestampNs");
    uint64_t tRecording, tSync;
    while (in.read_row(tRecording, tSync)) {
      timeSyncToTimeRecording[tSync] = tRecording;
    }
  }
  std::cout << "Loaded " << timeSyncToTimeRecording.size() << " sync timestamps " << std::endl;
  return timeSyncToTimeRecording;
}

std::vector<std::string> strSplit(const std::string& s, const char delimiter) {
  std::vector<std::string> result;
  std::stringstream ss(s);
  std::string item;
  while (getline(ss, item, delimiter)) {
    result.push_back(item);
  }
  return result;
}

} // namespace dataprovider
} // namespace datatools
} // namespace ark
