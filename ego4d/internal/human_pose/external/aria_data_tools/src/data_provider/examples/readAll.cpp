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

#include "data_provider/AriaVrsDataProvider.h"

int main(int argc, char** argv) {
  if (argc != 2) {
    fmt::print(stderr, "VRS file path must be provided as an argument, exiting.\n");
    return 0;
  }
  ark::datatools::dataprovider::AriaVrsDataProvider vrsDataProvider;
  if (!vrsDataProvider.openFile(argv[1])) {
    fmt::print(stderr, "Failed to open '{}'.\n", argv[1]);
  }
  // Check existing streams in the file and set players for all of them
  for (auto id : vrsDataProvider.getStreamsInFile()) {
    vrsDataProvider.setStreamPlayer(id);
  }
  vrsDataProvider.setVerbose(true);
  vrsDataProvider.readAllRecords();

  return 0;
}
