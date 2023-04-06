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

namespace ark::datatools::dataprovider {

// struct to hold one speech-to-text datum spanning the time from tStart to tEnd (in nano
// seconds).
struct SpeechToTextDatum {
  // start and end time of the utterance
  int64_t tStart_ns = 0;
  int64_t tEnd_ns = 0;
  // the transcribed text
  std::string text = "";
  // the confidence in the transcription
  float confidence = 0.;

  // compute duration in nano seconds of the utterance
  int64_t duration_ns() const {
    return tEnd_ns - tStart_ns;
  }
  // compute duration in seconds of the utterance
  double duration_s() const {
    return duration_ns() * 1e-9;
  }
};

} // namespace ark::datatools::dataprovider
