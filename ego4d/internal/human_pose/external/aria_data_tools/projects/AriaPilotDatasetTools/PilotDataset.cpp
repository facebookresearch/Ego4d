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

#include "PilotDataset.h"

#include "AriaStreamIds.h"
#include "utils.h"

#ifndef CSV_IO_NO_THREAD
#define CSV_IO_NO_THREAD
#endif
#include "fast-cpp-csv-parser/csv.h"

#include <filesystem>

namespace {
constexpr const char* kTrajectoryPathSuffix = "location/trajectory.csv";
constexpr const char* kEyetrackingPathSuffix = "eyetracking/et_in_rgb_stream.csv";
constexpr const char* kSpeechToTextPathSuffix = "speech2text/speech_aria_domain.csv";

std::map<int64_t, Sophus::SE3d> readPosesFromCsvFile(const std::string& inputPoseCsv) {
  std::map<int64_t, Sophus::SE3d> timeToPoseMap;
  if (!inputPoseCsv.empty()) {
    io::CSVReader<8> in(inputPoseCsv);
    in.read_header(
        io::ignore_extra_column,
        "# timestamp_unix_ns",
        "x_t_world_left_imu",
        "y_t_world_left_imu",
        "z_t_world_left_imu",
        "qw_R_world_left_imu",
        "qx_R_world_left_imu",
        "qy_R_world_left_imu",
        "qz_R_world_left_imu");
    int64_t ts;
    Eigen::Vector3d t;
    Eigen::Quaterniond q;
    while (in.read_row(ts, t(0), t(1), t(2), q.w(), q.x(), q.y(), q.z())) {
      timeToPoseMap[ts] = Sophus::SE3d(Sophus::SO3d(q), t);
    }
  }
  std::cout << "Loaded " << timeToPoseMap.size() << " poses" << std::endl;
  return timeToPoseMap;
}

std::map<int64_t, Eigen::Vector2f> readEyetrackingFromCsvFile(
    const std::string& inputEyetrackingCsv) {
  std::map<int64_t, Eigen::Vector2f> timeToEtMap;
  if (!inputEyetrackingCsv.empty()) {
    io::CSVReader<3> in(inputEyetrackingCsv);
    in.read_header(
        io::ignore_extra_column, "timestamp_unix_ns", "calib_x [pixel]", "calib_y [pixel]");
    int64_t ts;
    Eigen::Vector2f etCalib_im;
    while (in.read_row(ts, etCalib_im(0), etCalib_im(1))) {
      timeToEtMap[ts] = etCalib_im;
    }
  }
  std::cout << "Loaded " << timeToEtMap.size() << " eye tracking points" << std::endl;
  return timeToEtMap;
}

std::map<int64_t, ark::datatools::dataprovider::SpeechToTextDatum> readSpeechToTextFromCsvFile(
    const std::string& inputSpeechToTextCsv) {
  std::map<int64_t, ark::datatools::dataprovider::SpeechToTextDatum> timeToSpeechToTextMap;
  if (!inputSpeechToTextCsv.empty()) {
    io::CSVReader<4, io::trim_chars<' ', '\t'>, io::double_quote_escape<',', '"'>> in(
        inputSpeechToTextCsv);
    in.read_header(io::ignore_extra_column, "startTime_ns", "endTime_ns", "written", "confidence");
    int64_t tStart, tEnd;
    std::string text;
    float confidence;
    while (in.read_row(tStart, tEnd, text, confidence)) {
      timeToSpeechToTextMap[tStart] = {
          .tStart_ns = tStart, .tEnd_ns = tEnd, .text = text, .confidence = confidence};
    }
  }
  std::cout << "Loaded " << timeToSpeechToTextMap.size() << " speech2text points" << std::endl;
  return timeToSpeechToTextMap;
}

std::optional<Sophus::SE3d> queryPose(
    const int64_t timestamp,
    const std::map<int64_t, Sophus::SE3d>& timestampToPose) {
  if (timestamp < timestampToPose.begin()->first || timestamp > timestampToPose.rbegin()->first) {
    return {};
  }
  if (timestampToPose.find(timestamp) != timestampToPose.end()) {
    return {};
  }
  // Interpolation
  auto laterPosePtr = timestampToPose.lower_bound(timestamp);
  auto earlyPosePtr = std::prev(laterPosePtr);
  int64_t tsEarly = earlyPosePtr->first;
  int64_t tsLater = laterPosePtr->first;
  Sophus::SE3d poseEarly = earlyPosePtr->second;
  Sophus::SE3d poseLater = laterPosePtr->second;

  double interpFactor =
      static_cast<double>(timestamp - tsEarly) / static_cast<double>(tsLater - tsEarly);
  auto interpQ = poseEarly.unit_quaternion().slerp(interpFactor, poseLater.unit_quaternion());
  auto interpT =
      (1.0 - interpFactor) * poseEarly.translation() + interpFactor * poseLater.translation();
  Sophus::SE3d result;
  result.translation() = interpT;
  result.setRotationMatrix(interpQ.toRotationMatrix());
  return result;
}

std::optional<Eigen::Vector2f> queryEyetrack(
    const int64_t timestamp,
    const std::map<int64_t, Eigen::Vector2f>& timestampToEyetrack) {
  if (timestamp < timestampToEyetrack.begin()->first ||
      timestamp > timestampToEyetrack.rbegin()->first) {
    return {};
  }
  if (timestampToEyetrack.find(timestamp) != timestampToEyetrack.end()) {
    return {};
  }
  // Interpolation
  auto laterEyePtr = timestampToEyetrack.lower_bound(timestamp);
  auto earlyEyePtr = std::prev(laterEyePtr);
  int64_t tsEarly = earlyEyePtr->first;
  int64_t tsLater = laterEyePtr->first;
  Eigen::Vector2f eyeEarly = earlyEyePtr->second;
  Eigen::Vector2f eyeLater = laterEyePtr->second;

  double interpFactor =
      static_cast<double>(timestamp - tsEarly) / static_cast<double>(tsLater - tsEarly);
  auto interpEye = (1.0 - interpFactor) * eyeEarly + interpFactor * eyeLater;
  return interpEye;
}

std::optional<ark::datatools::dataprovider::SpeechToTextDatum> querySpeechToText(
    const int64_t timestamp,
    const std::map<int64_t, ark::datatools::dataprovider::SpeechToTextDatum>&
        timestampToSpeechToText) {
  if (timestamp < timestampToSpeechToText.begin()->first ||
      timestamp > timestampToSpeechToText.rbegin()->second.tEnd_ns) {
    return {};
  }
  if (timestampToSpeechToText.find(timestamp) != timestampToSpeechToText.end()) {
    return {};
  }
  // Interpolation
  auto laterEyePtr = timestampToSpeechToText.lower_bound(timestamp);
  auto earlyEyePtr = std::prev(laterEyePtr);
  int64_t tsEarlyStart = earlyEyePtr->first;
  int64_t tsLaterStart = laterEyePtr->first;
  int64_t tsEarlyEnd = earlyEyePtr->second.tEnd_ns;
  int64_t tsLaterEnd = laterEyePtr->second.tEnd_ns;

  if (tsEarlyStart <= timestamp && timestamp < tsEarlyEnd) {
    return earlyEyePtr->second;
  } else if (tsLaterStart <= timestamp && timestamp < tsLaterEnd) {
    return laterEyePtr->second;
  }
  return {};
}

} // namespace

namespace ark::datatools::dataprovider {

std::optional<std::map<int64_t, Sophus::SE3d>> loadPosesFromCsv(const std::string& posePath) {
  std::string trajectoryCsvFile = "";
  if (!std::filesystem::is_directory(posePath)) {
    trajectoryCsvFile = posePath;
  } else {
    trajectoryCsvFile = (std::filesystem::path(posePath) /= kTrajectoryPathSuffix);
  }
  std::filesystem::path trajectoryCsvPath(trajectoryCsvFile);
  if (!std::filesystem::exists(trajectoryCsvPath)) {
    std::cout << "No pose file found at " << trajectoryCsvPath << " , not visualizing poses"
              << std::endl;
    return {};
  }
  std::cout << "Loading poses file from " << trajectoryCsvFile << std::endl;
  return readPosesFromCsvFile(trajectoryCsvFile);
}

std::optional<std::map<int64_t, SpeechToTextDatum>> loadSpeechToTextFromCsv(
    const std::string& speechToTextPath) {
  std::string speechToTextCsvFile = "";
  if (!std::filesystem::is_directory(speechToTextPath)) {
    speechToTextCsvFile = speechToTextPath;
  } else {
    speechToTextCsvFile = (std::filesystem::path(speechToTextPath) /= kSpeechToTextPathSuffix);
  }
  std::filesystem::path speechToTextCsvPath(speechToTextCsvFile);
  if (!std::filesystem::exists(speechToTextCsvPath)) {
    std::cout << "No speechToText file found at " << speechToTextCsvPath
              << " , not visualizing speech2text" << std::endl;
    return {};
  }
  std::cout << "Loading speech2text file from " << speechToTextCsvFile << std::endl;
  return readSpeechToTextFromCsvFile(speechToTextCsvFile);
}

std::optional<std::map<int64_t, Eigen::Vector2f>> loadEyetrackingFromCsv(
    const std::string& eyetrackingPath) {
  std::string eyetrackingCsvFile = "";
  if (!std::filesystem::is_directory(eyetrackingPath)) {
    eyetrackingCsvFile = eyetrackingPath;
  } else {
    eyetrackingCsvFile = (std::filesystem::path(eyetrackingPath) /= kEyetrackingPathSuffix);
  }
  std::filesystem::path eyetrackingCsvPath(eyetrackingCsvFile);
  if (!std::filesystem::exists(eyetrackingCsvPath)) {
    std::cout << "No eyetracking file found at " << eyetrackingCsvPath
              << " , not visualizing eye tracks" << std::endl;
    return {};
  }
  std::cout << "Loading eye tracking file from " << eyetrackingCsvFile << std::endl;
  return readEyetrackingFromCsvFile(eyetrackingCsvFile);
}

PilotDatasetProvider::PilotDatasetProvider(
    const std::string& posePath,
    const std::string& eyetrackingPath,
    const std::string& speechToTextPath) {
  auto posesData = loadPosesFromCsv(posePath);
  auto eyeTrackingData = loadEyetrackingFromCsv(eyetrackingPath);
  auto speechData = loadSpeechToTextFromCsv(speechToTextPath);
  if (posesData) {
    hasPoses_ = true;
    imuLeftPoses_ = posesData.value();
  }
  if (eyeTrackingData) {
    hasEyetracks_ = true;
    eyetracksOnRgbImage_ = eyeTrackingData.value();
  }
  if (speechData) {
    hasSpeechToText_ = true;
    speechToText_ = speechData.value();
  }
}

std::optional<Eigen::Vector2f> PilotDatasetProvider::getEyetracksOnRgbImage() const {
  if (!hasEyetracks_) {
    return {};
  }

  // Always query using the rgb camera timestamp.
  int64_t rgbTimestampNs =
      static_cast<int64_t>(1e9 * getNextTimestampSec(vrs::StreamId(kRgbCameraStreamId)));
  return queryEyetrack(rgbTimestampNs, eyetracksOnRgbImage_);
}

std::optional<SpeechToTextDatum> PilotDatasetProvider::getSpeechToText() const {
  if (!hasSpeechToText_) {
    return {};
  }

  // Always query using the rgb camera timestamp.
  int64_t rgbTimestampNs =
      static_cast<int64_t>(1e9 * getNextTimestampSec(vrs::StreamId(kRgbCameraStreamId)));
  return querySpeechToText(rgbTimestampNs, speechToText_);
}

std::optional<Sophus::SE3d> PilotDatasetProvider::getLatestPoseOfStream(
    const vrs::StreamId& streamId) {
  if (!hasPoses_) {
    return {};
  }

  // get latest timestamp of pose
  int64_t currentTimestampNs = static_cast<int64_t>(1e9 * getNextTimestampSec(streamId));
  return getPoseOfStreamAtTimestampNs(streamId, currentTimestampNs);
}

std::optional<Sophus::SE3d> PilotDatasetProvider::getPoseOfStreamAtTimestampNs(
    const vrs::StreamId& streamId,
    const int64_t timestampNs) {
  std::optional<Sophus::SE3d> T_world_imuleft;
  T_world_imuleft = queryPose(timestampNs, imuLeftPoses_);
  if (!hasPoses_ || !T_world_imuleft) {
    return {};
  }
  Sophus::SE3d T_Device_stream;

  if (!kDeviceNumericIdToLabel.count(streamId.getNumericName())) {
    std::cerr << "Stream " << streamId.getName() << " not supported" << std::endl;
    return {};
  }
  const std::string labelName = kDeviceNumericIdToLabel.at(streamId.getNumericName());
  if (labelName.find("cam") != std::string::npos) {
    // Camera Calib
    T_Device_stream = deviceModel_.getCameraCalib(labelName)->T_Device_Camera;
  } else if (labelName.find("imu") != std::string::npos) {
    // IMU Calib
    T_Device_stream = deviceModel_.getImuCalib(labelName)->T_Device_Imu;
  }
  auto T_Device_imuleft = deviceModel_.getImuCalib("imu-left")->T_Device_Imu;
  auto T_imuleft_stream = T_Device_imuleft.inverse() * T_Device_stream;
  auto T_world_stream = T_world_imuleft.value() * T_imuleft_stream;
  return T_world_stream;
}

std::optional<Sophus::SE3d> PilotDatasetProvider::getPose() const {
  if (hasPoses_) {
    // Always query using the slam-camera-left timestamp.
    int64_t slamCameraLeftTimestampNs =
        static_cast<int64_t>(1e9 * getNextTimestampSec(kSlamLeftCameraStreamId));
    return queryPose(slamCameraLeftTimestampNs, imuLeftPoses_);
  } else {
    std::cerr << "No poses are loaded." << std::endl;
  }
  return {};
}

} // namespace ark::datatools::dataprovider
