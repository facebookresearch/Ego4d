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

#include <functional>
#include <iostream>

#include <cereal/external/rapidjson/document.h>
#include <cereal/external/rapidjson/rapidjson.h>

#include <camera/projection/FisheyeRadTanThinPrism.h>
#include <camera/projection/KannalaBrandtK3.h>
#include <models/DeviceModel.h>

namespace ark {
namespace datatools {
namespace sensors {

namespace {

// Circular mask radius value for full resolution RGB and SLAM cameras
const int kSlamValidRadius = 330;
const int kRgbValidRadius = 1415;

// Build the CAD pose from position and rotation vectors as a SE3D pose.
Sophus::SE3d buildCADPose(const std::array<float, 9>& poseArray) {
  Eigen::Vector3d position(poseArray[0], poseArray[1], poseArray[2]);

  Eigen::Matrix3d R;
  R.col(1) = Eigen::Vector3d(poseArray[3], poseArray[4], poseArray[5]);
  R.col(2) = Eigen::Vector3d(poseArray[6], poseArray[7], poseArray[8]);
  R.col(0) = R.col(1).cross(R.col(2));
  return Sophus::SE3d(Sophus::makeRotationMatrix(R), position);
}

// Hardcoded values to be used by getSensorPoseByLabel to compute poses
// for barometer, magnetometer and microphone sensors
// There are two subtypes for Aria devices: DVT-L(large) and DVT-S(small)
// Hardcoded values differ for the two subtypes
const std::unordered_map<std::string, const std::unordered_map<std::string, Sophus::SE3d>>
    T_Device_FrameMap = {
        {"DVT-L",
         {{"camera-slam-left",
           buildCADPose(
               {0.071351,
                0.002372,
                0.008454,
                0.793353,
                0.000000,
                -0.608761,
                0.607927,
                -0.052336,
                0.792266})},
          {"camera-slam-right",
           buildCADPose(
               {-0.071351,
                0.002372,
                0.008454,
                0.793353,
                0.000000,
                0.608761,
                -0.607927,
                -0.052336,
                0.792266})},
          {"camera-et-left",
           buildCADPose(
               {0.055753,
                -0.019589,
                0.004786,
                0.034096,
                -0.508436,
                -0.860424,
                -0.674299,
                0.623745,
                -0.395300})},
          {"camera-et-right",
           buildCADPose(
               {-0.055753,
                -0.019589,
                0.004786,
                -0.034096,
                -0.508436,
                -0.860424,
                0.674299,
                0.623745,
                -0.395300})},
          {"camera-rgb",
           buildCADPose(
               {0.058250,
                0.007186,
                0.012096,
                1.000000,
                0.000000,
                0.000000,
                0.000000,
                -0.130526,
                0.991445})},
          {"baro0",
           buildCADPose(
               {-0.009258,
                0.010842,
                0.017200,
                0.000000,
                0.000000,
                -1.000000,
                0.173648,
                0.984808,
                0.000000})},
          {"mag0",
           buildCADPose(
               {0.066201,
                -0.005760,
                -0.001777,
                0.588330,
                0.006520,
                -0.808595,
                -0.808020,
                0.043280,
                -0.587563})},
          {"mic0",
           buildCADPose(
               {-0.046137,
                -0.029296,
                0.006233,
                0.981006,
                -0.109790,
                0.159915,
                -0.087780,
                -0.986430,
                -0.138744})},
          {"mic1",
           buildCADPose(
               {0.009200, 0.010304, 0.01725, -0.984807, -0.173648, 0, -0.173648, 0.984807, 0})},
          {"mic2",
           buildCADPose(
               {0.046138,
                -0.029297,
                0.006233,
                -0.981006,
                -0.109790,
                0.159915,
                0.087780,
                -0.986430,
                -0.138744})},
          {"mic3",
           buildCADPose(
               {0.065925,
                0.011961,
                0.004305,
                -0.017386,
                0.001521,
                0.999848,
                0.087156,
                0.996195,
                0.000000})},
          {"mic4",
           buildCADPose(
               {-0.054800,
                0.013142,
                0.010960,
                0.965337,
                0.033710,
                0.258819,
                -0.034899,
                0.999391,
                0.000000})},
          {"mic5",
           buildCADPose(
               {0.072318,
                0.008272,
                -0.094955,
                0.002477,
                -0.996176,
                0.087334,
                0.990114,
                -0.009805,
                -0.139920})},
          {"mic6",
           buildCADPose(
               {-0.072319,
                0.008271,
                -0.094955,
                -0.002477,
                -0.996176,
                0.087334,
                -0.990114,
                -0.009805,
                -0.139920})}}},
        {"DVT-S",
         {{"camera-slam-left",
           buildCADPose(
               {0.069051,
                0.002372,
                0.009254,
                0.793353,
                0.000000,
                -0.608761,
                0.607927,
                -0.052336,
                0.792266})},
          {"camera-slam-right",
           buildCADPose(
               {-0.069051,
                0.002372,
                0.009254,
                0.793353,
                0.000000,
                0.608761,
                -0.607927,
                -0.052336,
                0.792266})},
          {"camera-et-left",
           buildCADPose(
               {0.054298,
                -0.018500,
                0.006210,
                0.034478,
                -0.508118,
                -0.860597,
                -0.674245,
                0.623794,
                -0.395316})},
          {"camera-et-right",
           buildCADPose(
               {-0.054298,
                -0.018500,
                0.006210,
                -0.034478,
                -0.508118,
                -0.860597,
                0.674245,
                0.623794,
                -0.395316})},
          {"camera-rgb",
           buildCADPose(
               {0.056000,
                0.007121,
                0.012883,
                1.000000,
                0.000000,
                0.000000,
                0.000000,
                -0.130526,
                0.991445})},
          {"baro0",
           buildCADPose(
               {-0.009258,
                0.010842,
                0.017200,
                0.000000,
                0.000000,
                -1.000000,
                0.173648,
                0.984808,
                0.000000})},
          {"mag0",
           buildCADPose(
               {0.064372,
                -0.005868,
                0.000699,
                0.587481,
                -0.015929,
                -0.809081,
                -0.808020,
                0.043280,
                -0.587563})},
          {"mic0",
           buildCADPose(
               {-0.045906,
                -0.027938,
                0.006667,
                0.97508224,
                -0.160939007,
                0.152686805,
                -0.14019156,
                -0.980440,
                -0.138144})},
          {"mic1",
           buildCADPose(
               {0.009161,
                0.010231,
                0.017250,
                -0.984808,
                -0.173648,
                0.000000,
                -0.173648,
                0.984808,
                0.000000})},
          {"mic2",
           buildCADPose(
               {0.045905,
                -0.027931,
                0.006668,
                -0.975082,
                -0.160938,
                0.152687,
                0.140190,
                -0.980440,
                -0.138146})},
          {"mic3",
           buildCADPose(
               {0.063471,
                0.012034,
                0.005566,
                -0.017386,
                0.001521,
                0.999848,
                0.087156,
                0.996195,
                0.000000})},
          {"mic4",
           buildCADPose(
               {-0.052398,
                0.013200,
                0.012160,
                0.965337,
                0.033710,
                0.258819,
                -0.034899,
                0.999391,
                0.000000})},
          {"mic5",
           buildCADPose(
               {0.069856,
                0.008270,
                -0.093105,
                0.002466,
                -0.996176,
                0.087334,
                0.990147,
                -0.009795,
                -0.139689})},
          {"mic6",
           buildCADPose(
               {-0.069822,
                0.008268,
                -0.093138,
                -0.002487,
                -0.996176,
                0.087333,
                -0.990081,
                -0.009815,
                -0.140151})}}}};

Eigen::VectorXd parseVectorXdFromJson(const fb_rapidjson::Value& json) {
  Eigen::VectorXd vec(json.Size());
  for (size_t i = 0; i < json.Size(); ++i) {
    vec(i) = json[i].GetDouble();
  }
  return vec;
}

Eigen::Vector3d parseVector3dFromJson(const fb_rapidjson::Value& json) {
  assert(json.Size() == 3);

  return {json[0].GetDouble(), json[1].GetDouble(), json[2].GetDouble()};
}

Eigen::Matrix3d parseMatrix3dFromJson(const fb_rapidjson::Value& json) {
  assert(json.Size() == 3);

  Eigen::Matrix3d mat;
  for (size_t i = 0; i < 3; ++i) {
    mat.row(i) = parseVector3dFromJson(json[i]).transpose().eval();
  }
  return mat;
}

Sophus::SE3d parseSe3dFromJson(const fb_rapidjson::Value& json) {
  Eigen::Vector3d translation = parseVector3dFromJson(json["Translation"]);

  assert(json["UnitQuaternion"].Size() == 2);
  double qReal = json["UnitQuaternion"][0].GetDouble();
  Eigen::Vector3d qImag = parseVector3dFromJson(json["UnitQuaternion"][1]);

  Eigen::Quaterniond rotation(qReal, qImag.x(), qImag.y(), qImag.z());
  return {rotation, translation};
}

CameraCalibration parseCameraCalibFromJson(const fb_rapidjson::Value& json) {
  CameraCalibration camCalib;
  camCalib.label = json["Label"].GetString();
  camCalib.T_Device_Camera = parseSe3dFromJson(json["T_Device_Camera"]);

  const std::string projectionModelName = json["Projection"]["Name"].GetString();
  if (projectionModelName == "FisheyeRadTanThinPrism") {
    camCalib.projectionModel.modelName = CameraProjectionModel::ModelType::Fisheye624;
  } else if (projectionModelName == "KannalaBrandtK3") {
    camCalib.projectionModel.modelName = CameraProjectionModel::ModelType::KannalaBrandtK3;
  }
  camCalib.projectionModel.projectionParams = parseVectorXdFromJson(json["Projection"]["Params"]);

  // Handle sensor valid radius
  if (camCalib.label == "camera-rgb") {
    camCalib.validRadius = kRgbValidRadius;
  } else if (camCalib.label == "camera-slam-left" || camCalib.label == "camera-slam-right") {
    camCalib.validRadius = kSlamValidRadius;
  }
  // else nothing is needed (value should be -1)

  return camCalib;
}

LinearRectificationModel parseRectModelFromJson(const fb_rapidjson::Value& json) {
  LinearRectificationModel model;
  model.rectificationMatrix = parseMatrix3dFromJson(json["Model"]["RectificationMatrix"]);
  model.bias = parseVector3dFromJson(json["Bias"]["Offset"]);
  return model;
}

ImuCalibration parseImuCalibFromJson(const fb_rapidjson::Value& json) {
  ImuCalibration imuCalib;
  imuCalib.label = json["Label"].GetString();
  imuCalib.accel = parseRectModelFromJson(json["Accelerometer"]);
  imuCalib.gyro = parseRectModelFromJson(json["Gyroscope"]);
  imuCalib.T_Device_Imu = parseSe3dFromJson(json["T_Device_Imu"]);
  return imuCalib;
}

Sophus::SE3d getSensorPoseByLabel(
    const DeviceModel& calib,
    const std::string& label,
    const std::string& deviceSubtype) {
  if (!calib.getCameraCalib("camera-slam-left").has_value()) {
    std::cerr << "Camera calibration must exist for loading device model, exiting" << std::endl;
    exit(1);
  }
  return calib.getCameraCalib("camera-slam-left").value().T_Device_Camera *
      T_Device_FrameMap.at(deviceSubtype).at("camera-slam-left").inverse() *
      T_Device_FrameMap.at(deviceSubtype).at(label);
}

MagnetometerCalibration parseMagnetometerCalibrationFromJson(
    const fb_rapidjson::Value& json,
    const DeviceModel& calib,
    const std::string& deviceSubtype) {
  MagnetometerCalibration magnetometerCalibration;
  magnetometerCalibration.label = json["Label"].GetString();
  magnetometerCalibration.model = parseRectModelFromJson(json);
  magnetometerCalibration.T_Device_Magnetometer =
      getSensorPoseByLabel(calib, magnetometerCalibration.label, deviceSubtype);
  return magnetometerCalibration;
}

BarometerCalibration parseBarometerCalibrationFromJson(
    const fb_rapidjson::Value& json,
    const DeviceModel& calib,
    const std::string& deviceSubtype) {
  BarometerCalibration barometerCalibration;
  barometerCalibration.label = json["Label"].GetString();
  barometerCalibration.pressure.slope = json["PressureModel"]["Slope"].GetDouble();
  barometerCalibration.pressure.offsetPa = json["PressureModel"]["OffsetPa"].GetDouble();
  barometerCalibration.T_Device_Barometer =
      getSensorPoseByLabel(calib, barometerCalibration.label, deviceSubtype);
  return barometerCalibration;
}

MicrophoneCalibration parseMicrophoneCalibrationFromJson(
    const fb_rapidjson::Value& json,
    const DeviceModel& calib,
    const std::string& deviceSubtype) {
  MicrophoneCalibration microphoneCalibration;
  microphoneCalibration.label = json["Label"].GetString();
  microphoneCalibration.dSensitivity1KDbv = json["DSensitivity1KDbv"].GetDouble();
  microphoneCalibration.T_Device_Microphone =
      getSensorPoseByLabel(calib, microphoneCalibration.label, deviceSubtype);
  return microphoneCalibration;
}

} // namespace

Eigen::Vector2d CameraProjectionModel::getFocalLengths() const {
  switch (modelName) {
    case ModelType::KannalaBrandtK3:
      return {
          projectionParams[KannalaBrandtK3Projection::kFocalXIdx],
          projectionParams[KannalaBrandtK3Projection::kFocalYIdx]};
    case ModelType::Fisheye624:
      return {projectionParams[Fisheye624::kFocalXIdx], projectionParams[Fisheye624::kFocalYIdx]};
  }
  // Intentionally skipping default to raise a compile error when new models are added.
  assert(false);
  // return 0s to remove compiler warning
  return Eigen::Vector2d::Zero();
}

Eigen::Vector2d CameraProjectionModel::getPrincipalPoint() const {
  switch (modelName) {
    case ModelType::KannalaBrandtK3:
      return {
          projectionParams[KannalaBrandtK3Projection::kPrincipalPointColIdx],
          projectionParams[KannalaBrandtK3Projection::kPrincipalPointRowIdx]};
    case ModelType::Fisheye624:
      return {
          projectionParams[Fisheye624::kPrincipalPointColIdx],
          projectionParams[Fisheye624::kPrincipalPointRowIdx]};
  }
  // Intentionally skipping default to raise a compile error when new models are added.
  assert(false);
  // return 0s to remove compiler warning
  return Eigen::Vector2d::Zero();
}

Eigen::Vector2d CameraProjectionModel::project(const Eigen::Vector3d& p) const {
  switch (modelName) {
    case ModelType::KannalaBrandtK3:
      return KannalaBrandtK3Projection::project(p, projectionParams);
    case ModelType::Fisheye624:
      return Fisheye624::project(p, projectionParams);
      // Intentionally skipping default to raise a compile error when new models are added.
  }
  assert(false);
  // return 0s to remove compiler warning
  return Eigen::Vector2d::Zero();
}

Eigen::Vector3d CameraProjectionModel::unproject(const Eigen::Vector2d& uv) const {
  switch (modelName) {
    case ModelType::KannalaBrandtK3:
      return KannalaBrandtK3Projection::unproject(uv, projectionParams);
    case ModelType::Fisheye624:
      return Fisheye624::unproject(uv, projectionParams);
      // Intentionally skipping default to raise a compile error when new models are added.
  }
  assert(false);
  // return 0s to remove compiler warning
  return Eigen::Vector3d::Zero();
}

Eigen::Vector3d LinearRectificationModel::compensateForSystematicErrorFromMeasurement(
    const Eigen::Vector3d& v_raw) const {
  return rectificationMatrix.inverse() * (v_raw - bias);
}

Eigen::Vector3d LinearRectificationModel::distortWithSystematicError(
    const Eigen::Vector3d& v_compensated) const {
  return rectificationMatrix * v_compensated + bias;
}

std::string DeviceModel::getDeviceSubtype() const {
  return deviceSubtype_;
}

std::optional<Sophus::SE3d> DeviceModel::getCADSensorPose(const std::string& label) const {
  if (deviceSubtype_.size() > 0 && T_Device_FrameMap.count(deviceSubtype_) &&
      T_Device_FrameMap.at(deviceSubtype_).count(label)) {
    return T_Device_FrameMap.at(deviceSubtype_).at(label);
  }
  return {};
}

std::optional<CameraCalibration> DeviceModel::getCameraCalib(const std::string& label) const {
  if (cameraCalibs_.find(label) == cameraCalibs_.end()) {
    return {};
  }
  return cameraCalibs_.at(label);
}

std::optional<ImuCalibration> DeviceModel::getImuCalib(const std::string& label) const {
  if (imuCalibs_.find(label) == imuCalibs_.end()) {
    return {};
  }
  return imuCalibs_.at(label);
}

std::optional<MagnetometerCalibration> DeviceModel::getMagnetometerCalib(
    const std::string& label) const {
  if (magnetometerCalibs_.find(label) == magnetometerCalibs_.end()) {
    return {};
  }
  return magnetometerCalibs_.at(label);
}

std::optional<BarometerCalibration> DeviceModel::getBarometerCalib(const std::string& label) const {
  if (barometerCalibs_.find(label) == barometerCalibs_.end()) {
    return {};
  }
  return barometerCalibs_.at(label);
}

std::optional<MicrophoneCalibration> DeviceModel::getMicrophoneCalib(
    const std::string& label) const {
  if (microphoneCalibs_.find(label) == microphoneCalibs_.end()) {
    return {};
  }
  return microphoneCalibs_.at(label);
}

namespace utils {
// Provide an alternative to std::regex_replace
void ReplaceStringInPlace(
    std::string& subject,
    const std::string& search,
    const std::string& replace) {
  size_t pos = 0;
  while ((pos = subject.find(search, pos)) != std::string::npos) {
    subject.replace(pos, search.length(), replace);
    pos += replace.length();
  }
}

// Replace Document json with an array version if we have a string
// Note: Online calibration data is saved with a string,
//  we can convert it to an array with this code.
// See DeviceModelTests.cpp to know more about the JSON message format.
fb_rapidjson::Value& NormalizeToArrayIfString(
    fb_rapidjson::Value& value,
    fb_rapidjson::Document& doc) {
  if (value.IsString()) {
    // Convert substring to Json object
    std::string temp = value.GetString();
    // Adjust string to be a valid JSON
    std::replace(temp.begin(), temp.end(), '\'', '\"');
    ReplaceStringInPlace(temp, "True", "true");
    ReplaceStringInPlace(temp, "False", "false");
    fb_rapidjson::Document doc_temp;
    doc_temp.Parse(temp.c_str());
    value.CopyFrom(doc_temp, doc.GetAllocator());
  }
  return value;
}
} // namespace utils

DeviceModel DeviceModel::fromJson(const fb_rapidjson::Document& json) {
  DeviceModel calib;
  {
    // Use a local Json Document copy if we need to overwrite some fields
    fb_rapidjson::Document jsonCpy;
    jsonCpy.CopyFrom(json, jsonCpy.GetAllocator());

    if (json.FindMember("CameraCalibrations") != json.MemberEnd()) {
      const fb_rapidjson::Value& v =
          utils::NormalizeToArrayIfString(jsonCpy["CameraCalibrations"], jsonCpy);
      for (const auto& camJson : v.GetArray()) {
        CameraCalibration camCalib = parseCameraCalibFromJson(camJson);
        auto& ref = calib.cameraCalibs_[camCalib.label];
        ref = std::move(camCalib);
      }
    }
    if (json.FindMember("ImuCalibrations") != json.MemberEnd()) {
      const fb_rapidjson::Value& v =
          utils::NormalizeToArrayIfString(jsonCpy["ImuCalibrations"], jsonCpy);
      for (const auto& imuJson : v.GetArray()) {
        ImuCalibration imuCalib = parseImuCalibFromJson(imuJson);
        auto& ref = calib.imuCalibs_[imuCalib.label];
        ref = std::move(imuCalib);
      }
    }
  }

  if (json.FindMember("DeviceClassInfo") != json.MemberEnd()) {
    calib.deviceSubtype_ = json["DeviceClassInfo"]["BuildVersion"].GetString();
    if (json.FindMember("MagCalibrations") != json.MemberEnd()) {
      for (const auto& magnetometerJson : json["MagCalibrations"].GetArray()) {
        MagnetometerCalibration magnetometerCalib =
            parseMagnetometerCalibrationFromJson(magnetometerJson, calib, calib.deviceSubtype_);
        auto& ref = calib.magnetometerCalibs_[magnetometerCalib.label];
        ref = std::move(magnetometerCalib);
      }
    }
    if (json.FindMember("BaroCalibrations") != json.MemberEnd()) {
      for (const auto& barometerJson : json["BaroCalibrations"].GetArray()) {
        BarometerCalibration barometerCalib =
            parseBarometerCalibrationFromJson(barometerJson, calib, calib.deviceSubtype_);
        auto& ref = calib.barometerCalibs_[barometerCalib.label];
        ref = std::move(barometerCalib);
      }
    }
    if (json.FindMember("MicCalibrations") != json.MemberEnd()) {
      for (const auto& microphoneJson : json["MicCalibrations"].GetArray()) {
        MicrophoneCalibration microphoneCalib =
            parseMicrophoneCalibrationFromJson(microphoneJson, calib, calib.deviceSubtype_);
        auto& ref = calib.microphoneCalibs_[microphoneCalib.label];
        ref = std::move(microphoneCalib);
      }
    }
  }
  return calib;
}

DeviceModel DeviceModel::fromJson(const std::string& jsonStr) {
  fb_rapidjson::Document doc;
  doc.Parse(jsonStr.c_str());
  return DeviceModel::fromJson(doc);
}

Eigen::Vector3d DeviceModel::transform(
    const Eigen::Vector3d& p_Source,
    const std::string& sourceSensorLabel,
    const std::string& destSensorLabel) const {
  Sophus::SE3d T_Device_Source, T_Device_Dest;
  if (cameraCalibs_.find(sourceSensorLabel) != cameraCalibs_.end()) {
    T_Device_Source = cameraCalibs_.at(sourceSensorLabel).T_Device_Camera;
  } else if (imuCalibs_.find(sourceSensorLabel) != imuCalibs_.end()) {
    T_Device_Source = imuCalibs_.at(sourceSensorLabel).T_Device_Imu;
  } else {
    assert(false);
  }

  if (cameraCalibs_.find(destSensorLabel) != cameraCalibs_.end()) {
    T_Device_Dest = cameraCalibs_.at(destSensorLabel).T_Device_Camera;
  } else if (imuCalibs_.find(destSensorLabel) != imuCalibs_.end()) {
    T_Device_Dest = imuCalibs_.at(destSensorLabel).T_Device_Imu;
  } else {
    assert(false);
  }

  return T_Device_Dest.inverse() * T_Device_Source * p_Source;
}

std::vector<std::string> DeviceModel::getCameraLabels() const {
  std::vector<std::string> cameraLabels;
  for (const auto& [key, _] : cameraCalibs_) {
    cameraLabels.push_back(key);
  }
  return cameraLabels;
}

std::vector<std::string> DeviceModel::getImuLabels() const {
  std::vector<std::string> imuLabels;
  for (const auto& [key, _] : imuCalibs_) {
    imuLabels.push_back(key);
  }
  return imuLabels;
}

std::vector<std::string> DeviceModel::getMagnetometerLabels() const {
  std::vector<std::string> magnetometerLabels;
  for (const auto& [key, _] : magnetometerCalibs_) {
    magnetometerLabels.push_back(key);
  }
  return magnetometerLabels;
}

std::vector<std::string> DeviceModel::getBarometerLabels() const {
  std::vector<std::string> barometerLabels;
  for (const auto& [key, _] : barometerCalibs_) {
    barometerLabels.push_back(key);
  }
  return barometerLabels;
}

std::vector<std::string> DeviceModel::getMicrophoneLabels() const {
  std::vector<std::string> microphoneLabels;
  for (const auto& [key, _] : microphoneCalibs_) {
    microphoneLabels.push_back(key);
  }
  return microphoneLabels;
}

bool DeviceModel::tryCropAndScaleCameraCalibration(
    const std::string& label,
    const int nativeResolution,
    const int newWidth) {
  // Camera calibration should be rectified only once
  if (updatedCameraCalibs_.count(label)) {
    return true;
  }
  if (cameraCalibs_.find(label) != cameraCalibs_.end()) {
    auto& cameraCalib = cameraCalibs_.at(label);
    // Aria supports two RGB resolution:
    // - Full res 2880x2880 & Medium res 1408x1408
    // We are applying here the necessary intrinsics parameters change
    //  to fit camera calibration to the used resolution
    if (label == "camera-rgb") {
      Eigen::VectorXd& camParams = cameraCalib.projectionModel.projectionParams;
      // Testing if principal point is appropriate for this image width
      if (camParams[Fisheye624::kPrincipalPointColIdx] * 2 >
          newWidth) { // We need to rescale calibration parameters

        // Assume the resolution change follows the following steps:
        // - centered cropping -> sensor pixel binning
        const double rescaleFactor = std::floor(
            nativeResolution / static_cast<double>(newWidth)); // binning can only be an integer
        const double halfCroppedSize = (nativeResolution - newWidth * rescaleFactor) / 2.0;
        camParams[Fisheye624::kPrincipalPointColIdx] -= halfCroppedSize;
        camParams[Fisheye624::kPrincipalPointRowIdx] -= halfCroppedSize;

        camParams[Fisheye624::kFocalXIdx] /= rescaleFactor;
        camParams[Fisheye624::kPrincipalPointColIdx] /= rescaleFactor;
        camParams[Fisheye624::kPrincipalPointRowIdx] /= rescaleFactor;

        if (cameraCalib.validRadius != -1) {
          cameraCalib.validRadius /= rescaleFactor;
        }

        updatedCameraCalibs_.insert(label);
        return true;
      }
    } else if (label == "camera-et-left" || label == "camera-et-right") {
      Eigen::VectorXd& camParams = cameraCalib.projectionModel.projectionParams;
      // Testing if principal point is appropriate for this image width
      if (camParams[KannalaBrandtK3Projection::kPrincipalPointColIdx] * 2 >
          newWidth) { // We need to rescale calibration parameters

        // Assume the resolution change following a linear rescaling
        const double rescaleFactor = std::floor(nativeResolution / static_cast<double>(newWidth));
        camParams[KannalaBrandtK3Projection::kFocalXIdx] /= rescaleFactor;
        camParams[KannalaBrandtK3Projection::kFocalYIdx] /= rescaleFactor;
        camParams[KannalaBrandtK3Projection::kPrincipalPointColIdx] /= rescaleFactor;
        camParams[KannalaBrandtK3Projection::kPrincipalPointRowIdx] /= rescaleFactor;

        updatedCameraCalibs_.insert(label);
        return true;
      }
    }
  }
  return false;
}

} // namespace sensors
} // namespace datatools
} // namespace ark
