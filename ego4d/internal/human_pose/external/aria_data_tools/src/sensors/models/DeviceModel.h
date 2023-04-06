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

#include <cereal/external/rapidjson/document.h>
#include <sophus/se3.hpp>
#include <Eigen/Core>

#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ark {
namespace datatools {
namespace sensors {

struct CameraProjectionModel {
  enum class ModelType {
    // Kannala and Brandt Like 'Generic' Projection Model
    // http://cs.iupui.edu/~tuceryan/pdf-repository/Kannala2006.pdf
    // https://april.eecs.umich.edu/wiki/Camera_suite
    KannalaBrandtK3,
    Fisheye624,
  };
  ModelType modelName;
  Eigen::VectorXd projectionParams;

  Eigen::Vector2d project(const Eigen::Vector3d& p) const;
  Eigen::Vector3d unproject(const Eigen::Vector2d& uv) const;

  // Return principal point location as {cx, cy}
  Eigen::Vector2d getPrincipalPoint() const;
  // Return focal lengths as {fx, fy}
  Eigen::Vector2d getFocalLengths() const;
};

struct CameraCalibration {
  std::string label;
  CameraProjectionModel projectionModel;
  Sophus::SE3d T_Device_Camera;
  // If the lens is Fisheye and does not cover the entire sensor, we are
  // storing here the radial valid image area.
  // Notes:
  //  - This circle radius is defined from the principal point.
  //  - If radius is < 0, it means you have full sensor data.
  int validRadius = -1;
};

struct LinearRectificationModel {
  Eigen::Matrix3d rectificationMatrix;
  Eigen::Vector3d bias;

  // Compensates the input vector (acceleration for accelerator, or angular
  // velocity for gyroscope) with a linear model
  Eigen::Vector3d compensateForSystematicErrorFromMeasurement(const Eigen::Vector3d& v_raw) const;
  // inverse function of compensateForSystematicErrorFromMeasurement
  Eigen::Vector3d distortWithSystematicError(const Eigen::Vector3d& v_compensated) const;
};

struct ImuCalibration {
  std::string label;
  LinearRectificationModel accel;
  LinearRectificationModel gyro;
  Sophus::SE3d T_Device_Imu;
};

struct MagnetometerCalibration {
  std::string label;
  LinearRectificationModel model;
  Sophus::SE3d T_Device_Magnetometer;
};

struct LinearPressureModel {
  // Linear model is a 1-d linear correction to the barometer pressure readings:
  // corrected_reading = slope * actual_reading + offset.
  // Slope is unitless, and unit of offset is Pa.
  double slope;
  double offsetPa;
};

struct BarometerCalibration {
  std::string label;
  LinearPressureModel pressure;
  Sophus::SE3d T_Device_Barometer;
};

struct MicrophoneCalibration {
  std::string label;
  // Sensitivity difference between this instance and the reference mic at 1kHz,
  // a.k.a. dutMicSen - refMicSen, in unit of dBV.
  double dSensitivity1KDbv;
  Sophus::SE3d T_Device_Microphone;
};

class DeviceModel {
 public:
  static DeviceModel fromJson(const fb_rapidjson::Document& json);
  static DeviceModel fromJson(const std::string& jsonStr);

  std::optional<CameraCalibration> getCameraCalib(const std::string& label) const;
  std::optional<ImuCalibration> getImuCalib(const std::string& label) const;
  std::optional<MagnetometerCalibration> getMagnetometerCalib(const std::string& label) const;
  std::optional<BarometerCalibration> getBarometerCalib(const std::string& label) const;
  std::optional<MicrophoneCalibration> getMicrophoneCalib(const std::string& label) const;

  std::optional<Sophus::SE3d> getCADSensorPose(const std::string& label) const;

  bool tryCropAndScaleCameraCalibration(
      const std::string& label,
      const int nativeResolution,
      const int imageWidth);

  std::vector<std::string> getCameraLabels() const;
  std::vector<std::string> getImuLabels() const;
  std::vector<std::string> getMagnetometerLabels() const;
  std::vector<std::string> getBarometerLabels() const;
  std::vector<std::string> getMicrophoneLabels() const;

  std::string getDeviceSubtype() const;

  Eigen::Vector3d transform(
      const Eigen::Vector3d& p_source,
      const std::string& sourceSensorLabel,
      const std::string& destSensorLabel) const;

 private:
  std::unordered_map<std::string, CameraCalibration> cameraCalibs_;
  std::unordered_map<std::string, ImuCalibration> imuCalibs_;
  std::unordered_map<std::string, MagnetometerCalibration> magnetometerCalibs_;
  std::unordered_map<std::string, BarometerCalibration> barometerCalibs_;
  std::unordered_map<std::string, MicrophoneCalibration> microphoneCalibs_;
  std::unordered_set<std::string> updatedCameraCalibs_;
  std::string deviceSubtype_; // "DVT-L" or "DVT-S"
};

} // namespace sensors
} // namespace datatools
} // namespace ark
