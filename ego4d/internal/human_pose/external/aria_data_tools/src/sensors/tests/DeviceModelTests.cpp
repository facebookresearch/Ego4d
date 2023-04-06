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

#include <models/DeviceModel.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sophus/se3.hpp>
#include <Eigen/Eigen>

using namespace ark::datatools::sensors;

const char* kTestCalibStr = R"rawJsonDelimiter({
  "CameraCalibrations": [
    {
      "Label": "camera-slam-left",
      "Projection": {
        "Name": "FisheyeRadTanThinPrism",
        "Params": [
          240, 320, 240, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]
      },
      "T_Device_Camera": {
        "Translation": [0, 0, 0],
        "UnitQuaternion": [
          1,
          [0, 0, 0]
        ]
      }
    },
    {
      "Label": "camera-et-left",
      "Projection": {
        "Name": "KannalaBrandtK3",
        "Params": [
          552, 552, 320, 240, 0, 0, 0, 0
        ]
      },
      "T_Device_Camera": {
        "Translation": [1, 2, 3],
        "UnitQuaternion": [
          1,
          [0, 0, 0]
        ]
      }
    }
  ],
  "ImuCalibrations": [
    {
      "Accelerometer": {
        "Bias": {
          "Name": "Constant",
          "Offset": [
            0.1, 0.2, 0.3
          ]
        },
        "Model": {
          "Name": "Linear",
          "RectificationMatrix": [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
          ]
        }
      },
      "Gyroscope": {
        "Bias": {
          "Name": "Constant",
          "Offset": [
            -0.3, -0.2, -0.1
          ]
        },
        "Model": {
          "Name": "Linear",
          "RectificationMatrix": [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
          ]
        }
      },
      "Label": "imu-left",
      "T_Device_Imu": {
        "Translation": [-1, -2, -3],
        "UnitQuaternion": [
          1,
          [0, 0, 0]
        ]
      }
    }
  ],
  "DeviceClassInfo": {
    "BuildVersion": "DVT-S",
    "DeviceClass": "Aria"
  }
})rawJsonDelimiter";

TEST(DeviceModelTest, ParseJsonAndProjectCamera) {
  DeviceModel model = DeviceModel::fromJson(std::string(kTestCalibStr));
  ASSERT_THAT(model.getCameraLabels(), testing::ElementsAre("camera-et-left", "camera-slam-left"));

  // Test FisheyeRadTanThinPrism projection
  const auto slamLeft = model.getCameraCalib("camera-slam-left").value();
  Eigen::Vector3d p_slamLeft{2.0, 3.0, 1.0};
  Eigen::Vector2d uv_slamLeft = slamLeft.projectionModel.project(p_slamLeft);
  Eigen::Vector2d uv_slamLeft_actual{493.09928578162271, 499.64892867243407};
  EXPECT_TRUE((uv_slamLeft - uv_slamLeft_actual).norm() < 1e-6);

  Eigen::Vector3d p_slamLeft_convertBack = slamLeft.projectionModel.unproject(uv_slamLeft);
  EXPECT_TRUE((p_slamLeft_convertBack - p_slamLeft).norm() < 1e-6);

  // Test KB3 projection
  const auto etLeft = model.getCameraCalib("camera-et-left").value();
  Eigen::Vector3d p_etLeft{-1.0, -2.0, 1.0};
  Eigen::Vector2d uv_etLeft = etLeft.projectionModel.project(p_etLeft);
  Eigen::Vector2d uv_etLeft_actual{36.044133853218739, -327.91173229356252};
  EXPECT_TRUE((uv_etLeft - uv_etLeft_actual).norm() < 1e-6);

  Eigen::Vector3d p_etLeft_convertBack = etLeft.projectionModel.unproject(uv_etLeft);
  EXPECT_TRUE((p_etLeft_convertBack - p_etLeft).norm() < 1e-6);
}

TEST(DeviceModelTest, ParseJsonAndTransformBetweenSensors) {
  DeviceModel model = DeviceModel::fromJson(std::string(kTestCalibStr));

  Eigen::Vector3d p_slamLeft{0.1, 0.2, 0.3};
  Eigen::Vector3d p_imuLeft = model.transform(p_slamLeft, "camera-slam-left", "imu-left");
  Eigen::Vector3d p_imuLeft_actual{1.1, 2.2, 3.3};
  EXPECT_TRUE((p_imuLeft - p_imuLeft_actual).norm() < 1e-6);

  Eigen::Vector3d p_slamLeft_convertBack =
      model.transform(p_imuLeft, "imu-left", "camera-slam-left");
  EXPECT_TRUE((p_slamLeft_convertBack - p_slamLeft).norm() < 1e-6);
}

TEST(DeviceModelTest, ParseJsonAndRectifyImu) {
  DeviceModel model = DeviceModel::fromJson(std::string(kTestCalibStr));

  ASSERT_THAT(model.getImuLabels(), testing::ElementsAre("imu-left"));
  const auto imuLeft = model.getImuCalib("imu-left").value();

  Eigen::Vector3d p_imuLeft{1.0, 2.0, 3.0};
  Eigen::Vector3d p_imuLeft_gyroRectified =
      imuLeft.gyro.compensateForSystematicErrorFromMeasurement(p_imuLeft);
  Eigen::Vector3d p_imuLeft_gyroRectified_actual{1.3, 2.2, 3.1};
  EXPECT_TRUE((p_imuLeft_gyroRectified - p_imuLeft_gyroRectified_actual).norm() < 1e-6);

  Eigen::Vector3d p_imuLeft_accelRectified =
      imuLeft.accel.compensateForSystematicErrorFromMeasurement(p_imuLeft);
  Eigen::Vector3d p_imuLeft_accelRectified_actual{0.9, 1.8, 2.7};
  EXPECT_TRUE((p_imuLeft_accelRectified - p_imuLeft_accelRectified_actual).norm() < 1e-6);
}

const char* kTestOnlineCalibStr = R"rawJsonDelimiter({
  "CameraCalibrations": "[{'Calibrated': True, 'T_Device_Camera': {'Translation': [2.7442755533291227e-11, -2.0700694630670917e-11, 3.038745990946445e-11], 'UnitQuaternion': [1, [5.465220653444834e-19, -5.293380953971809e-19, 2.5680847627109642e-17]]}, 'SerialNumber': '', 'Projection': {'Params': [241.3652801513672, 319.0837097167969, 239.8876495361328, -0.025480309501290002, 0.09820848703384401, -0.06668509542942, 0.009074256755411, 0.002338790334761, -0.000557322637178, 0.000256432715104, -7.223937245000001e-06, -0.00047156144864800005, -5.8604848163e-05, -0.000441436044638, -0.00013711846258900002], 'Description': 'see FisheyeRadTanThinPrism.h', 'Name': 'FisheyeRadTanThinPrism'}, 'Label': 'camera-slam-left'}, {'Calibrated': True, 'T_Device_Camera': {'Translation': [0.004482182841862001, -0.108131690430966, -0.08500801742958301], 'UnitQuaternion': [0.7886234982626421, [0.61403683489882, 0.000135906869031, 0.032120474852546]]}, 'SerialNumber': '', 'Projection': {'Params': [242.75344848632812, 320.8656005859375, 239.8474884033203, -0.025075752288103003, 0.095797672867774, -0.062207240611314, 0.004991019144654001, 0.004106123931705, -0.000840131309814, 0.0009432031656610001, -0.0018507961649440001, -0.001268782070837, 8.856149179e-06, 0.0031910687685010003, -7.439093315e-05], 'Description': 'see FisheyeRadTanThinPrism.h', 'Name': 'FisheyeRadTanThinPrism'}, 'Label': 'camera-slam-right'}, {'Calibrated': True, 'T_Device_Camera': {'Translation': [-0.004944041676207001, -0.012380072847971001, -0.005207711439804001], 'UnitQuaternion': [0.9421878577571071, [0.33122444610891805, 0.040283696470263, 0.030816080114211]]}, 'SerialNumber': '', 'Projection': {'Params': [1223.431008735967, 1459.943864799954, 1449.616710075279, 0.39886307891167805, -0.417100458966277, -0.10906924050385601, 1.5913716628564991, -2.029597437093853, 0.738166502212516, 0.000688098578876, -0.000494212153263, -0.00036830936309300004, 0.000122299996715, 0.0007339957811940001, -1.3312766379e-05], 'Description': 'see FisheyeRadTanThinPrism.h', 'Name': 'FisheyeRadTanThinPrism'}, 'Label': 'camera-rgb'}]",
  "ImuCalibrations": "[{'Calibrated': True, 'T_Device_Imu': {'Translation': [0.00037632955352800003, -0.00034075006331600003, -0.006808482085253], 'UnitQuaternion': [0.025173180888057003, [-0.699717743094947, -0.713130455510509, 0.034732468749922]]}, 'SerialNumber': '', 'Gyroscope': {'TimeOffsetSec_Device_Gyro': 0.001823980361223, 'Bias': {'Name': 'Constant', 'Offset': [-0.021546694558283, 1.3278324737000002e-05, 0.005974857889363001]}, 'Model': {'Name': 'LinearGSensitivity', 'RectificationMatrix': [[1.01220691204071, -0.001971792196854, -0.015941612422466], [0.000579295388888, 0.9958923459053041, -0.007231364957988], [-0.010747583582997001, 0.008998625911772001, 0.9768686294555661]], 'GSensitivityMatrix': [[0, 0, 0], [0, 0, 0], [0, 0, 0]]}}, 'Accelerometer': {'Bias': {'Name': 'Constant', 'Offset': [0.25904281294791603, 0.012216238275613, 0.129008279649234]}, 'TimeOffsetSec_Device_Accel': 0.0012179089244450001, 'Model': {'Name': 'UpperTriagonalLinear', 'RectificationMatrix': [[0.9908447265625, -0.0006875188555560001, 0.00164186861366], [0, 0.9923104047775261, -0.005662510171532], [0, 0, 1.002624273300171]]}}, 'Label': 'imu-left'}, {'Calibrated': True, 'T_Device_Imu': {'Translation': [0.0050472652867060005, -0.102036279192868, -0.08708896155690801], 'UnitQuaternion': [0.609973967667058, [-0.7860518581501571, -0.08168563505661801, 0.05815231803280001]]}, 'SerialNumber': '', 'Gyroscope': {'TimeOffsetSec_Device_Gyro': 0.004136259667575, 'Bias': {'Name': 'Constant', 'Offset': [0.004183419319852001, 0.000199526010658, 0.001053268716896]}, 'Model': {'Name': 'LinearGSensitivity', 'RectificationMatrix': [[1.003318905830383, -0.0034844430629160004, -0.000642404309473], [0.0036751888692370003, 0.9956092238426201, 0.00031748012406700003], [0.002127291634678, -0.0031102816574270003, 1.002527713775634]], 'GSensitivityMatrix': [[0, 0, 0], [0, 0, 0], [0, 0, 0]]}}, 'Accelerometer': {'Bias': {'Name': 'Constant', 'Offset': [-0.032921547323447, -0.027796719379716003, -0.017245245532147003]}, 'TimeOffsetSec_Device_Accel': 0.003047367325052, 'Model': {'Name': 'UpperTriagonalLinear', 'RectificationMatrix': [[0.9995805621147151, -0.0008915517828420001, 0.00038420167402300004], [0, 1.002301216125488, -0.0012162600178270001], [0, 0, 1.000445246696472]]}}, 'Label': 'imu-right'}]",
  "tracking_timestamp_us": "1357695322",
  "utc_timestamp_ns": "1669844355099598323"
})rawJsonDelimiter";

TEST(DeviceModelTest, ParseOnlineCalib) {
  const DeviceModel model = DeviceModel::fromJson(std::string(kTestOnlineCalibStr));
  EXPECT_EQ(model.getCameraLabels().size(), 3);
  EXPECT_EQ(model.getImuLabels().size(), 2);
}
