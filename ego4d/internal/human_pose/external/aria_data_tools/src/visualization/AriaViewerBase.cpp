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

#include "AriaViewerBase.h"
#include "AriaStreamIds.h"

namespace ark {
namespace datatools {
namespace visualization {

AriaViewerBase::AriaViewerBase(
    datatools::dataprovider::AriaDataProvider* dataProvider,
    int width,
    int height,
    const std::string& name,
    int id)
    : width_(width),
      height_(height),
      name_(name + std::to_string(id)),
      id_(id),
      dataProvider_(dataProvider) {}

std::pair<double, double> AriaViewerBase::initDataStreams(
    const std::vector<vrs::StreamId>& kImageStreamIds,
    const std::vector<vrs::StreamId>& kImuStreamIds,
    const std::vector<vrs::StreamId>& kDataStreams) {
  imageStreamIds_ = kImageStreamIds;
  imuStreamIds_ = kImuStreamIds;
  dataStreams_ = kDataStreams;

  auto vrsDataProvider =
      dynamic_cast<ark::datatools::dataprovider::AriaVrsDataProvider*>(dataProvider_);

  // Streams should be set after opening VRS file in AriaVrsDataProvider
  bool vrsContainsImageStream = false;
  for (auto& streamId : imageStreamIds_) {
    if (vrsDataProvider && vrsDataProvider->getStreamsInFile().count(streamId)) {
      dataProvider_->setStreamPlayer(streamId);
      vrsContainsImageStream = true;
    }
  }
  double fastestNominalRateHz = 0;
  double currentTimestampSec = 0;
  if (vrsContainsImageStream) {
    fastestNominalRateHz = dataProvider_->getFastestNominalRateHz();
    currentTimestampSec = dataProvider_->getFirstTimestampSec();
  }

  // Setup the other datastreams; this is done after the image streams and
  // getting their fastestNominalRate.
  for (auto& streamId : imuStreamIds_) {
    if (vrsDataProvider && vrsDataProvider->getStreamsInFile().count(streamId)) {
      dataProvider_->setStreamPlayer(streamId);
      if (vrsDataProvider) {
        vrsDataProvider->readFirstConfigurationRecord(streamId);
      }
    }
  }
  if (!vrsContainsImageStream) {
    fastestNominalRateHz = dataProvider_->getFastestNominalRateHz();
    currentTimestampSec = dataProvider_->getFirstTimestampSec();
  }

  for (auto& streamId : dataStreams_) {
    if (vrsDataProvider && vrsDataProvider->getStreamsInFile().count(streamId)) {
      dataProvider_->setStreamPlayer(streamId);
      if (vrsDataProvider) {
        vrsDataProvider->readFirstConfigurationRecord(streamId);
      }
    }
  }

  // Safe to load device model now for both provider modes, VRS configuration records were read
  dataProvider_->loadDeviceModel();
  // init device model (intrinsic and extrinsic calibration)
  deviceModel_ = dataProvider_->getDeviceModel();

  return {currentTimestampSec, fastestNominalRateHz};
}

bool AriaViewerBase::readData(double currentTimestampSec) {
  if (isPlaying()) {
    {
      std::unique_lock<std::mutex> dataLock(dataMutex_);
      auto vrsDataProvider =
          dynamic_cast<ark::datatools::dataprovider::AriaVrsDataProvider*>(dataProvider_);

      // Handle image streams
      for (auto& streamId : imageStreamIds_) {
        if (vrsDataProvider && vrsDataProvider->getStreamsInFile().count(streamId)) {
          if (dataProvider_->tryFetchNextData(streamId, currentTimestampSec)) {
            setDataChanged(true, streamId);
            auto imageBufferVector = dataProvider_->getImageBufferVector(streamId);
            if (imageBufferVector) {
              setCameraImageBuffer(*imageBufferVector, streamId);
            }
          }
        }
      }
      // Handle left and right imu streams
      for (auto& streamId : imuStreamIds_) {
        if (vrsDataProvider && vrsDataProvider->getStreamsInFile().count(streamId)) {
          std::vector<Eigen::Vector3f> accMSec2, gyroRadSec;
          while (dataProvider_->tryFetchNextData(streamId, currentTimestampSec)) {
            accMSec2.push_back(dataProvider_->getMotionAccelData(streamId));
            gyroRadSec.push_back(dataProvider_->getMotionGyroData(streamId));
          }
          setImuDataChunk(streamId, accMSec2, gyroRadSec);
        }
      }
      // handle magnetometer stream
      if (std::find(
              dataStreams_.begin(), dataStreams_.end(), dataprovider::kMagnetometerStreamId) !=
          dataStreams_.end()) {
        std::vector<Eigen::Vector3f> magTesla;
        if (vrsDataProvider &&
            vrsDataProvider->getStreamsInFile().count(dataprovider::kMagnetometerStreamId)) {
          while (dataProvider_->tryFetchNextData(
              dataprovider::kMagnetometerStreamId, currentTimestampSec)) {
            auto magnetometerData = dataProvider_->getMagnetometerData();
            magTesla.emplace_back(magnetometerData[0], magnetometerData[1], magnetometerData[2]);
          }
        }
        setMagnetometerChunk(dataprovider::kMagnetometerStreamId, magTesla);
      }

      // handle barometer stream
      if (std::find(dataStreams_.begin(), dataStreams_.end(), dataprovider::kBarometerStreamId) !=
          dataStreams_.end()) {
        std::vector<float> temperature, pressure;
        if (vrsDataProvider &&
            vrsDataProvider->getStreamsInFile().count(dataprovider::kBarometerStreamId)) {
          while (dataProvider_->tryFetchNextData(
              dataprovider::kBarometerStreamId, currentTimestampSec)) {
            temperature.emplace_back(dataProvider_->getBarometerTemperature());
            pressure.emplace_back(dataProvider_->getBarometerPressure());
          }
        }
        setBarometerChunk(dataprovider::kBarometerStreamId, temperature, pressure);
      }

      // handle audio stream
      if (std::find(dataStreams_.begin(), dataStreams_.end(), dataprovider::kAudioStreamId) !=
          dataStreams_.end()) {
        std::vector<std::vector<float>> audio;
        if (vrsDataProvider &&
            vrsDataProvider->getStreamsInFile().count(dataprovider::kAudioStreamId)) {
          while (
              dataProvider_->tryFetchNextData(dataprovider::kAudioStreamId, currentTimestampSec)) {
            auto audioStreamData = dataProvider_->getAudioData();
            if (audioStreamData) {
              // get the audio data chunk
              const auto& audioData = audioStreamData->get();
              // Get property of the local data chunk
              const size_t C = dataProvider_->getAudioNumChannels();
              const auto N = audioData.size() / C;
              assert(audioData.size() % C == 0);
              for (size_t i = 0; i < N; ++i) {
                audio.emplace_back();
                for (size_t c = 0; c < C; ++c) {
                  // Audio samples are 32bit; convert to float for visualization
                  audio.back().emplace_back(
                      (float)(audioData[i * C + c] / (double)std::numeric_limits<int32_t>::max()));
                }
              }
            }
          }
        }
        setAudioChunk(dataprovider::kAudioStreamId, audio);
      }
      // Make sure we fetch next data for wifi, bluetooth, and gps so that
      // callbacks can printout sensor information to the terminal.
      const std::array<const vrs::StreamId, 4> callbackStreamIds = {
          dataprovider::kWifiStreamId,
          dataprovider::kBluetoothStreamId,
          dataprovider::kGpsStreamId};
      for (const auto& streamId : callbackStreamIds) {
        if (std::find(dataStreams_.begin(), dataStreams_.end(), streamId) != dataStreams_.end()) {
          if (vrsDataProvider && vrsDataProvider->getStreamsInFile().count(streamId)) {
            while (dataProvider_->tryFetchNextData(streamId, currentTimestampSec)) {
            }
          }
        }
      }
    }
    return true;
  }
  return false;
}

} // namespace visualization
} // namespace datatools
} // namespace ark
