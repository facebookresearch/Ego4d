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

#include "AriaViewer.h"
#include <pangolin/display/image_view.h>
#include <pangolin/gl/glpixformat.h>
#include <pangolin/pangolin.h>
#include "AriaStreamIds.h"

// font defined in pangolin
extern const unsigned char AnonymousPro_ttf[];

namespace ark {
namespace datatools {
namespace visualization {

AriaViewer::AriaViewer(
    datatools::dataprovider::AriaDataProvider* dataProvider,
    int width,
    int height,
    const std::string& name,
    int id)
    : AriaViewerBase(dataProvider, width, height, name + std::to_string(id), id) {}

void AriaViewer::run() {
  std::cout << "Start " << name_ << "!" << std::endl;
  // get a static render mutex across all AriaViewer windows
  static std::unique_ptr<std::mutex> render_mutex(new std::mutex());
  std::mutex* p_render_mutex = render_mutex.get();

  pangolin::CreateWindowAndBind(name_, width_, height_);

  using namespace ark::datatools::dataprovider;

  setDataChanged(false, kSlamLeftCameraStreamId);
  setDataChanged(false, kSlamRightCameraStreamId);
  setDataChanged(false, kRgbCameraStreamId);
  setDataChanged(false, kEyeCameraStreamId);

  auto slamLeftCameraImageWidth = dataProvider_->getImageWidth(kSlamLeftCameraStreamId);
  auto slamLeftCameraImageHeight = dataProvider_->getImageHeight(kSlamLeftCameraStreamId);
  auto slamRightCameraImageWidth = dataProvider_->getImageWidth(kSlamRightCameraStreamId);
  auto slamRightCameraImageHeight = dataProvider_->getImageHeight(kSlamRightCameraStreamId);
  auto rgbCameraImageWidth = dataProvider_->getImageWidth(kRgbCameraStreamId);
  auto rgbCameraImageHeight = dataProvider_->getImageHeight(kRgbCameraStreamId);
  auto eyeCameraImageWidth = dataProvider_->getImageWidth(kEyeCameraStreamId);
  auto eyeCameraImageHeight = dataProvider_->getImageHeight(kEyeCameraStreamId);

  pangolin::ImageView cameraSlamLeftView = pangolin::ImageView(); //"slam left");
  pangolin::ImageView cameraSlamRightView = pangolin::ImageView(); //"slam right");
  pangolin::ImageView cameraRgbView = pangolin::ImageView();
  pangolin::ImageView cameraEyeView = pangolin::ImageView();

  // setup a loggers to show the imu signals
  pangolin::DataLog logAcc;
  logAcc.SetLabels(
      {"accRight_x [m/s2]",
       "accRight_y [m/s2]",
       "accRight_z [m/s2]",
       "accLeft_x [m/s2]",
       "accLeft_y [m/s2]",
       "accLeft_z [m/s2]"});
  pangolin::Plotter plotAcc(&logAcc, 0.0f, 1500.f, -20., 20., 100, 5.);
  plotAcc.Track("$i");
  pangolin::DataLog logGyro;
  logGyro.SetLabels(
      {"gyroRight_x [rad/s]",
       "gyroRight_y [rad/s]",
       "gyroRight_z [rad/s]",
       "gyroLeft_x [rad/s]",
       "gyroLeft_y [rad/s]",
       "gyroLeft_z [rad/s]"});
  pangolin::Plotter plotGyro(&logGyro, 0.0f, 1500.f, -3.14, 3.14, 100, 1.);
  plotGyro.Track("$i");
  // magnetometer
  pangolin::DataLog logMag;
  logMag.SetLabels({"mag_x [Tesla]", "mag_y [Tesla]", "mag_z [Tesla]"});
  pangolin::Plotter plotMag(&logMag, 0.0f, 75.f, -100., 100., 100, 1.);
  plotMag.Track("$i");
  // barometer and temperature logs
  pangolin::DataLog logBaro;
  logBaro.SetLabels({"barometer [Pa]"});
  pangolin::DataLog logTemp;
  logTemp.SetLabels({"temperature [C]"});
  // setup a logger to show the audio signals
  pangolin::DataLog logAudio;
  logAudio.SetLabels({"m0", "m1", "m2", "m3", "m4", "m5", "m6"});
  pangolin::Plotter plotAudio(&logAudio, 0.0f, 3 * 48000.f, -5e-2, 5e-2, 10000, 1e-3f);
  plotAudio.Track("$i");

  auto& container = pangolin::CreateDisplay()
                        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(180), 1.0)
                        .SetLayout(pangolin::LayoutEqual)
                        .AddDisplay(cameraSlamLeftView)
                        .AddDisplay(cameraRgbView)
                        .AddDisplay(cameraSlamRightView)
                        .AddDisplay(plotAcc)
                        .AddDisplay(plotGyro)
                        .AddDisplay(plotMag)
                        .AddDisplay(plotAudio)
                        .AddDisplay(cameraEyeView);

  // prefix to give each viewer its own set of controls (otherwise they are
  // shared if multiple viewers are opened)
  const std::string prefix = "ui" + std::to_string(id_);
  pangolin::CreatePanel(prefix).SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(180));
  // Settings
  pangolin::Var<bool> playButton(prefix + ".Play", isPlaying_, true);
  pangolin::Var<float> playbackSlide(
      prefix + ".playback_speed", playbackSpeedFactor_, 0.1, 10, false);
  pangolin::Var<int> sparsitySlide(prefix + ".camSparsity", 1, 1, 10, false);
#if (PANGOLIN_VERSION_MAJOR == 0) && (PANGOLIN_VERSION_MINOR >= 7)
  pangolin::Var<std::function<void(void)>> save_window(prefix + ".Snapshot UI", [&container]() {
    pangolin::SaveWindowOnRender("snapshot", container.v);
  });
#else
  pangolin::Var<std::function<void(void)>> save_window(
      prefix + ".Snapshot UI", []() { pangolin::SaveWindowOnRender("snapshot"); });
#endif
  pangolin::Var<bool> showLeftCamImg(prefix + ".LeftImg", true, true);
  pangolin::Var<bool> showRightCamImg(prefix + ".RightImg", true, true);
  pangolin::Var<bool> showRgbCamImg(prefix + ".RgbImg", true, true);
  pangolin::Var<bool> showEyeImg(prefix + ".EyeImg", true, true);
  // IMU
  pangolin::Var<bool> showLeftImu(prefix + ".LeftImu", true, true);
  pangolin::Var<bool> showRightImu(prefix + ".RightImu", true, true);
  pangolin::Var<bool> showMagnetometer(prefix + ".Magnetometer", true, true);
  // Audio
  pangolin::Var<bool> showAudio(prefix + ".Audio", true, true);
  // print gps, wifi, bluetooth to terminal output.
  pangolin::Var<bool> printGps(prefix + ".print GPS log", true, true);
  pangolin::Var<bool> printWifi(prefix + ".print Wifi log", true, true);
  pangolin::Var<bool> printBluetooth(prefix + ".print Bluetooth log", true, true);
  // temperature and pressure display on the side of the menu
  pangolin::Var<float> temperatureDisplay(prefix + ".temp [C]", 0., 0.0, 0., false);
  pangolin::Var<float> pressureDisplay(prefix + ".pres [kPa]", 0., 0.0, 0., false);

  // Main loop
  while (!pangolin::ShouldQuit()) {
    isPlaying_ = playButton;
    playbackSpeedFactor_ = playbackSlide;
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    {
      const std::lock_guard<std::mutex> dataLock(dataMutex_);
      // Draw images.
      if (cameraSlamLeftView.IsShown() && isDataChanged(kSlamLeftCameraStreamId)) {
        cameraSlamLeftView.SetImage(
            static_cast<void*>(cameraImageBufferMap_[kSlamLeftCameraStreamId.getTypeId()]
                                                    [kSlamLeftCameraStreamId.getInstanceId()]
                                                        .data()),
            slamLeftCameraImageWidth,
            slamLeftCameraImageHeight,
            slamLeftCameraImageWidth,
            pangolin::PixelFormatFromString("GRAY8"));
        setDataChanged(false, kSlamLeftCameraStreamId);
      }
      if (cameraRgbView.IsShown() && isDataChanged(kRgbCameraStreamId)) {
        cameraRgbView.SetImage(
            static_cast<void*>(cameraImageBufferMap_[kRgbCameraStreamId.getTypeId()]
                                                    [kRgbCameraStreamId.getInstanceId()]
                                                        .data()),
            rgbCameraImageWidth,
            rgbCameraImageHeight,
            rgbCameraImageWidth * 3,
            pangolin::PixelFormatFromString("RGB24"));
        setDataChanged(false, kRgbCameraStreamId);
      }
      if (cameraEyeView.IsShown() && isDataChanged(kEyeCameraStreamId)) {
        cameraEyeView.SetImage(
            static_cast<void*>(cameraImageBufferMap_[kEyeCameraStreamId.getTypeId()]
                                                    [kEyeCameraStreamId.getInstanceId()]
                                                        .data()),
            eyeCameraImageWidth,
            eyeCameraImageHeight,
            eyeCameraImageWidth,
            pangolin::PixelFormatFromString("GRAY8"));
        setDataChanged(false, kEyeCameraStreamId);
      }
      if (cameraSlamRightView.IsShown() && isDataChanged(kSlamRightCameraStreamId)) {
        cameraSlamRightView.SetImage(
            static_cast<void*>(cameraImageBufferMap_[kSlamRightCameraStreamId.getTypeId()]
                                                    [kSlamRightCameraStreamId.getInstanceId()]
                                                        .data()),
            slamRightCameraImageWidth,
            slamRightCameraImageHeight,
            slamRightCameraImageWidth,
            pangolin::PixelFormatFromString("GRAY8"));
        setDataChanged(false, kSlamRightCameraStreamId);
      }
    }

    dataProvider_->setWifiBeaconPlayerVerbose(printWifi);
    dataProvider_->setBluetoothBeaconPlayerVerbose(printBluetooth);
    dataProvider_->setGpsPlayerVerbose(printGps);

    if (isDataChanged(kImuRightStreamId) && isDataChanged(kImuLeftStreamId)) {
      setDataChanged(false, kImuRightStreamId);
      setDataChanged(false, kImuLeftStreamId);
      auto& accRightMSec2 =
          accMSec2Map_[kImuRightStreamId.getTypeId()][kImuRightStreamId.getInstanceId()];
      auto& accLeftMSec2 =
          accMSec2Map_[kImuLeftStreamId.getTypeId()][kImuLeftStreamId.getInstanceId()];
      for (size_t i = 0; i < std::min(accLeftMSec2.size(), accRightMSec2.size()); ++i) {
        std::vector<float> acc(6, std::nanf(""));
        if (showRightImu && i < accRightMSec2.size()) {
          acc[0] = accRightMSec2[i](0);
          acc[1] = accRightMSec2[i](1);
          acc[2] = accRightMSec2[i](2);
        }
        if (showLeftImu && i < accLeftMSec2.size()) {
          acc[3] = accLeftMSec2[i](0);
          acc[4] = accLeftMSec2[i](1);
          acc[5] = accLeftMSec2[i](2);
        }
        logAcc.Log(acc);
      }
      auto& gyroRightRadSec =
          gyroRadSecMap_[kImuRightStreamId.getTypeId()][kImuRightStreamId.getInstanceId()];
      auto& gyroLeftRadSec =
          gyroRadSecMap_[kImuLeftStreamId.getTypeId()][kImuLeftStreamId.getInstanceId()];
      for (size_t i = 0; i < std::min(gyroLeftRadSec.size(), gyroRightRadSec.size()); ++i) {
        std::vector<float> gyro(6, std::nanf(""));
        if (showRightImu && i < gyroRightRadSec.size()) {
          gyro[0] = gyroRightRadSec[i](0);
          gyro[1] = gyroRightRadSec[i](1);
          gyro[2] = gyroRightRadSec[i](2);
        }
        if (showLeftImu && i < gyroLeftRadSec.size()) {
          gyro[3] = gyroLeftRadSec[i](0);
          gyro[4] = gyroLeftRadSec[i](1);
          gyro[5] = gyroLeftRadSec[i](2);
        }
        logGyro.Log(gyro);
      }
    }
    if (isDataChanged(kMagnetometerStreamId)) {
      setDataChanged(false, kMagnetometerStreamId);
      for (const auto& mag : magTesla_) {
        logMag.Log(std::vector<float>{mag[0], mag[1], mag[2]});
      }
    }
    if (isDataChanged(kAudioStreamId)) {
      setDataChanged(false, kAudioStreamId);
      for (const auto& audio : audio_) {
        logAudio.Log(audio);
      }
    }
    if (isDataChanged(kBarometerStreamId)) {
      setDataChanged(false, kBarometerStreamId);
      for (const auto& p : pressure_) {
        logBaro.Log(p);
      }
      for (const auto& temp : temperature_) {
        logTemp.Log(temp);
      }
      pressureDisplay = pressure_.back() * 1e-3; // kPa
      temperatureDisplay = temperature_.back(); // C
    }

    // propagate show parameters
    container[0].Show(showLeftCamImg);
    container[1].Show(showRightCamImg);
    container[2].Show(showRgbCamImg);
    cameraSlamLeftView.Show(showLeftCamImg);
    cameraSlamRightView.Show(showRightCamImg);
    cameraRgbView.Show(showRgbCamImg);
    cameraEyeView.Show(showEyeImg);
    plotAudio.Show(showAudio);
    plotAcc.Show(showLeftImu || showRightImu);
    plotGyro.Show(showLeftImu || showRightImu);

    {
      std::lock_guard<std::mutex> lock(*p_render_mutex);
      pangolin::FinishFrame();
    }
  }
  std::cout << "\nQuit Viewer." << std::endl;
  exit(1);
}

std::pair<double, double> AriaViewer::initDataStreams(
    const std::vector<vrs::StreamId>& kImageStreamIds,
    const std::vector<vrs::StreamId>& kImuStreamIds,
    const std::vector<vrs::StreamId>& kDataStreams) {
  std::unique_lock<std::mutex> dataLock(dataMutex_);

  // Call mother initialization
  const auto speedDataRate =
      AriaViewerBase::initDataStreams(kImageStreamIds, kImuStreamIds, kDataStreams);
  return speedDataRate;
}

} // namespace visualization
} // namespace datatools
} // namespace ark
