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

#include "PilotDatasetViewer.h"

#include <pangolin/display/image_view.h>
#include <pangolin/gl/glpixformat.h>
#include <pangolin/pangolin.h>
#include "AriaStreamIds.h"

// font defined in pangolin
extern const unsigned char AnonymousPro_ttf[];

namespace ark {
namespace datatools {
namespace visualization {

PilotDatasetViewer::PilotDatasetViewer(
    dataprovider::PilotDatasetProvider* dataProvider,
    int width,
    int height,
    const std::string& name,
    int id)
    : AriaViewerBase(dataProvider, width, height, name + std::to_string(id), id),
      dataProvider_(dataProvider) {}

void PilotDatasetViewer::run() {
  std::cout << "Start " << name_ << "!" << std::endl;
  // get a static render mutex across all PilotDatasetViewer windows
  static std::unique_ptr<std::mutex> render_mutex(new std::mutex());
  std::mutex* p_render_mutex = render_mutex.get();

  pangolin::CreateWindowAndBind(name_, width_, height_);

  // Configure initial 3D viewport and projection
  pangolin::OpenGlRenderState Visualization3D_camera(
      pangolin::ProjectionMatrix(
          width_ / 2, height_ / 2, width_ / 2, height_ / 2, width_ / 4, height_ / 4, 0.1, 100),
      pangolin::ModelViewLookAt(2.0, 2.0, 1.0, 0, 0, 0, pangolin::AxisZ));
  auto* handler = new pangolin::Handler3D(Visualization3D_camera);
  pangolin::OpenGlMatrix Twc;
  Twc.SetIdentity();
  pangolin::View& camTraj = pangolin::Display("camTrajectory").SetAspect(640 / (float)640);
  camTraj.SetHandler(handler);

  using namespace ark::datatools::dataprovider;

  setDataChanged(false, kSlamLeftCameraStreamId);
  setDataChanged(false, kSlamRightCameraStreamId);
  setDataChanged(false, kRgbCameraStreamId);

  auto slamLeftCameraImageWidth = dataProvider_->getImageWidth(kSlamLeftCameraStreamId);
  auto slamLeftCameraImageHeight = dataProvider_->getImageHeight(kSlamLeftCameraStreamId);
  auto slamRightCameraImageWidth = dataProvider_->getImageWidth(kSlamRightCameraStreamId);
  auto slamRightCameraImageHeight = dataProvider_->getImageHeight(kSlamRightCameraStreamId);
  auto rgbCameraImageWidth = dataProvider_->getImageWidth(kRgbCameraStreamId);
  auto rgbCameraImageHeight = dataProvider_->getImageHeight(kRgbCameraStreamId);

  pangolin::ImageView cameraSlamLeftView = pangolin::ImageView(); //"slam left");
  pangolin::ImageView cameraSlamRightView = pangolin::ImageView(); //"slam right");
  pangolin::ImageView cameraRgbView = pangolin::ImageView();

  auto& container = pangolin::CreateDisplay()
                        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(180), 1.0)
                        .SetLayout(pangolin::LayoutEqual)
                        .AddDisplay(cameraSlamLeftView)
                        .AddDisplay(cameraRgbView)
                        .AddDisplay(cameraSlamRightView)
                        .AddDisplay(camTraj);

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
  // 3D visualization
  pangolin::Var<bool> showLeftCam3D(prefix + ".LeftRigCam", true, true);
  pangolin::Var<bool> showRightCam3D(prefix + ".RightRigCam", true, true);
  pangolin::Var<bool> showRgbCam3D(prefix + ".RgbRigCam", true, true);
  pangolin::Var<bool> showRig3D(prefix + ".Rig", true, true);
  pangolin::Var<bool> showTraj(prefix + ".Trajectory", true, true);
  pangolin::Var<bool> showWorldCoordinateSystem(prefix + ".World Coord.", true, true);

  // enable blending to allow showing the text-to-speech overlay
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  pangolin::GlFont glFontSpeechToText(AnonymousPro_ttf, 24);

  cameraRgbView.extern_draw_function = [this,
                                        rgbCameraImageWidth,
                                        rgbCameraImageHeight,
                                        &glFontSpeechToText](pangolin::View& v) {
    v.ActivatePixelOrthographic();
    v.ActivateAndScissor();

    const auto speech = dataProvider_->getSpeechToText();
    if (speech) {
      std::stringstream text;
      // show formatted as "text (confidence, duration)"
      text << speech.value().text << " (" << std::fixed << std::setprecision(0)
           << 100 * speech.value().confidence << "%, " << std::fixed << std::setw(5)
           << std::setprecision(3) << speech.value().duration_s() << "s)";
      glFontSpeechToText.Text(text.str()).DrawWindow(v.v.l, v.v.t() - glFontSpeechToText.Height());
    }
    glLineWidth(3);
    glColor3f(1.0, 0.0, 0.0);
    const auto eyeGaze = dataProvider_->getEyetracksOnRgbImage();
    if (eyeGaze) {
      // scale to the current display size
      const float scale = (float)v.v.w / (float)rgbCameraImageWidth;
      pangolin::glDrawCross(
          scale * eyeGaze.value().x(), scale * (rgbCameraImageHeight - eyeGaze.value().y()), 10);
    }
    v.GetBounds().DisableScissor();
  };

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
        // Update the pose of the device
        setPose(dataProvider_->getPose());
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

      if (!T_World_Camera_.empty()) {
        // draw 3D
        camTraj.Activate(Visualization3D_camera);
        if (showWorldCoordinateSystem) {
          // draw origin of world coordinate system
          glLineWidth(3);
          pangolin::glDrawAxis(0.3);
        }
        if (showTraj) {
          drawTraj();
        }
        drawRigs(showRig3D, showLeftCam3D, showRightCam3D, showRgbCam3D, sparsitySlide.Get());
      }
    }

    // propagate show parameters
    container[0].Show(showLeftCamImg);
    container[1].Show(showRightCamImg);
    container[2].Show(showRgbCamImg);
    cameraSlamLeftView.Show(showLeftCamImg);
    cameraSlamRightView.Show(showRightCamImg);
    cameraRgbView.Show(showRgbCamImg);

    {
      std::lock_guard<std::mutex> lock(*p_render_mutex);
      pangolin::FinishFrame();
    }
  }
  std::cout << "\nQuit Viewer." << std::endl;
  exit(1);
}

std::pair<double, double> PilotDatasetViewer::initDataStreams(
    const std::vector<vrs::StreamId>& kImageStreamIds,
    const std::vector<vrs::StreamId>&,
    const std::vector<vrs::StreamId>&) {
  std::unique_lock<std::mutex> dataLock(dataMutex_);

  // Call mother initialization
  const auto speedDataRate = AriaViewerBase::initDataStreams(kImageStreamIds, {}, {});

  using namespace ark::datatools::dataprovider;
  // Deal with specifics to this implementation
  //
  // init transformation from camera coordinate systems to the pose coordinate system (imuLeft).
  if (deviceModel_.getImuCalib("imu-left") && deviceModel_.getCameraCalib("camera-slam-left") &&
      deviceModel_.getImuCalib("camera-rgb")) {
    auto T_ImuLeft_Device = deviceModel_.getImuCalib("imu-left")->T_Device_Imu.inverse();
    T_ImuLeft_cameraMap_[kSlamLeftCameraStreamId.getTypeId()]
                        [kSlamLeftCameraStreamId.getInstanceId()] = T_ImuLeft_Device *
        deviceModel_.getCameraCalib("camera-slam-left")->T_Device_Camera;
    T_ImuLeft_cameraMap_[kRgbCameraStreamId.getTypeId()][kRgbCameraStreamId.getInstanceId()] =
        T_ImuLeft_Device * deviceModel_.getCameraCalib("camera-rgb")->T_Device_Camera;
    T_ImuLeft_cameraMap_[kSlamRightCameraStreamId.getTypeId()]
                        [kSlamRightCameraStreamId.getInstanceId()] = T_ImuLeft_Device *
        deviceModel_.getCameraCalib("camera-slam-right")->T_Device_Camera;
  }

  return speedDataRate;
}

void PilotDatasetViewer::setPose(const std::optional<Sophus::SE3d>& T_World_ImuLeft) {
  if (!T_World_ImuLeft) {
    return;
  }
  if (T_World_Camera_.empty()) {
    // Use first pose translation to define the world frame.
    // Rotation is gravity aligned so we leave it as is.
    T_Viewer_World_ = Sophus::SE3d(Sophus::SO3d(), -T_World_ImuLeft.value().translation());
  }

  // Set pose based on slam-left-camera timestamps.
  if (isDataChanged(dataprovider::kSlamLeftCameraStreamId)) {
    T_World_Camera_.emplace_back(T_Viewer_World_ * T_World_ImuLeft.value());
  }
}

void PilotDatasetViewer::drawTraj() {
  glColor3f(1, 0.8, 0);
  glLineWidth(3);
  std::vector<Eigen::Vector3d> trajectory;
  trajectory.reserve(T_World_Camera_.size());
  for (auto const& T_wi : T_World_Camera_) {
    trajectory.emplace_back(T_wi.translation());
  }
  pangolin::glDrawLineStrip(trajectory);
}

void PilotDatasetViewer::drawRigs(
    bool showRig3D,
    bool showLeftCam3D,
    bool showRightCam3D,
    bool showRgbCam3D,
    int camSparsity) {
  const float sz = 0.03;

  using namespace ark::datatools::dataprovider;
  glLineWidth(3);
  int counter = 0;
  while (counter < T_World_Camera_.size()) {
    auto const& T_World_ImuLeft = T_World_Camera_[counter];
    const auto T_World_CamSlamLeft = T_World_ImuLeft *
        T_ImuLeft_cameraMap_[kSlamLeftCameraStreamId.getTypeId()]
                            [kSlamLeftCameraStreamId.getInstanceId()];
    const auto T_World_CamSlamRight = T_World_ImuLeft *
        T_ImuLeft_cameraMap_[kSlamRightCameraStreamId.getTypeId()]
                            [kSlamRightCameraStreamId.getInstanceId()];
    const auto T_World_CamRgb = T_World_ImuLeft *
        T_ImuLeft_cameraMap_[kRgbCameraStreamId.getTypeId()][kRgbCameraStreamId.getInstanceId()];

    if (counter == T_World_Camera_.size() - 1) {
      glColor3f(0.0, 1.0, 0.0);
    } else {
      glColor3f(0.0, 0.0, 1.0);
    }

    if (showRig3D) {
      // Rig
      pangolin::glDrawAxis(T_World_ImuLeft.matrix(), sz / 2);
    }
    if (showLeftCam3D) {
      auto camSlamLeft = deviceModel_.getCameraCalib("camera-slam-left");
      if (camSlamLeft) {
        auto fs = camSlamLeft->projectionModel.getFocalLengths();
        auto cs = camSlamLeft->projectionModel.getPrincipalPoint();
        // Left cam
        pangolin::glSetFrameOfReference(T_World_CamSlamLeft.matrix());
        pangolin::glDrawFrustum(
            -cs(0) / fs(0), -cs(1) / fs(1), 1. / fs(0), 1. / fs(1), 640, 480, sz * 0.8);
        pangolin::glUnsetFrameOfReference();
      }
    }

    if (showRightCam3D) {
      auto camSlamRight = deviceModel_.getCameraCalib("camera-slam-right");
      if (camSlamRight) {
        auto fs = camSlamRight->projectionModel.getFocalLengths();
        auto cs = camSlamRight->projectionModel.getPrincipalPoint();
        // Right cam
        pangolin::glSetFrameOfReference(T_World_CamSlamRight.matrix());
        pangolin::glDrawFrustum(
            -cs(0) / fs(0), -cs(1) / fs(1), 1. / fs(0), 1. / fs(1), 640, 480, sz * 0.8);
        pangolin::glUnsetFrameOfReference();
      }
    }

    if (showRgbCam3D) {
      auto camRgb = deviceModel_.getCameraCalib("camera-rgb");
      if (camRgb) {
        auto fs = camRgb->projectionModel.getFocalLengths();
        auto cs = camRgb->projectionModel.getPrincipalPoint();
        // Rgb cam
        pangolin::glSetFrameOfReference(T_World_CamRgb.matrix());
        pangolin::glDrawFrustum(
            -cs(0) / fs(0), -cs(1) / fs(1), 1. / fs(0), 1. / fs(1), 1408, 1408, sz * 0.8);
        pangolin::glUnsetFrameOfReference();
      }
    }

    // draw line connecting rig coordinate frames
    pangolin::glDrawLineStrip(std::vector<Eigen::Vector3d>{
        T_World_CamSlamLeft.translation(),
        T_World_CamRgb.translation(),
        T_World_ImuLeft.translation(),
        T_World_CamSlamRight.translation(),
    });

    // Always draw the latest camera.
    if (counter != T_World_Camera_.size() - 1 && counter + camSparsity >= T_World_Camera_.size()) {
      counter = T_World_Camera_.size() - 1;
    } else {
      counter += camSparsity;
    }
  }
}

} // namespace visualization
} // namespace datatools
} // namespace ark
