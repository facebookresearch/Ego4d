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
#include <pangolin/gl/gldraw.h>
#include <pangolin/pangolin.h>
#include "AriaStreamIds.h"
#include "mps_io/eyeGazeReader.h"

namespace {

using namespace ark::datatools::dataprovider;

const std::vector<vrs::StreamId> kImageStreamIds = {
    kEyeCameraStreamId,
    kRgbCameraStreamId,
    kSlamLeftCameraStreamId,
    kSlamRightCameraStreamId};

} // namespace

namespace ark::datatools::visualization {

AriaViewer::AriaViewer(
    datatools::dataprovider::AriaDataProvider* dataProvider,
    int width,
    int height,
    const std::string& eyeTrackingFilepath,
    const std::string& name)
    : AriaViewerBase(dataProvider, width, height, name) {
  // Initialize EyeGaze data
  {
    const TemporalEyeGazeData eyeGaze_records = readEyeGaze(eyeTrackingFilepath);
    // Convert eyeGaze data to a simpler convention
    for (const auto& eyeGazeData : eyeGaze_records) {
      eyeGazeData_[eyeGazeData.first] =
          std::make_pair(eyeGazeData.second.gaze_vector, eyeGazeData.second.uncertainty);
    }
  }
}

void AriaViewer::run() {
  std::cout << "Start " << name_ << "!" << std::endl;
  // get a static render mutex across all AriaViewer windows
  static std::unique_ptr<std::mutex> render_mutex(new std::mutex());
  std::mutex* p_render_mutex = render_mutex.get();

  pangolin::CreateWindowAndBind(name_, width_, height_);
  pangolin::OpenGlMatrix Twc;
  Twc.SetIdentity();

  using namespace ark::datatools::dataprovider;
  //
  // Initialize EyeTracking data visualization widgets
  // - Images: Left & Right eye
  // - Time series: EyeGaze yaw, pitch, uncertainty
  // - Radar: EyeGaze yaw, pitch and history (visualize direction change in time)
  // - EyeGaze bearing vector: visualize eye gaze vector projected in RGB and/or SLAM images
  //
  // Initialize for the EyeTracking camera view
  setDataChanged(false, kEyeCameraStreamId);
  auto etCameraImageWidth = dataProvider_->getImageWidth(kEyeCameraStreamId);
  auto etCameraImageHeight = dataProvider_->getImageHeight(kEyeCameraStreamId);
  pangolin::ImageView cameraETView = pangolin::ImageView();

  // Initialize for the RGB camera view
  setDataChanged(false, kRgbCameraStreamId);
  pangolin::ImageView cameraRGBView = pangolin::ImageView();

  // Initialize for the SLAM left and right camera view
  setDataChanged(false, kSlamLeftCameraStreamId);
  pangolin::ImageView cameraSlamLView = pangolin::ImageView();
  setDataChanged(false, kSlamRightCameraStreamId);
  pangolin::ImageView cameraSlamRView = pangolin::ImageView();

  // Time Series setup for {yaw, pitch, uncertainty}
  pangolin::DataLog logEyeGaze;
  logEyeGaze.SetLabels({"eyeGaze_yaw [rad]", "eyeGaze_pitch [rad]", "uncertainty"});
  pangolin::Plotter plotEyeGaze(&logEyeGaze, 0.0f, 200.f, -3.14, 3.14, 50, 1.);
  plotEyeGaze.Track("$i");

  // 2D Radar View setup (setup an orthogonal camera viewpoint)
  pangolin::View& eyeGazeRadar = pangolin::Display("eyeGazeRadar").SetAspect(1.0);
  pangolin::OpenGlRenderState radar_view_camera(
      pangolin::ProjectionMatrixOrthographic(-2.14, 2.14, -2.14, 2.14, -1.0, 1.0));
  auto* handler = new pangolin::Handler3D(radar_view_camera);
  eyeGazeRadar.SetHandler(handler);

  // Arrange the views in layout
  auto& container = pangolin::CreateDisplay()
                        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(180), 1.0)
                        .SetLayout(pangolin::LayoutEqual)
                        .AddDisplay(cameraETView)
                        .AddDisplay(plotEyeGaze)
                        .AddDisplay(eyeGazeRadar)
                        .AddDisplay(cameraRGBView)
                        .AddDisplay(cameraSlamLView)
                        .AddDisplay(cameraSlamRView);

  // prefix to give each viewer its own set of controls
  const std::string prefix = "ui";
  pangolin::CreatePanel(prefix).SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(180));
  // Settings
  pangolin::Var<bool> playButton(prefix + ".Play", isPlaying_, true);
  pangolin::Var<float> playbackSlide(
      prefix + ".playback_speed", playbackSpeedFactor_, 0.1, 10, false);
  // Widget layout (On/Off)
  pangolin::Var<bool> showETCamImg(prefix + ".ET Img", true, true);
  pangolin::Var<bool> showETTemporal(prefix + ".EyeGaze Temporal", true, true);
  pangolin::Var<bool> showETRadar(prefix + ".EyeGaze Radar", true, true);
  pangolin::Var<bool> showRGBCamImg(prefix + ".RGB Img", true, true);
  pangolin::Var<bool> showSlamLCamImg(prefix + ".SLam Left Img", true, true);
  pangolin::Var<bool> showSlamRCamImg(prefix + ".Slam Right Img", true, true);
  pangolin::Var<std::int64_t> timestampDisplay(prefix + ".Timestamp us", 0., 0., 0., false);

  // Define a function to draw EyeGazeVector on a VRS image
  std::function<void(
      pangolin::View&, pangolin::ImageView&, const vrs::StreamId&, const std::string&)>
      f_drawEyeGazeVector = [this](
                                pangolin::View& v,
                                pangolin::ImageView& imageView,
                                const vrs::StreamId& vrsStreamId,
                                const std::string& cameraString) {
        v.ActivatePixelOrthographic();
        v.ActivateAndScissor();

        // Get stream image size
        auto cameraImageWidth = dataProvider_->getImageWidth(vrsStreamId);
        auto cameraImageHeight = dataProvider_->getImageHeight(vrsStreamId);

        bool isRgbStream = vrsStreamId == kRgbCameraStreamId;

        if (imageView.IsShown()) {
          if (isDataChanged(vrsStreamId)) {
            // Draw vrs image
            imageView.SetImage(
                static_cast<void*>(
                    cameraImageBufferMap_[vrsStreamId.getTypeId()][vrsStreamId.getInstanceId()]
                        .data()),
                cameraImageWidth,
                cameraImageHeight,
                isRgbStream ? cameraImageWidth * 3 : cameraImageWidth,
                isRgbStream ? pangolin::PixelFormatFromString("RGB24")
                            : pangolin::PixelFormatFromString("GRAY8"));
            setDataChanged(false, vrsStreamId);
          }

          // Display EyeGaze vector reprojection in the VRS image
          {
            // scale to the current display size
            const float scale = (float)v.v.w / (float)cameraImageWidth;

            glLineWidth(10);
            glColor3f(1.0, 1.0, 1.0);

            std::map<double, Eigen::Vector2d> camProjectionPerDepth;
            const double minDepth = .1; // 10cm
            const double maxDepth = 8.; // 8 meter
            const double depthSamples = 30.;
            ProjectEyeGazeRayInCamera(
                cameraString,
                deviceModel_,
                lastEyeGazeRecord_.first,
                minDepth,
                maxDepth,
                depthSamples,
                camProjectionPerDepth,
                cameraImageWidth,
                cameraImageHeight);

            // Plot the resulted "projected" ray
            // 1. compensate drawing in an OpenGL window
            std::vector<Eigen::Vector2d> pointsAlongGaze;
            {
              for (auto& it : camProjectionPerDepth) {
                // Flip Y axis to compensate rotation of the image and OpenGL axis swap
                it.second.y() = cameraImageHeight - it.second.y();
                // Rescale to the display context size
                it.second *= scale;
                pointsAlongGaze.push_back(it.second);
              }
            }
            // 2. Display the ray
            glLineWidth(3);
            glColor3f(1.0, 1.0, 1.0);
            pangolin::glDrawLineStrip(pointsAlongGaze);

            // Retrieve closest point to 1 and 2 meters and display a circle
            const auto itOneMeter = camProjectionPerDepth.lower_bound(1.);
            const auto itTwoMeter = camProjectionPerDepth.lower_bound(2.);
            if (itOneMeter != camProjectionPerDepth.end()) {
              pangolin::glDrawCircle(itOneMeter->second);
            }
            if (itTwoMeter != camProjectionPerDepth.end()) {
              pangolin::glDrawCircle(itTwoMeter->second);
            }
          }
        }
        v.GetBounds().DisableScissor();
      };

  cameraRGBView.extern_draw_function = [&](pangolin::View& v) {
    f_drawEyeGazeVector(v, cameraRGBView, kRgbCameraStreamId, "camera-rgb");
  };

  cameraSlamLView.extern_draw_function = [&](pangolin::View& v) {
    f_drawEyeGazeVector(v, cameraSlamLView, kSlamLeftCameraStreamId, "camera-slam-left");
  };

  cameraSlamRView.extern_draw_function = [&](pangolin::View& v) {
    f_drawEyeGazeVector(v, cameraSlamRView, kSlamRightCameraStreamId, "camera-slam-right");
  };

  // Main loop
  while (!pangolin::ShouldQuit()) {
    isPlaying_ = playButton;
    playbackSpeedFactor_ = playbackSlide;
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    timestampDisplay = currentTimestamp_;
    {
      const std::lock_guard<std::mutex> dataLock(dataMutex_);

      // Draw ET image and time series data as event are coming
      if (cameraETView.IsShown() && isDataChanged(kEyeCameraStreamId)) {
        // Draw ET Images
        cameraETView.SetImage(
            static_cast<void*>(cameraImageBufferMap_[kEyeCameraStreamId.getTypeId()]
                                                    [kEyeCameraStreamId.getInstanceId()]
                                                        .data()),
            etCameraImageWidth,
            etCameraImageHeight,
            etCameraImageWidth,
            pangolin::PixelFormatFromString("GRAY8"));
        setDataChanged(false, kEyeCameraStreamId);

        // Draw ET Gaze vector (time series)
        {
          const float pitch = -asin(-lastEyeGazeRecord_.first.y());
          const float yaw = atan2(-lastEyeGazeRecord_.first.x(), lastEyeGazeRecord_.first.z());
          const float uncertainty = lastEyeGazeRecord_.second;
          logEyeGaze.Log({yaw, pitch, uncertainty});

          // Add this measurement to the EyeGaze history buffer
          const Eigen::Vector2d center(yaw, pitch);
          eyeGazeHistory_.push_back(center);
          // If buffer too large remove first element
          // - create a rolling buffer
          if (eyeGazeHistory_.size() > 10) {
            eyeGazeHistory_.pop_front();
          }
        }
      }

      // Draw radar (using history saved data)
      if (!eyeGazeHistory_.empty()) {
        eyeGazeRadar.Activate(radar_view_camera);

        // Draw radar view background
        glColor3f(1, 1., 1.);
        for (double r = .1; r < 2; r += 0.5)
          pangolin::glDrawCirclePerimeter(Eigen::Vector2d(0, 0), r);

        // Draw eye gaze yaw, pitch history
        glColor3f(1, 0.8, 0);
        std::vector<Eigen::Vector2d> eyeGazeHistory{eyeGazeHistory_.begin(), eyeGazeHistory_.end()};
        pangolin::glDrawLineStrip(eyeGazeHistory);
        // Draw current gaze yaw, pitch
        const Eigen::Vector2f center(eyeGazeHistory_.back().x(), eyeGazeHistory_.back().y());
        pangolin::glDrawVertices(std::vector<Eigen::Vector2f>{center}, GL_POINTS);
        for (double r = .1; r < .3; r += 0.1)
          pangolin::glDrawCirclePerimeter(center.cast<double>(), r);
      }
    }

    // propagate show parameters
    container[0].Show(showETCamImg);
    cameraETView.Show(showETCamImg);
    container[1].Show(showETTemporal);
    plotEyeGaze.Show(showETTemporal);
    container[2].Show(showETRadar);
    eyeGazeRadar.Show(showETRadar);
    container[3].Show(showRGBCamImg);
    cameraRGBView.Show(showRGBCamImg);
    container[4].Show(showSlamLCamImg);
    cameraSlamLView.Show(showSlamLCamImg);
    container[5].Show(showSlamRCamImg);
    cameraSlamRView.Show(showSlamRCamImg);

    {
      std::lock_guard<std::mutex> lock(*p_render_mutex);
      pangolin::FinishFrame();
    }
  }
  std::cout << "\nQuit Viewer." << std::endl;
  exit(1);
}

// Convert a Timestamp stored in double to microseconds with std::chrono
constexpr auto durationDoubleToChronoUsCast(const double time_s) {
  using namespace std::chrono;
  using fsec = duration<double>;
  return round<microseconds>(fsec{time_s});
}

// Return Eye Gaze output
std::optional<std::pair<Eigen::Vector3d, float>> queryEyetrack(
    std::chrono::microseconds timestampUs,
    AriaViewer::EyeGazeDataRecords& timestampToEyetrack) {
  if (timestampUs < timestampToEyetrack.begin()->first ||
      timestampUs > timestampToEyetrack.rbegin()->first) {
    return {};
  }
  if (timestampToEyetrack.find(timestampUs) != timestampToEyetrack.end()) {
    return timestampToEyetrack.at(timestampUs);
  }
  // Linear interpolation if timestamp is falling before two records
  const auto laterEyePtr = timestampToEyetrack.lower_bound(timestampUs);
  const auto earlyEyePtr = std::prev(laterEyePtr);
  const int64_t timestamp = timestampUs.count();
  const int64_t tsEarly = earlyEyePtr->first.count();
  const int64_t tsLater = laterEyePtr->first.count();
  const Eigen::Vector3d eyeEarly = earlyEyePtr->second.first;
  const Eigen::Vector3d eyeLater = laterEyePtr->second.first;

  const double interpFactor =
      static_cast<double>(timestamp - tsEarly) / static_cast<double>(tsLater - tsEarly);
  const auto interpEye = (1.0 - interpFactor) * eyeEarly + interpFactor * eyeLater;
  const auto interpUncertainty =
      (1.0 - interpFactor) * earlyEyePtr->second.second + interpFactor * laterEyePtr->second.second;
  return std::make_pair(interpEye, interpUncertainty);
}

bool AriaViewer::readData(double currentTimestampSec) {
  if (isPlaying()) {
    {
      std::unique_lock<std::mutex> dataLock(dataMutex_);
      currentTimestamp_ = durationDoubleToChronoUsCast(currentTimestampSec).count();
      // Handle image streams & update eye gaze data if available
      for (auto& streamId : kImageStreamIds) {
        if (dataProvider_->tryFetchNextData(streamId, currentTimestampSec)) {
          if (streamId == dataprovider::kEyeCameraStreamId) {
            const auto ret =
                queryEyetrack(durationDoubleToChronoUsCast(currentTimestampSec), eyeGazeData_);
            if (ret)
              lastEyeGazeRecord_ = *ret;
          }
          auto imageBufferVector = dataProvider_->getImageBufferVector(streamId);
          if (imageBufferVector) {
            setDataChanged(true, streamId);
            setCameraImageBuffer(*imageBufferVector, streamId);
          }
        }
      }
    }
    return true;
  }
  return false;
}

} // namespace ark::datatools::visualization
