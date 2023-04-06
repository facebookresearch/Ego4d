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

#include <pangolin/display/display.h>
#include <pangolin/display/view.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/gl/glfont.h>
#include <pangolin/gl/gltext.h>
#include <pangolin/scene/axis.h>
#include <pangolin/scene/scenehandler.h>
#include <cstdlib>

#include "data_provider/AriaVrsDataProvider.h"
#include "models/DeviceModel.h"

using namespace ark::datatools;

extern const unsigned char AnonymousPro_ttf[];
static pangolin::GlFont kGlFont(AnonymousPro_ttf, 20);

int main(int argc, char** argv) {
  if (argc < 1) {
    std::cout << "VRS file must be provided as the argument, exiting." << std::endl;
    return EXIT_FAILURE;
  }

  const std::string vrsPath = argv[1];

  // get and open the VRS data provider
  dataprovider::AriaVrsDataProvider dataProvider;
  if (!dataProvider.open(vrsPath)) {
    std::cerr << "Failed to open " << vrsPath << std::endl;
    return EXIT_FAILURE;
  }

  // Enable all Streams available in the file and read their config records
  for (auto id : dataProvider.getStreamsInFile()) {
    dataProvider.setStreamPlayer(id);
    dataProvider.readFirstConfigurationRecord(id);
  }

  // Load and retrieve the device model (calibration information)
  dataProvider.loadDeviceModel();
  const sensors::DeviceModel& deviceModelData = dataProvider.getDeviceModel();

  // Retrieve microphone labels and sort them in alphabetic order to ease some drawing
  std::vector<std::string> micLabels = deviceModelData.getMicrophoneLabels();
  std::sort(micLabels.begin(), micLabels.end());
  std::vector<Eigen::Vector3d> micTranslations;
  for (const auto& micLabelIt : micLabels) {
    const auto& pose = deviceModelData.getMicrophoneCalib(micLabelIt)->T_Device_Microphone;
    micTranslations.push_back(pose.translation());
  }

  // Configure the Window and pangolin OpenGL Context for 3D drawing
  pangolin::CreateWindowAndBind("Aria Sensors Viewer", 640, 480);
  glEnable(GL_DEPTH_TEST);

  // Define Projection and initial ModelView matrix
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.02, 100),
      pangolin::ModelViewLookAt(0.15, 0.15, 0.06, 0, 0, 0, pangolin::AxisZ));

  pangolin::Renderable tree;
  // Create Interactive View in window
  pangolin::SceneHandler handler(tree, s_cam);
  pangolin::View& d_cam = pangolin::CreateDisplay()
                              .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f)
                              .SetHandler(&handler);

  while (!pangolin::ShouldQuit()) {
    // Clear screen and activate view to render into
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    d_cam.Activate(s_cam);
    pangolin::glDrawAxis(0.1);

    // Draw cameras
    for (const auto& label_it : deviceModelData.getCameraLabels()) {
      const auto& camera = deviceModelData.getCameraCalib(label_it);
      const auto& pose = camera->T_Device_Camera;
      pangolin::glSetFrameOfReference(pose.matrix());
      glColor3f(1.0f, 0.0f, 1.0f);

      const float sz = 0.03;
      auto fs = camera->projectionModel.getFocalLengths();
      auto cs = camera->projectionModel.getPrincipalPoint();
      pangolin::glDrawAxis(0.01);
      if (camera->label.find("slam") != std::string::npos) {
        pangolin::glDrawFrustum(
            -cs(0) / fs(0), -cs(1) / fs(1), 1. / fs(0), 1. / fs(1), 640, 480, sz * 0.8);
      } else if (camera->label.find("et") != std::string::npos) {
        pangolin::glDrawFrustum(
            -cs(0) / fs(0), -cs(1) / fs(1), 1. / fs(0), 1. / fs(1), 320, 240, sz * 0.8);
      } else if (camera->label.find("rgb") != std::string::npos) {
        pangolin::glDrawFrustum(
            -cs(0) / fs(0), -cs(1) / fs(1), 1. / fs(0), 1. / fs(1), 1408, 1408, sz * 0.8);
      }
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
      kGlFont.Text(label_it.c_str()).Draw(0, 0, 0);
      glDisable(GL_BLEND);
      pangolin::glUnsetFrameOfReference();
    }

    // Draw microphones
    for (const auto& label_it : deviceModelData.getMicrophoneLabels()) {
      const auto& pose = deviceModelData.getMicrophoneCalib(label_it)->T_Device_Microphone;
      glColor3f(0.0f, 1.0f, 0.0f);
      pangolin::glSetFrameOfReference(pose.matrix());
      pangolin::glDrawCross(0.0, 0.0, 0.0, 0.01);
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
      kGlFont.Text(label_it.c_str()).Draw(0, 0, 0);
      glDisable(GL_BLEND);
      pangolin::glUnsetFrameOfReference();
    }

    // Draw Barometer
    for (const auto& label_it : deviceModelData.getBarometerLabels()) {
      const auto& pose = deviceModelData.getBarometerCalib(label_it)->T_Device_Barometer;
      glColor3f(1.0f, 1.0f, 0.0f);
      pangolin::glSetFrameOfReference(pose.matrix());
      pangolin::glDrawCross(0.0, 0.0, 0.0, 0.01);
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
      kGlFont.Text(label_it.c_str()).Draw(0, 0, 0);
      glDisable(GL_BLEND);
      pangolin::glUnsetFrameOfReference();
    }

    // Draw Magnetometer
    for (const auto& label_it : deviceModelData.getMagnetometerLabels()) {
      const auto& pose = deviceModelData.getMagnetometerCalib(label_it)->T_Device_Magnetometer;
      glColor3f(0.0f, 1.0f, 1.0f);
      pangolin::glSetFrameOfReference(pose.matrix());
      pangolin::glDrawCross(0.0, 0.0, 0.0, 0.01);
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
      kGlFont.Text(label_it.c_str()).Draw(0, 0, 0);
      glDisable(GL_BLEND);
      pangolin::glUnsetFrameOfReference();
    }

    if (!micTranslations.empty()) {
      // Draw some wireframe lines to better visualize the glasses
      std::vector<Eigen::Vector3d> lines = {
          micTranslations[5],
          micTranslations[3],
          micTranslations[2],
          micTranslations[1],
          micTranslations[3],
          micTranslations[4],
          micTranslations[0],
          micTranslations[1],
          micTranslations[4],
          micTranslations[6]};
      pangolin::glDrawLineStrip(lines);
    }

    // Swap frames and Process Events
    pangolin::FinishFrame();
  }

  return EXIT_SUCCESS;
}
