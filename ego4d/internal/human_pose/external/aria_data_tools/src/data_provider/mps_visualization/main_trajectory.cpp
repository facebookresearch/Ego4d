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
#include <pangolin/scene/axis.h>
#include <pangolin/scene/scenehandler.h>
#include <sophus/types.hpp>
#include <algorithm>

#include "eyeGazeReader.h"
#include "trajectoryReader.h"

using namespace ark::datatools;

int main(int argc, char** argv) {
  if (argc < 1) {
    std::cout << "Location file must be provided as the argument, exiting." << std::endl;
    return EXIT_FAILURE;
  }

  // Data IO
  Trajectory loadedPoses;
  TemporalEyeGazeData eyeGazeRecords;
  {
    // Read the camera trajectory if any
    {
      // - It can be an open or close loop trajectory file.
      const std::string locationFilePath = argv[1];

      if (locationFilePath.find("open_loop_") != std::string::npos) {
        std::cout << "Reading an Open Loop Trajectory file" << '\n';
        loadedPoses = readOpenLoop(locationFilePath);
      } else if (locationFilePath.find("closed_loop_") != std::string::npos) {
        loadedPoses = readCloseLoop(locationFilePath);
      } else {
        std::cerr
            << "The filename does not contains the expected prefix 'open_loop_' or 'closed_loop_'"
            << std::endl;
        return EXIT_FAILURE;
      }
    }

    // Read Eye Gaze data record if any
    eyeGazeRecords = (argc >= 3) ? readEyeGaze(argv[2]) : TemporalEyeGazeData();
  }

  // Convert data for 3D visualization
  std::vector<Eigen::Vector3d> devicePoses;
  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> eyeGazeVectors;
  {
    // Convert the devicePoses to a format we can display with OpenGL
    {
      devicePoses.reserve(loadedPoses.size());
      for (const auto& pose : loadedPoses) {
        devicePoses.push_back(pose.T_world_device.translation());
      }
    }

    // Convert read Eye Gaze data for 3D visualization
    {
      // For each data point:
      //  - retrieve the closest pose
      //  - build a line with the eye gaze ray (default assume 1 meter depth)
      for (const auto& it : eyeGazeRecords) {
        const auto eyeGazeTimestamp = it.first;
        // Retrieve the closest pose
        auto pose_info = std::lower_bound(
            loadedPoses.begin(),
            loadedPoses.end(),
            eyeGazeTimestamp,
            [](const TrajectoryPose& pose, std::chrono::microseconds value) {
              return pose.tracking_timestamp_us < value;
            });
        if (pose_info != loadedPoses.end()) {
          const Eigen::Vector3d center = pose_info->T_world_device.translation();
          const Eigen::Vector3d gazeEndPoint =
              center + pose_info->T_world_device.rotationMatrix() * it.second.gaze_vector;
          // WIP: TBD Apply transform?
          eyeGazeVectors.emplace_back(center, gazeEndPoint);
        }
      }
    }
  }

  // Configure the Window and pangolin OpenGL Context for 3D drawing
  pangolin::CreateWindowAndBind("AriaTrajectoryViewer", 640, 480);
  glEnable(GL_DEPTH_TEST);

  // Define Projection and initial ModelView matrix
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
      pangolin::ModelViewLookAt(2.0, 2.0, 1.0, 0, 0, 0, pangolin::AxisZ));

  pangolin::Renderable tree;
  auto axis_i = std::make_shared<pangolin::Axis>();
  axis_i->T_pc = pangolin::OpenGlMatrix::Translate(0., 0., 0);
  tree.Add(axis_i);

  // Create Interactive View in window
  pangolin::SceneHandler handler(tree, s_cam);
  pangolin::View& d_cam = pangolin::CreateDisplay()
                              .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f)
                              .SetHandler(&handler);

  d_cam.SetDrawFunction([&](pangolin::View& view) {
    view.Activate(s_cam);
    tree.Render();

    // Display device trajectory
    glColor3f(1, 0.8, 0);
    glLineWidth(3);
    pangolin::glDrawLineStrip(devicePoses);

    // Display Eye Gaze vectors
    glColor3f(0.4, 0.8, 1);
    for (const auto& it : eyeGazeVectors) {
      pangolin::glDrawLine(
          it.first.x(), it.first.y(), it.first.z(), it.second.x(), it.second.y(), it.second.z());
    }
  });

  while (!pangolin::ShouldQuit()) {
    // Clear screen and activate view to render into
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Swap frames and Process Events
    pangolin::FinishFrame();
  }

  return EXIT_SUCCESS;
}
