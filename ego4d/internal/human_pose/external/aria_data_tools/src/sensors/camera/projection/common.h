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

#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <Eigen/Core>

#include <camera/math/optimization/IgnoreJetInfinitesimal.h>
#include <camera/portability/Inline.h>

namespace ark {
namespace datatools {
namespace sensors {
namespace CameraNewtonsMethod {
constexpr int kMaxIterations = 50;
constexpr float kFloatTolerance = 1e-5;
constexpr float kDoubleTolerance = 1e-7;

template <typename T>
constexpr ARK_HOST_DEVICE float getConvergenceTolerance() {
  if (std::is_same<T, ceres::Jet<float, T::DIMENSION>>::value) {
    return kFloatTolerance;
  }
  if (std::is_same<T, ceres::Jet<double, T::DIMENSION>>::value) {
    // largest number that passes project / unproject test to within 1e-8 pixels for all models.
    return kDoubleTolerance;
  }
}

template <>
constexpr ARK_HOST_DEVICE float getConvergenceTolerance<float>() {
  return kFloatTolerance;
}

template <>
constexpr ARK_HOST_DEVICE float getConvergenceTolerance<double>() {
  return kDoubleTolerance;
}

template <typename T>
inline ARK_HOST_DEVICE bool hasConverged(const T& step) {
  using std::abs;
  return abs(IgnoreJetInfinitesimal(step)) < getConvergenceTolerance<T>();
}

template <typename T>
inline ARK_HOST_DEVICE bool hasConverged(const T& stepA, const T& stepB) {
  return IgnoreJetInfinitesimal(stepA * stepA + stepB * stepB) <
      getConvergenceTolerance<T>() * getConvergenceTolerance<T>();
}

template <typename T>
inline ARK_HOST_DEVICE bool hasConverged(const Eigen::Matrix<T, 2, 1>& step) {
  return IgnoreJetInfinitesimal(step.squaredNorm()) <
      getConvergenceTolerance<T>() * getConvergenceTolerance<T>();
}

template <typename T>
inline ARK_HOST_DEVICE T initTheta(const T& r) {
  // tried a lot of fancy ways to initialize, this seems to work best.
  // the model has just gone through an tan(x) at this point, so while most points lie in the range
  // of about 0 to 2, points that were near the camera corners might take extreme values of 1000+.
  // For zero distortion the optimal output is theta = r, however as r increases all the lenses we
  // use seem to bias towards lower values. This warrents futher investigation however with the
  // naive zero distortion initialization some models fail for some points near 90 degrees. The sqrt
  // relationship offers a good tradeoff as it is fast to calculate and has minimal impact on most
  // points while pulling back extreme points to be more reasonable.
  using std::sqrt;
  return sqrt(r);
}
} // namespace CameraNewtonsMethod
} // namespace sensors
} // namespace datatools
} // namespace ark
