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

// HOST / DEVICE Annotations
#if defined(__CUDACC__)

// For the NVCC specific function specifiers
#include <cuda_runtime.h>

#define ARVR_HOST_DEVICE __host__ __device__
#define ARVR_DEVICE_INLINE __device__ __forceinline__
#define ARVR_HOST_DEVICE_INLINE __host__ __device__ __forceinline__

#else // !defined(__CUDACC__)

#define ARVR_HOST_DEVICE
#define ARVR_DEVICE_INLINE inline
#define ARVR_HOST_DEVICE_INLINE inline

#endif // !defined(__CUDACC__)
