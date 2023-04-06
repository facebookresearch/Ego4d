# Sensor Models and Calibration

## Introduction
Aria devices include multiple types of sensors, all of which are calibrated at manufacturing
stage for every device. The calibration process derives intrinsic and extrinsic parameters
(relative poses between sensors) between some sensors. These information
is stored on every device and stamped on every VRS data file it records. One can fetch this
info from VRS using Aria Data Provider and parse it into a structured representation as:
```
>>> import pyark.datatools as datatools
>>> vrs_data_provider = datatools.dataprovider.AriaVrsDataProvider()
>>> vrs_data_provider.openFile('./data/aria_unit_test_sequence_calib.vrs')
# There are calibration strings for each image and motion stream
# Reading the configuration record for any one of them will load the device model
>>> slam_camera_recordable_type_id = 1201
>>> slam_left_camera_instance_id = 1
>>> slam_left_camera_stream_id = datatools.dataprovider.StreamId(slam_camera_recordable_type_id, slam_left_camera_instance_id)
>>> vrs_data_provider.setStreamPlayer(slam_left_camera_stream_id)
>>> vrs_data_provider.readFirstConfigurationRecord(slam_left_camera_stream_id)
True
>>> vrs_data_provider.loadDeviceModel()
True
>>> device_model = vrs_data_provider.getDeviceModel()
>>> device_model
<pyark.datatools.sensors.DeviceModel object at 0x7f955808c2b0>
```

## Sensors
An Aria device includes the following sensors:
* SLAM cameras (x2)
* Eyetracking cameras (x2)
* RGB camera (x1)
* IMU (x2)
* Barometer (x1)
* GPS (x1)
* Microphones array (x7)
* Wifi and Bluetooth receiver

Each sensor records its data as a stream in a VRS file (with eyetracking be the only exception
where the two cameras generate one concatenated image). For now, ARK Sensor Model and Calibration
SDK supports cameras and IMUs.

In calibration and other applications, a sensor is associated with an instance-invariant name.
For example, the left SLAM camera is named as "camera-slam-left". The name set of supported sensors
can be fetched as:
```
>>> device_model.getCameraLabels()
['camera-et-left', 'camera-et-right', 'camera-rgb', 'camera-slam-left', 'camera-slam-right']
>>> device_model.getImuLabels()
['imu-left', 'imu-right']
>>> device_model.getMagnetometerLabels()
['mag0']
>>> device_model.getBarometerLabels()
['baro0']
>>> device_model.getMicrophoneLabels()
['mic6', 'mic5', 'mic4', 'mic1', 'mic3', 'mic2', 'mic0']
```

## Concepts

### Coordinate Systems

Applications like stereo vision and navigation usually handle 2D and 3D points in different
spaces and transforms between them. In ARK, we attach a local coordinate frame R^3 to each sensor.
More specifically:

* **Camera**: The origin is at the optical axis in the central pupil plane. When facing the camera,
  X+ points to right, Y+ points down and Z+ points outwards.
* **IMU**: The accelerometer is picked as the origin of the IMU frame. When facing the chip, X+
  points to right, Y+ points to up and Z+ points outwards.

### Extrinsics

Extrinsics is the transformation of a 3D point between two coordinate systems. This is represented
as an SE(3). In ARK, we use [Sophus](https://github.com/strasdat/Sophus) library for SE(3)
representation and arithmatics, with the rotation part (SO(3)) as a unit quaternion and translation
as an R^3.

In both the code and the documentation throughout this project, we use the following notations:
* `p_sensor` represents an R^3 point in the local coordinate system of `sensor`. e.g. `p_slamLeft`.
* `T_sensor1_sensor2` reprensents an SE3 transformation from `sensor2` frame to `sensor1` frame.
An easy mnemonics is the chaining principle:
`T_sensor1_sensor2 * T_sensor2_sensor3 * p_sensor3 = p_sensor1`.

One can transform a point from one frame to the other with the `transform()` API:
```
>>> import numpy as np
>>> p_slamLeft = np.array([3.0, 2.0, 1.0])
>>> p_imuRight = device_model.transform(p_slamLeft, 'camera-slam-left', 'imu-right')
>>> p_imuRight
array([ 3.33343274, -1.41484796,  1.20512771])
>>> device_model.transform(p_imuRight, 'imu-right', 'camera-slam-left')
array([3., 2., 1.])
```

### Intrinsics

Cameras can be formulated as a function that maps a 3D point in its local coordinate frame to
the image pixel space. The parameters of this projection function are called the intrinsic parameter
of a camera. Note that all cameras on Aria are fisheye cameras, meaning they are modelled by
a spehrical projection plus subsequent additional distortion correction (rather than being modelled
by a pinhole projection plus distortion).

For Aria devices, we use:

* Kannala-Brandt model for eyetracking cameras;
* FisheyeRadTanThinPrism model for SLAM and RGB cameras.

One can perform the projection and unprojection operations as follows:

```
>>> p_slamLeft = np.array([3.0, 2.0, 1.0])
>>> uv_slamLeft = device_model.getCameraCalib('camera-slam-left').projectionModel.project(p_slamLeft)
>>> uv_slamLeft
array([583.48105528, 411.98136675])
>>> device_model.getCameraCalib('camera-slam-left').projectionModel.unproject(uv_slamLeft)
array([3., 2., 1.])
```

IMUs use a linear rectification model for both accelerometers and gyroscopes to rectify
an R^3 point in its local coordinate system. The model include a 3x3 rectification matrix A
(correcting scale and non-orthogonality) and a 3x1 bias vector `b`. The rectification process
applies the following formula:
```
p_real = A.inv() * (p_raw - b)
```
For accelerometer, `p_raw` is the acceleration, for gyroscope it is the angular velocity.
One can perform the rectification as follows:

```
>>> p_imuLeft = np.array([3.0, 2.0, 1.0])
>>> device_model.getImuCalib('imu-left').accel.compensateForSystematicErrorFromMeasurement(p_imuLeft)
array([2.93735023, 2.02130446, 0.87514154])
```

### Time
Every signal (or Record in VRS terms) collected by sensors is stamped with timestamp from a common
clock. For Aria dataset this is usually the board clock. All records are sorted in monotonically
increasing order in a VRS file. In addition, we have some sensor-specific conventions on timestamps:

* For cameras, the timestamp of a frame is the center exposure time, a.k.a. the middle point of
  exposure interval.
* RGB camera uses a rolling shutter, with a readout time of 5ms(low-res)/15ms(high-res) from top to
  bottom. The recorded timestamp of an RGB frame is the center exposure timestamp of the center row
  of the image. SLAM and eyetracking cameras use global shutters and do not have this issue.
* For IMUs, both accelerometers and gyroscopes may have a time offset from the board clock. This is
  calibrated and stored in the JSON as `TimeOffsetSec_Device_Gyro` and `TimeOffsetSec_Device_Accel`.

## Remarks
Until further notice, the units of all numerical values in the code and documentations use
the following conventions:

* Coordinates, location and distance in world space: meters (m)
* Coordinates in image space: pixels
* Timestamp and time intervals: seconds (s)
* Angles: radians (rad)
* Acceleration: m/s^2
* Rotation: rad/s
