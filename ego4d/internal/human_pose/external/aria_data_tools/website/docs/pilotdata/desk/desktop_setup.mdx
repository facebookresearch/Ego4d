---
sidebar_position: 2
id: desktop_setup
title: Desktop Activities Capture Setup
---
# Desktop Activities Capture Setup

The Desktop Activities dataset was captured with a Project Aria device and a multi-view motion capture system.

## Hardware setup

To record this dataset we built a system with 16 [OptiTrack Prime X 13 W](https://optitrack.com/cameras/primex-13w/) motion tracking cameras and 12 [OptiTrack Prime Color FS](https://optitrack.com/cameras/prime-color/buy.html) color cameras with 1080 x 1920 pixel resolution. These cameras are mounted on a desktop rig, similar to the setup used to collect the [Assembly101 dataset](https://assembly-101.github.io/).

The multi-view system is calibrated with [OptiTrack Motive](https://optitrack.com/software/motive/) to obtain the intrinsics and extrinsics of all cameras. We attached markers to the Project Aria device and objects being manipulated to track their motion. We also calibrated the Project Aria device to obtain the sensor trajectories with respect to multi-view camera coordinates.


## Project Aria device sensor profile

Sensor profiles allow researchers to choose which sensors on the Project Aria device to use when collecting data.

For Desktop Activities, we used Sensor Profile M, so each Project Aria device recording contains:

* One RGB camera stream with 1408x1408 pixel resolution
* Two SLAM camera streams with 640x480 pixel resolution
* One eye tracking (ET) camera stream with 320x240 pixel resolution
* Two IMU sensor streams (1KHz and 800Hz)


**Table 1:** *Sensor Profile M*

|Sensor |Profile M |
|--- |--- |
|SLAM - resolution |640x480 |
|SLAM - FPS |15 |
|SLAM - encoding format |RAW |
|SLAM - bits per pixel (bpp) |8 |
|RGB - gain, exposure and temperature |Yes |
|ET - resolution |320x240 |
|ET - FPS |15 |
|ET - encoding format |JPEG |
|ET - bpp |8 |
|ET - gain and exposure |Yes |
|RGB - resolution |1408x1408 |
|RGB - FPS |15 |
|RGB - encoding format |JPEG |
|RGB - bpp |8 |
|RGB - gain, exposure and temperature |Yes |
|IMU - RIGHT acc/gyro - rate |1kHz |
|IMU - RIGHT temperature - rate |~1Hz |
|IMU - LEFT acc/gyro - rate |800Hz |
|IMU - LEFT temperature - rate |~1Hz |

## Trigger alignment and synchronized frames

During recording, the multi-view system and the Project Aria device operated at different frame rates. The Project Aria device recorded at 15FPS and with multi-view cameras recorded at 60 FPS. When recording an activity, the multi-view system and the Project Aria device started and stopped recording asynchronously.Leveraging [SMPTE timecode](https://en.wikipedia.org/wiki/SMPTE_timecode), all sensors were synchronized to a global timeline. In addition, all cameras were trigger aligned while recording.

Using Sensor Profile M, the Project Aria device produced 4 synchronized camera images (1 RGB, 2 SLAM, and 1 ET image) per frame. The multi-view system produced 12 synchronized RGB camera images per frame.

During the overlapping capture time, the camera trigger alignment let us accurately associate frames from the Project Aria device to frames from the multi-view system. With 15 FPS for the Project Aria device and 60 FPS for the multi-view system, 1 out of every 4 multi-view frames was trigger aligned to one Project Aria device frame.
