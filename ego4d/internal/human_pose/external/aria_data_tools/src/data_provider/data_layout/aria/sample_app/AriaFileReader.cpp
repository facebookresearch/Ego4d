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

#include <cassert>

#include <cstdio>

#include <vrs/ErrorCode.h>
#include <vrs/RecordFileReader.h>
#include <vrs/RecordFormatStreamPlayer.h>

#include <data_layout/aria/AudioMetadata.h>
#include <data_layout/aria/BarometerMetadata.h>
#include <data_layout/aria/BluetoothBeaconMetadata.h>
#include <data_layout/aria/GpsMetadata.h>
#include <data_layout/aria/ImageSensorMetadata.h>
#include <data_layout/aria/MotionSensorMetadata.h>
#include <data_layout/aria/TimeSyncMetadata.h>
#include <data_layout/aria/WifiBeaconMetadata.h>

using namespace std;
using namespace vrs;

namespace {

void printDataLayout(const CurrentRecord& r, DataLayout& datalayout) {
  fmt::print(
      "{:.3f} {} record, {} [{}]\n",
      r.timestamp,
      toString(r.recordType),
      r.streamId.getName(),
      r.streamId.getNumericName());
  datalayout.printLayoutCompact(cout, "  ");
}

class AriaImageSensorPlayer : public RecordFormatStreamPlayer {
  bool onDataLayoutRead(const CurrentRecord& r, size_t blockIndex, DataLayout& dl) override {
    if (r.recordType == Record::Type::CONFIGURATION) {
      auto& config = getExpectedLayout<aria::ImageSensorConfigRecordMetadata>(dl, blockIndex);
      // Read config record metadata...
      printDataLayout(r, config);
    } else if (r.recordType == Record::Type::DATA) {
      auto& data = getExpectedLayout<aria::ImageSensorDataRecordMetadata>(dl, blockIndex);
      // Read data record metadata...
      printDataLayout(r, data);
    }
    return true;
  }
  bool onImageRead(const CurrentRecord& r, size_t /*idx*/, const ContentBlock& cb) override {
    // the image data was not read yet: allocate your own buffer & read!
    vector<uint8_t> frameBytes(cb.getBlockSize());
    const auto& imageSpec = cb.image();
    // Synchronously read the image data, which is jpg compressed with Aria
    if (cb.image().getImageFormat() == ImageFormat::JPG && r.reader->read(frameBytes) == 0) {
      /// do your thing with the jpg data...
      fmt::print(
          "{:.3f} {} [{}]: {}, {} bytes.\n",
          r.timestamp,
          r.streamId.getName(),
          r.streamId.getNumericName(),
          imageSpec.asString(),
          imageSpec.getBlockSize());
    }
    return true; // read next blocks, if any
  }
};

class AriaMotionSensorPlayer : public RecordFormatStreamPlayer {
  bool onDataLayoutRead(const CurrentRecord& r, size_t blockIndex, DataLayout& dl) override {
    if (r.recordType == Record::Type::CONFIGURATION) {
      auto& config = getExpectedLayout<aria::MotionSensorConfigRecordMetadata>(dl, blockIndex);
      // Read config record metadata...
      printDataLayout(r, config);
    } else if (r.recordType == Record::Type::DATA) {
      auto& data = getExpectedLayout<aria::MotionSensorDataRecordMetadata>(dl, blockIndex);
      // Read data record metadata...
      printDataLayout(r, data);
    }
    return true;
  }
};

class AriaAudioPlayer : public RecordFormatStreamPlayer {
  bool onDataLayoutRead(const CurrentRecord& r, size_t blockIndex, DataLayout& dl) override {
    if (r.recordType == Record::Type::CONFIGURATION) {
      auto& config = getExpectedLayout<aria::AudioConfigRecordMetadata>(dl, blockIndex);
      // Read config record metadata...
      printDataLayout(r, config);
    } else if (r.recordType == Record::Type::DATA) {
      auto& data = getExpectedLayout<aria::AudioDataRecordMetadata>(dl, blockIndex);
      // Read data record metadata...
      printDataLayout(r, data);
    }
    return true;
  }
  bool onAudioRead(const CurrentRecord& r, size_t /*blockIdx*/, const ContentBlock& cb) override {
    const AudioContentBlockSpec& audioSpec = cb.audio();
    assert(audioSpec.getSampleFormat() == AudioSampleFormat::S32_LE);
    vector<int32_t> audioData(audioSpec.getSampleCount() * audioSpec.getChannelCount());
    // actually read the audio data
    if (r.reader->read(audioData) == 0) {
      fmt::print(
          "{:.3f} {} [{}]: {} {}x{} samples.\n",
          r.timestamp,
          r.streamId.getName(),
          r.streamId.getNumericName(),
          audioSpec.asString(),
          audioSpec.getSampleCount(),
          audioSpec.getChannelCount());
    }
    return true;
  }
};

class AriaWifiBeaconPlayer : public RecordFormatStreamPlayer {
  bool onDataLayoutRead(const CurrentRecord& r, size_t blockIndex, DataLayout& dl) override {
    if (r.recordType == Record::Type::CONFIGURATION) {
      auto& config = getExpectedLayout<aria::WifiBeaconConfigRecordMetadata>(dl, blockIndex);
      // Read config record metadata...
      printDataLayout(r, config);
    } else if (r.recordType == Record::Type::DATA) {
      auto& data = getExpectedLayout<aria::WifiBeaconDataRecordMetadata>(dl, blockIndex);
      // Read data record metadata...
      printDataLayout(r, data);
    }
    return true;
  }
};

class AriaBluetoothBeaconPlayer : public RecordFormatStreamPlayer {
  bool onDataLayoutRead(const CurrentRecord& r, size_t blockIndex, DataLayout& dl) override {
    if (r.recordType == Record::Type::CONFIGURATION) {
      auto& config = getExpectedLayout<aria::BluetoothBeaconConfigRecordMetadata>(dl, blockIndex);
      // Read config record metadata...
      printDataLayout(r, config);
    } else if (r.recordType == Record::Type::DATA) {
      auto& data = getExpectedLayout<aria::BluetoothBeaconDataRecordMetadata>(dl, blockIndex);
      // Read data record metadata...
      printDataLayout(r, data);
    }
    return true;
  }
};

class AriaGpsPlayer : public RecordFormatStreamPlayer {
  bool onDataLayoutRead(const CurrentRecord& r, size_t blockIndex, DataLayout& dl) override {
    if (r.recordType == Record::Type::CONFIGURATION) {
      auto& config = getExpectedLayout<aria::GpsConfigRecordMetadata>(dl, blockIndex);
      // Read config record metadata...
      printDataLayout(r, config);
    } else if (r.recordType == Record::Type::DATA) {
      auto& data = getExpectedLayout<aria::GpsDataRecordMetadata>(dl, blockIndex);
      // Read data record metadata...
      printDataLayout(r, data);
    }
    return true;
  }
};

class AriaBarometerPlayer : public RecordFormatStreamPlayer {
  bool onDataLayoutRead(const CurrentRecord& r, size_t blockIndex, DataLayout& dl) override {
    if (r.recordType == Record::Type::CONFIGURATION) {
      auto& config = getExpectedLayout<aria::BarometerConfigRecordMetadata>(dl, blockIndex);
      // Read config record metadata...
      printDataLayout(r, config);
    } else if (r.recordType == Record::Type::DATA) {
      auto& data = getExpectedLayout<aria::BarometerDataRecordMetadata>(dl, blockIndex);
      // Read data record metadata...
      printDataLayout(r, data);
    }
    return true;
  }
};

class AriaTimeSyncPlayer : public RecordFormatStreamPlayer {
  bool onDataLayoutRead(const CurrentRecord& r, size_t blockIndex, DataLayout& dl) override {
    if (r.recordType == Record::Type::CONFIGURATION) {
      auto& config = getExpectedLayout<aria::TimeSyncConfigRecordMetadata>(dl, blockIndex);
      // Read config record metadata...
      printDataLayout(r, config);
    } else if (r.recordType == Record::Type::DATA) {
      auto& data = getExpectedLayout<aria::TimeSyncDataRecordMetadata>(dl, blockIndex);
      // Read data record metadata...
      printDataLayout(r, data);
    }
    return true;
  }
};

struct AriaFileReader {
  /// This function is the entry point for your reader
  static void readFile(const string& vrsFilePath) {
    RecordFileReader reader;
    int status = reader.openFile(vrsFilePath);
    if (status == SUCCESS) {
      vector<unique_ptr<StreamPlayer>> streamPlayers;
      // Map the devices referenced in the file to stream player objects
      // Just ignore the device(s) you do not care for
      for (auto id : reader.getStreams()) {
        unique_ptr<StreamPlayer> streamPlayer;
        switch (id.getTypeId()) {
          case RecordableTypeId::SlamCameraData:
          case RecordableTypeId::RgbCameraRecordableClass:
          case RecordableTypeId::EyeCameraRecordableClass:
            streamPlayer = make_unique<AriaImageSensorPlayer>();
            break;
          case RecordableTypeId::SlamImuData:
          case RecordableTypeId::SlamMagnetometerData:
            streamPlayer = make_unique<AriaMotionSensorPlayer>();
            break;
          case RecordableTypeId::WifiBeaconRecordableClass:
            streamPlayer = make_unique<AriaWifiBeaconPlayer>();
            break;
          case RecordableTypeId::StereoAudioRecordableClass:
            streamPlayer = make_unique<AriaAudioPlayer>();
            break;
          case RecordableTypeId::BluetoothBeaconRecordableClass:
            streamPlayer = make_unique<AriaBluetoothBeaconPlayer>();
            break;
          case RecordableTypeId::GpsRecordableClass:
            streamPlayer = make_unique<AriaGpsPlayer>();
            break;
          case RecordableTypeId::BarometerRecordableClass:
            streamPlayer = make_unique<AriaBarometerPlayer>();
            break;
          case RecordableTypeId::TimeRecordableClass:
            streamPlayer = make_unique<AriaTimeSyncPlayer>();
            break;
          default:
            fmt::print("Unexpected stream: {}, {}.\n", id.getNumericName(), id.getName());
            break;
        }
        if (streamPlayer) {
          reader.setStreamPlayer(id, streamPlayer.get());
          streamPlayers.emplace_back(std::move(streamPlayer));
        }
      }
      if (streamPlayers.empty()) {
        fmt::print(stderr, "Found no Aria stream in '{}'...\n", vrsFilePath);
      } else {
        fmt::print("Found {} Aria streams in '{}'.\n", streamPlayers.size(), vrsFilePath);
        reader.readAllRecords();
      }
    } else {
      fmt::print(stderr, "Failed to open '{}', {}.\n", vrsFilePath, errorCodeToMessage(status));
    }
  }
};

} // namespace

int main(int argc, char** argv) {
  if (argc > 1) {
    AriaFileReader::readFile(argv[1]);
  }
  return 0;
}
