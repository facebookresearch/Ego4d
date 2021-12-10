#! /bin/bash

if [ $# -lt 2 ]; then
    echo "Usage: $0 in-video-dir out-wav-dir"
    exit 1
fi

which ffmpeg

videoDir=$1
wavDir=$2

mkdir -p $wavDir

for f in ${videoDir}/*.mp4; do
   clip=$(basename $f)
   wavFile=${wavDir}/${clip%.mp4}.wav
   echo $wavFile
   ffmpeg -i "$f" -ac 1 -c pcm_s16le -ar 16000 \
          -y -f wav ${wavFile} < /dev/null;
done
