# pyre-unsafe

import json
import math
import os
import subprocess
import boto3
from botocore.exceptions import ClientError
import logging
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

@dataclass(frozen=True)
class VideoInfo:
    fps: Fraction
    sar: Fraction
    dar: Optional[Fraction]
    sample_width: int
    sample_height: int
    codec: Optional[str] = None
    audio_sample_rate: Optional[int] = None
    audio_channel_layout: Optional[str] = None
    rotate: Optional[int] = None
    unstructured_stream_data: Optional[Dict[str, Any]] = None

    @property
    def display_height(self) -> int:
        return self.sample_height

    @property
    def display_width(self) -> int:
        assert self.sar is not None
        return int(self.sample_width * (self.sar.numerator / self.sar.denominator))



def create_presigned_url(s3_client, bucket_name, object_name, expiration=5):
    """Generate a presigned URL to share an S3 object

    :param bucket_name: string
    :param object_name: string
    :param expiration: Time in seconds for the presigned URL to remain valid
    :return: Presigned URL as string. If error, returns None.
    """

    # Generate a presigned URL for the S3 object
    # s3_client = boto3.client('s3')
    try:
        response = s3_client.generate_presigned_url('get_object',
                                                    Params={'Bucket': bucket_name,
                                                            'Key': object_name},
                                                    ExpiresIn=expiration)
    except ClientError as e:
        logging.error(e)
        return None

    # The response contains the presigned URL
    return response

def get_video_info(s3_client, bucket_name, object_name, expiration=5) -> VideoInfo:
    filename = create_presigned_url(s3_client, bucket_name, object_name, expiration)
    cmd = [
        "ffprobe",
        "-show_streams",
        "-of",
        "json",
        f"{filename}",
    ]

    print("Command =")
    print(" ".join(cmd))

    def to_fraction(fps):
        if fps is None:
            return 0
        a, b = fps.split("/")
        if int(b) == 0:
            return 0
        return Fraction(int(a), int(b))

    try:
        result = subprocess.run(cmd, encoding="utf-8", capture_output=True)

    except Exception as e:
        logging.error(e)
        return None

    data = json.loads(result.stdout)

    fps = None
    width = None
    height = None
    sar = Fraction(1, 1)
    dar = None
    codec = None
    audio_sample_rate = None
    channel_layout = None
    rotate = None

    # it's possible for some files
    # to not have this information available
    if "streams" in data:
        for stream in data["streams"]:
            if "avg_frame_rate" in stream:
                s_fps = to_fraction(stream["avg_frame_rate"])
                if fps is None or fps < s_fps:
                    fps = s_fps

            if "sample_aspect_ratio" in stream:
                a, b = stream["sample_aspect_ratio"].split(":")
                sar = Fraction(int(a), int(b))

            if "display_aspect_ratio" in stream:
                a, b = stream["display_aspect_ratio"].split(":")
                dar = Fraction(int(a), int(b))

            if "width" in stream:
                # don't do this for audio streams
                if "codec_name" in stream:
                    codec = stream["codec_name"]

                width = stream["width"]

            if "height" in stream:
                height = stream["height"]

            if "sample_rate" in stream:
                audio_sample_rate = int(stream["sample_rate"])

            if "channel_layout" in stream:
                channel_layout = stream["channel_layout"]

            if "tags" in stream:
                if "rotate" in stream["tags"]:
                    rotate = int(stream["tags"]["rotate"])

    video_info = VideoInfo(
        sar=sar,
        dar=dar,
        sample_width=width,
        sample_height=height,
        codec=codec,
        fps=fps,
        audio_sample_rate=audio_sample_rate,
        audio_channel_layout=channel_layout,
        rotate=rotate,
        unstructured_stream_data=data,
    )
    return video_info
