# pyre-unsafe

import json
import random
import subprocess
import time
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

from ego4d.internal.validation.manifest import Error, ErrorLevel


@dataclass(frozen=True)
class VideoInfo:
    fps: Fraction
    sar: Fraction
    dar: Optional[Fraction]
    sample_width: int
    sample_height: int
    vcodec: Optional[str] = None
    acodec: Optional[str] = None
    vstart: Optional[float] = None
    astart: Optional[float] = None
    vduration: Optional[float] = None
    aduration: Optional[float] = None
    mp4_duration: Optional[float] = None
    video_time_base: Optional[Fraction] = None
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


def get_video_info(
    filename: str,
    name: Optional[str] = None,
) -> Tuple[Optional[VideoInfo], Optional[Error]]:
    """
    Return:
        VideoInfo: information of the video stored in bucket_name specified by the object_name
        If error, returns None and log the error.
    """
    name = name if name is not None else filename

    cmd = [
        "ffprobe",
        "-i",
        f"{filename}",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        "-hide_banner",
    ]

    def to_fraction(fps):
        if fps is None:
            return 0
        a, b = fps.split("/")
        if int(b) == 0:
            return 0
        return Fraction(int(a), int(b))

    result = None
    x = 1
    for _ in range(10):
        time.sleep(random.random() * x + x / 2)
        try:
            result = subprocess.run(cmd, encoding="utf-8", capture_output=True)
        except Exception as ex:
            return None, Error(
                ErrorLevel.ERROR,
                name,
                "ffmpeg_cannot_read_error",
                f"{ex}",
            )

        if result.stderr:
            if "tls" in result.stderr:
                x *= 2.5
                continue
            return None, Error(
                ErrorLevel.ERROR,
                name,
                "ffmpeg_cannot_read_error",
                f"{result.stderr}",
            )

        break

    assert result is not None

    # return result
    data = json.loads(result.stdout)

    fps = None
    width = None
    height = None
    sar = Fraction(1, 1)
    dar = None
    vcodec = None
    acodec = None
    vstart = None
    astart = None
    vduration = None
    aduration = None
    audio_sample_rate = None
    channel_layout = None
    rotate = None
    vtb = None
    mp4_duration = None

    if "format" in data:
        if "duration" in data["format"]:
            mp4_duration = float(data["format"]["duration"])

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

            if "codec_name" in stream:
                if "width" in stream:
                    vcodec = stream["codec_name"]
                    width = stream["width"]
                else:
                    acodec = stream["codec_name"]

            if "height" in stream:
                height = stream["height"]

            if "duration" in stream:
                if "width" in stream:
                    vduration = float(stream["duration"])
                else:
                    aduration = float(stream["duration"])

            if "start_time" in stream:
                if "width" in stream:
                    vstart = float(stream["start_time"])
                else:
                    astart = float(stream["start_time"])

            if "time_base" in stream and "width" in stream:
                vtb = to_fraction(stream["time_base"])

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
        vcodec=vcodec,
        acodec=acodec,
        vduration=vduration,
        aduration=aduration,
        vstart=vstart,
        astart=astart,
        fps=fps,
        audio_sample_rate=audio_sample_rate,
        audio_channel_layout=channel_layout,
        rotate=rotate,
        unstructured_stream_data=data,
        video_time_base=vtb,
        mp4_duration=mp4_duration,
    )
    return video_info, None
