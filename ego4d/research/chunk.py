# pyre-strict
import json
import os
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterator, List, Optional

import av
import numpy as np
import torch
from iopath.common.file_io import PathManager


@dataclass
class VideoChunk:
    """
    Normalized from 0-255 (uint8). Shape == (t, c, h, w)
    """

    video_frames: np.array

    """
    Not normalized from 0-1, shape yet to be normalized
    """
    audio_frames: Optional[np.array]

    video_frames_pts: List[int]
    audio_frames_pts: Optional[List[int]]

    video_frames_sec: List[Fraction]
    audio_frames_sec: Optional[List[Fraction]]

    video_timebase: Fraction
    audio_timebase: Fraction
    audio_sample_rate: int
    video_fps: Fraction


def split_av_frames(
    video_path: str,
    window_size_sec: int = 5,
    subsample_n_frames: Optional[int] = None,
    limit: int = -1,
) -> Iterator[VideoChunk]:
    with av.open(video_path) as container:
        has_audio = len(container.streams.audio) > 0
        has_audio = False

        streams_to_decode = {"video": 0}
        video_tb = container.streams.video[0].time_base
        video_fps = container.streams.video[0].framerate
        audio_tb = None
        audio_sample_rate = None
        if has_audio:
            streams_to_decode["audio"] = 0
            audio_tb = container.streams.audio[0].time_base
            audio_sample_rate = container.streams.audio[0].sample_rate

        vf_buffer = {}
        af_buffer = {}
        curr_window_start_sec = 0
        curr_window_end_sec = curr_window_start_sec + window_size_sec
        i = 0
        for frame in container.decode(**streams_to_decode):
            t_pts_sec = frame.pts * frame.time_base
            if t_pts_sec >= curr_window_end_sec:
                i += 1
                curr_window_start_sec = curr_window_start_sec + window_size_sec
                curr_window_end_sec = curr_window_end_sec + window_size_sec
                if len(vf_buffer) > 0 or len(af_buffer) > 0:
                    vf_frames_sorted = sorted(vf_buffer.items(), key=lambda x: x[0])
                    af_frames_sorted = sorted(af_buffer.items(), key=lambda x: x[0])
                    vf_frames_np = np.stack(
                        [x.to_ndarray(format="rgb24") for _, x in vf_frames_sorted]
                    ).transpose(0, 3, 1, 2)  # THWC -> TCHW
                    assert vf_frames_np.dtype == np.uint8
                    indices = None
                    if subsample_n_frames is not None:
                        # ref: https://fburl.com/code/a8viadi2
                        t = vf_frames_np.shape[0]
                        indices = torch.linspace(0, t - 1, subsample_n_frames)
                        indices = torch.clamp(indices, 0, t - 1).long()
                        vf_frames_np = vf_frames_np[indices]
                        indices = set(indices.tolist())

                    # NOTE: do we want to normalize audio from 0-1 as well?
                    af_frames_np = (
                        np.stack([x.to_ndarray() for _, x in af_frames_sorted])
                        if len(af_frames_sorted) > 0
                        else None
                    )
                    vf_frames_pts = [
                        x[0]
                        for i, x in enumerate(vf_frames_sorted)
                        if indices is None or i in indices
                    ]
                    af_frames_pts = (
                        [x[0] for x in vf_frames_sorted]
                        if len(af_frames_sorted) > 0
                        else None
                    )
                    # pyre-ignore
                    yield VideoChunk(
                        video_frames=vf_frames_np,
                        audio_frames=af_frames_np,
                        video_frames_pts=vf_frames_pts,
                        audio_frames_pts=af_frames_pts,
                        video_frames_sec=[x * video_tb for x in vf_frames_pts],
                        audio_frames_sec=(
                            [x * audio_tb for x in af_frames_pts]
                            if af_frames_pts is not None
                            else None
                        ),
                        video_timebase=video_tb,
                        audio_timebase=audio_tb,
                        video_fps=video_fps,
                        audio_sample_rate=audio_sample_rate,
                    )
                vf_buffer = {}
                af_buffer = {}
            else:
                if t_pts_sec < curr_window_start_sec:
                    raise AssertionError("packets from previous window")

            if isinstance(frame, av.AudioFrame):
                af_buffer[frame.pts] = frame
            else:
                assert isinstance(frame, av.VideoFrame)
                vf_buffer[frame.pts] = frame

            if limit > 0 and i >= limit:
                break


def save_chunk(chunk: VideoChunk, out_dir: str, pathmgr: Optional[PathManager]) -> None:
    if pathmgr is None:
        pathmgr = PathManager()

    has_audio = chunk.audio_frames is not None
    is_audio_mono = None
    if has_audio:
        is_audio_mono = (
            len(chunk.audio_frames.shape) == 1 or chunk.audio_frames.shape[1] == 1
        )
        audio_path = os.path.join(out_dir, "audio.npy")

    video_path = os.path.join(out_dir, "video.npy")
    metadata_path = os.path.join(out_dir, "metadata.json")

    metadata = {
        "video_frame_pts": chunk.video_frames_pts,
        "audio_frame_pts": chunk.audio_frames_pts,
        "video_frame_sec": [float(x) for x in chunk.video_frames_sec],
        "audio_frame_sec": (
            [float(x) for x in chunk.audio_frames_sec] if has_audio else None
        ),
        "video_timebase_numerator": chunk.video_timebase.numerator,
        "video_timebase_denom": chunk.video_timebase.denominator,
        "audio_timebase_numerator": (
            chunk.audio_timebase.numerator if chunk.audio_timebase is not None else None
        ),
        "audio_timebase_denom": (
            chunk.audio_timebase.denominator
            if chunk.audio_timebase is not None
            else None
        ),
        "has_audio": has_audio,
        "num_frames": chunk.video_frames.shape[0],
        "num_audio_frames": chunk.audio_frames.shape[0] if has_audio else 0,
        "frame_height": chunk.video_frames.shape[2],
        "frame_width": chunk.video_frames.shape[3],
        "is_audio_mono": is_audio_mono,
        "video_fps_numeratator": chunk.video_fps.numerator,
        "video_fps_denom": chunk.video_fps.denominator,
        "audio_sample_rate": chunk.audio_sample_rate,
    }
    json.dump(metadata, pathmgr.open(metadata_path, "w"), indent=2)

    # TODO: is torch.save better?
    np.save(pathmgr.open(video_path, "wb"), chunk.video_frames)
    if has_audio:
        np.save(pathmgr.open(audio_path, "wb"), chunk.audio_frames)
