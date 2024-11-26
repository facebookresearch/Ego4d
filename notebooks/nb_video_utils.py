# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

from fractions import Fraction

import av


def pts_to_time_seconds(pts: int, base: Fraction) -> Fraction:
    return pts * base


def frame_index_to_pts(frame: int, start_pt: int, diff_per_frame: int) -> int:
    return start_pt + frame * diff_per_frame


def pts_difference_per_frame(fps: Fraction, time_base: Fraction) -> int:
    pt = (1 / fps) * (1 / time_base)
    assert pt.denominator == 1, "should be whole number"
    return int(pt)


def _get_frames_pts(
    video_pts_set,  # this has type List[int]
    container: av.container.Container,
    include_audio: bool,
    include_additional_audio_pts: int,
):  # -> Iterable[av.frame.Frame]
    assert len(container.streams.video) == 1

    min_pts = min(video_pts_set)
    max_pts = max(video_pts_set)
    video_pts_set = set(video_pts_set)  # for O(1) lookup

    video_stream = container.streams.video[0]
    fps: Fraction = video_stream.average_rate
    video_base: Fraction = video_stream.time_base
    video_pt_diff = pts_difference_per_frame(fps, video_base)

    # [start, end) time
    clip_start_sec = pts_to_time_seconds(min_pts, video_base)
    clip_end_sec = pts_to_time_seconds(max_pts, video_base)

    # add some additional time for audio packets
    clip_end_sec += max(
        pts_to_time_seconds(include_additional_audio_pts, video_base), 1 / fps
    )

    # --- setup
    streams_to_decode = {"video": 0}
    if (
        include_audio
        and container.streams.audio is not None
        and len(container.streams.audio) > 0
    ):
        assert len(container.streams.audio) == 1
        streams_to_decode["audio"] = 0
        audio_base: Fraction = container.streams.audio[0].time_base

    # seek to the point we need in the video
    # with some buffer room, just in-case the seek is not precise
    seek_pts = max(0, min_pts - 2 * video_pt_diff)
    video_stream.seek(seek_pts)
    if "audio" in streams_to_decode:
        assert len(container.streams.audio) == 1
        audio_stream = container.streams.audio[0]
        audio_seek_pts = int(seek_pts * video_base / audio_base)
        audio_stream.seek(audio_seek_pts)

    # --- iterate over video

    # used for validation
    previous_video_pts = None
    previous_audio_pts = None

    for frame in container.decode(**streams_to_decode):
        if isinstance(frame, av.AudioFrame):
            assert include_audio
            # ensure frames are in order
            assert previous_audio_pts is None or previous_audio_pts < frame.pts
            previous_audio_pts = frame.pts

            # pyre-fixme[61]: `audio_base` may not be initialized here.
            audio_time_sec = pts_to_time_seconds(frame.pts, audio_base)

            # we want all the audio frames in this region
            if audio_time_sec >= clip_start_sec and audio_time_sec < clip_end_sec:
                yield frame
            elif audio_time_sec >= clip_end_sec:
                break

        elif isinstance(frame, av.VideoFrame):
            video_time_sec = pts_to_time_seconds(frame.pts, video_base)
            if video_time_sec >= clip_end_sec:
                break

            # ensure frames are in order
            assert previous_video_pts is None or previous_video_pts < frame.pts

            if frame.pts in video_pts_set:
                # check that the frame is in range
                assert (
                    video_time_sec >= clip_start_sec and video_time_sec < clip_end_sec
                ), f"""
                video frame at time={video_time_sec} (pts={frame.pts})
                out of range for time [{clip_start_sec}, {clip_end_sec}]
                """

                yield frame


def _get_frames(
    video_frames,  # this has type List[int]
    container: av.container.Container,
    include_audio: bool,
    audio_buffer_frames: int = 0,
):  # -> Iterable[av.frame.Frame]
    assert len(container.streams.video) == 1

    video_stream = container.streams.video[0]
    video_start: int = video_stream.start_time
    video_base: Fraction = video_stream.time_base
    fps: Fraction = video_stream.average_rate
    video_pt_diff = pts_difference_per_frame(fps, video_base)

    audio_buffer_pts = (
        frame_index_to_pts(audio_buffer_frames, 0, video_pt_diff)
        if include_audio
        else 0
    )

    time_pts_set = [
        frame_index_to_pts(f, video_start, video_pt_diff) for f in video_frames
    ]
    return _get_frames_pts(time_pts_set, container, include_audio, audio_buffer_pts)
