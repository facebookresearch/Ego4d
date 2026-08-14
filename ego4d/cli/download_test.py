# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import csv
from pathlib import Path

import pytest

from .config import ValidatedConfig
from .download import list_videos_for_download


UID_1 = "000cd456-ff8d-499b-b0c1-4acead128a8b"
UID_2 = "111cd456-ff8d-499b-b0c1-4acead128a8b"
UID_3 = "222cd456-ff8d-499b-b0c1-4acead128a8b"
UNKNOWN_UID = "fffcd456-ff8d-499b-b0c1-4acead128a8b"


def _config(tmp_path: Path, dataset: str, video_uids) -> ValidatedConfig:
    return ValidatedConfig(
        output_directory=tmp_path,
        assume_yes=False,
        version="v2_1",
        datasets={dataset},
        benchmarks=set(),
        aws_profile_name="default",
        metadata=False,
        manifest=False,
        bypass_version_check=False,
        skip_s3_checks=False,
        video_uids=set(video_uids),
        universities=set(),
        annotations=True,
        list_datasets=False,
        manifest_override_path=None,
    )


def _write_manifest(path: Path, fieldnames, rows) -> Path:
    with path.open("w", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_video_uids_filter_video_dataset(tmp_path, capsys):
    manifest_path = _write_manifest(
        tmp_path / "manifest.csv",
        ["video_uid", "canonical_s3_location"],
        [
            {
                "video_uid": UID_1,
                "canonical_s3_location": f"s3://bucket/{UID_1}.mp4",
            },
            {
                "video_uid": UID_2,
                "canonical_s3_location": f"s3://bucket/{UID_2}.mp4",
            },
        ],
    )

    videos = list_videos_for_download(
        _config(tmp_path, "full_scale", [UID_1]),
        "full_scale",
        manifest_path,
    )

    assert [video.uid for video in videos] == [UID_1]
    assert "A total of 1 file will be downloaded" in capsys.readouterr().out


def test_video_uids_filter_manifest_with_explicit_video_uids(tmp_path):
    manifest_path = _write_manifest(
        tmp_path / "manifest.csv",
        ["file_uid", "video_uid", "s3_path", "type"],
        [
            {
                "file_uid": "feature-1",
                "video_uid": UID_1,
                "s3_path": f"s3://bucket/{UID_1}.pt",
                "type": "file",
            },
            {
                "file_uid": "feature-2",
                "video_uid": UID_2,
                "s3_path": f"s3://bucket/{UID_2}.pt",
                "type": "file",
            },
        ],
    )

    videos = list_videos_for_download(
        _config(tmp_path, "custom_features", [UID_1]),
        "custom_features",
        manifest_path,
    )

    assert [video.uid for video in videos] == ["feature-1"]
    assert [video.video_uid for video in videos] == [UID_1]


def test_video_uids_filter_multiple_feature_entries(tmp_path):
    manifest_path = _write_manifest(
        tmp_path / "manifest.csv",
        ["video_uid", "s3_path", "type"],
        [
            {
                "video_uid": uid,
                "s3_path": f"s3://bucket/{uid}.pt",
                "type": "file",
            }
            for uid in [UID_1, UID_2, UID_3]
        ],
    )

    videos = list_videos_for_download(
        _config(
            tmp_path,
            "slowfast8x8_r101_k400",
            [UID_1, UID_3],
        ),
        "slowfast8x8_r101_k400",
        manifest_path,
    )

    assert {video.video_uid for video in videos} == {UID_1, UID_3}


def test_unknown_video_uid_for_addressable_file_dataset_raises(tmp_path):
    manifest_path = _write_manifest(
        tmp_path / "manifest.csv",
        ["video_uid", "s3_path", "type"],
        [
            {
                "video_uid": UID_1,
                "s3_path": f"s3://bucket/{UID_1}.pt",
                "type": "file",
            }
        ],
    )

    with pytest.raises(RuntimeError) as exc_info:
        list_videos_for_download(
            _config(
                tmp_path,
                "slowfast8x8_r101_k400",
                [UNKNOWN_UID],
            ),
            "slowfast8x8_r101_k400",
            manifest_path,
        )

    assert UNKNOWN_UID in str(exc_info.value)
    assert "could not be found" in str(exc_info.value)


def test_video_uids_do_not_filter_file_uid_only_dataset(tmp_path, capsys):
    manifest_path = _write_manifest(
        tmp_path / "manifest.csv",
        ["file_uid", "s3_path", "type"],
        [
            {
                "file_uid": UID_1,
                "s3_path": "s3://bucket/annotations.json",
                "type": "file",
            },
            {
                "file_uid": "metadata",
                "s3_path": "s3://bucket/metadata.json",
                "type": "file",
            },
        ],
    )

    videos = list_videos_for_download(
        _config(tmp_path, "annotations", [UID_1]),
        "annotations",
        manifest_path,
    )
    output = capsys.readouterr().out

    assert [video.uid for video in videos] == [UID_1, "metadata"]
    assert "the filter will be ignored" in output
    assert f"['{UID_1}']" in output
    assert "Only downloading a subset" not in output


def test_no_video_uid_filter_preserves_all_manifest_entries(tmp_path, capsys):
    manifest_path = _write_manifest(
        tmp_path / "manifest.csv",
        ["file_uid", "s3_path", "type"],
        [
            {
                "file_uid": "annotations",
                "s3_path": "s3://bucket/annotations.json",
                "type": "file",
            },
            {
                "file_uid": "metadata",
                "s3_path": "s3://bucket/metadata.json",
                "type": "file",
            },
        ],
    )

    videos = list_videos_for_download(
        _config(tmp_path, "annotations", []),
        "annotations",
        manifest_path,
    )

    assert [video.uid for video in videos] == ["annotations", "metadata"]
    assert capsys.readouterr().out == ""
