# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import csv
import tempfile
from pathlib import Path

import pytest

from .manifest import list_videos_in_manifest, VideoMetadata
from .universities import BUCKET_TO_UNIV


def test_constructor():
    row = {
        "video_uid": "12345",
        "canonical_s3_location": "s3://ego4d-georgiatech/object/key/123",
    }

    metadata = VideoMetadata(row=row)

    # Known university
    assert metadata.uid == "12345"
    assert metadata.s3_path == "s3://ego4d-georgiatech/object/key/123"
    assert metadata.s3_bucket == "ego4d-georgiatech"
    assert metadata.s3_object_key == "object/key/123"
    assert metadata.university == BUCKET_TO_UNIV["ego4d-georgiatech"]

    # Unknown university
    row["canonical_s3_location"] = "s3://unrecognized-university/object/key/123"
    # Metadata should copy the row
    assert (
        metadata.raw_data["canonical_s3_location"]
        == "s3://ego4d-georgiatech/object/key/123"
    )

    unknown_univ = VideoMetadata(row=row)
    assert unknown_univ.s3_path == "s3://unrecognized-university/object/key/123"
    assert unknown_univ.s3_bucket == "unrecognized-university"
    assert unknown_univ.s3_object_key == "object/key/123"
    assert not unknown_univ.university


def test_csv_reader():
    with tempfile.NamedTemporaryFile(mode="w") as f:
        writer = csv.DictWriter(f, fieldnames=["video_uid", "canonical_s3_location"])
        writer.writeheader()
        writer.writerow(
            {
                "video_uid": "12345",
                "canonical_s3_location": "s3://ego4d-georgiatech/object/key/123",
            }
        )
        writer.writerow(
            {
                "video_uid": "7890",
                "canonical_s3_location": "s3://unrecognized-university/other/key/456",
            }
        )
        f.flush()

        full_generator = list_videos_in_manifest(
            manifest_path=Path(f.name), for_universities=set(), benchmarks=set()
        )
        georgiatech_generator = list_videos_in_manifest(
            manifest_path=Path(f.name),
            for_universities={BUCKET_TO_UNIV["ego4d-georgiatech"]},
            benchmarks=set(),
        )

        # Full generator should return both videos
        metadata = next(full_generator)
        assert metadata.uid == "12345"
        assert metadata.s3_path == "s3://ego4d-georgiatech/object/key/123"
        assert metadata.s3_bucket == "ego4d-georgiatech"
        assert metadata.s3_object_key == "object/key/123"
        assert metadata.university == BUCKET_TO_UNIV["ego4d-georgiatech"]

        metadata = next(full_generator)
        assert metadata.uid == "7890"
        assert metadata.s3_path == "s3://unrecognized-university/other/key/456"
        assert metadata.s3_bucket == "unrecognized-university"
        assert metadata.s3_object_key == "other/key/456"
        assert not metadata.university

        with pytest.raises(StopIteration):
            next(full_generator)

        # Georgia Tech should only return one video
        metadata = next(georgiatech_generator)
        assert metadata.uid == "12345"
        assert metadata.s3_path == "s3://ego4d-georgiatech/object/key/123"
        assert metadata.s3_bucket == "ego4d-georgiatech"
        assert metadata.s3_object_key == "object/key/123"
        assert metadata.university == BUCKET_TO_UNIV["ego4d-georgiatech"]

        with pytest.raises(StopIteration):
            next(georgiatech_generator)


def test_csv_benchmarks():
    with tempfile.NamedTemporaryFile(mode="w") as f:
        writer = csv.DictWriter(
            f, fieldnames=["video_uid", "canonical_s3_location", "benchmarks"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "video_uid": "12345",
                "canonical_s3_location": "s3://ego4d-georgiatech/object/key/123",
                "benchmarks": "[FHO |EM]",
            }
        )
        writer.writerow(
            {
                "video_uid": "7890",
                "canonical_s3_location": "s3://unrecognized-university/other/key/456",
                "benchmarks": "",
            }
        )
        writer.writerow(
            {
                "video_uid": "6789",
                "canonical_s3_location": "s3://unrecognized-university/other/key/456",
                "benchmarks": "[]",
            }
        )
        writer.writerow(
            {
                "video_uid": "23456",
                "canonical_s3_location": "s3://unrecognized-university/other/key/456",
                "benchmarks": "[ EM]",
            }
        )
        f.flush()

        full_generator = list_videos_in_manifest(
            manifest_path=Path(f.name), for_universities=set(), benchmarks=set()
        )
        em_generator = list_videos_in_manifest(
            manifest_path=Path(f.name),
            for_universities=set(),
            benchmarks=set(["em"]),
        )

        # Full generator should return both videos
        metadata = next(full_generator)
        assert metadata.uid == "12345"
        assert metadata.s3_path == "s3://ego4d-georgiatech/object/key/123"
        assert metadata.s3_bucket == "ego4d-georgiatech"
        assert metadata.s3_object_key == "object/key/123"
        assert metadata.university == BUCKET_TO_UNIV["ego4d-georgiatech"]

        metadata = next(full_generator)
        assert metadata.uid == "7890"
        assert metadata.s3_path == "s3://unrecognized-university/other/key/456"
        assert metadata.s3_bucket == "unrecognized-university"
        assert metadata.s3_object_key == "other/key/456"
        assert not metadata.university

        metadata = next(full_generator)
        assert metadata.uid == "6789"

        metadata = next(full_generator)
        assert metadata.uid == "23456"

        with pytest.raises(StopIteration):
            next(full_generator)

        # EM should only return two videos
        metadata = next(em_generator)
        assert metadata.uid == "12345"

        metadata = next(em_generator)
        assert metadata.uid == "23456"

        with pytest.raises(StopIteration):
            next(em_generator)
