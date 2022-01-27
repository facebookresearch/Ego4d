# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
Functionality related to parsing the video manifest and storing an in-memory
representation for the download operation.
"""
import csv
from enum import Enum
import logging
from pathlib import Path
import re
from typing import Dict, Set, Iterable

from ego4d.cli.s3path import bucket_and_key_from_path
from ego4d.cli.universities import BUCKET_TO_UNIV

__MANIFEST_BUCKET = "ego4d-consortium-sharing"


class VideoMetadata:
    """
    Data object that corresponds to a single video entry in a manifest CSV file.
    """

    __VIDEO_UID_KEY = "video_uid"
    __S3_LOCATION_KEY = "canonical_s3_location"
    __FILE_TYPE_KEY = "type"
    __BENCHMARKS_KEY = "benchmarks"

    def __init__(self, row: Dict[str, str]):
        # The raw contents of the CSV row
        self.raw_data: Dict[str, str] = dict(row)

        # Unique identifier for the vido
        self.uid: str = row[self.__VIDEO_UID_KEY]

        # Path to the video file on AWS S3 (e.g. "s3://bucket/key")
        self.s3_path: str = row[self.__S3_LOCATION_KEY]

        # Name of the S3 bucket that holds the video
        self.s3_bucket: str = None

        # S3 object key for the video
        self.s3_object_key: str = None

        if self.s3_path:
            self.s3_bucket, self.s3_object_key = bucket_and_key_from_path(self.s3_path)
            self.university: str = BUCKET_TO_UNIV.get(self.s3_bucket, "")

        type = row.get(self.__FILE_TYPE_KEY)
        if not type or type in ["mp4", "video"]:
            self.file_download = False
        else:
            assert type in ["file", "json"]
            self.file_download = True

        benchmarks = row.get(self.__BENCHMARKS_KEY)
        if benchmarks:
            self.benchmarks = re.sub(r"\s+", "", benchmarks.lower())
        else:
            self.benchmarks = None


def list_videos_in_manifest(
    manifest_path: Path, benchmarks: Set[str], for_universities: Set[str]
) -> Iterable[VideoMetadata]:
    """
    Creates a generator that reads every row of a manifest CSV file and returns the row
    as a VideoMetadata object.

    Args:
        manifest_path: Path on local disk to a manifest CSV file
        for_universities: Only videos belonging to universities in this set will be
            returned. If the set is empty then all videos will be returned.
    """
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)

        has_benchmarks = False
        if len(benchmarks) > 0:
            if "benchmarks" in reader.fieldnames:
                has_benchmarks = True
                benchmarks = [x.lower() for x in benchmarks]
                b_re = re.compile(r'\[(\w+)?(?:\,(\w+))*\]', re.IGNORECASE)
                print(f"Filtering by benchmarks: {benchmarks}")
            else:
                print("Benchmarks specified but ignored without a benchmarks field in manifest.")
        
        for row in reader:
            metadata = VideoMetadata(row)
            if has_benchmarks:
                # print(f"row benchmarks: {metadata.benchmarks}")
                if not metadata.benchmarks:
                    continue
                m = b_re.match(metadata.benchmarks)
                if not m:
                    if metadata.benchmarks:
                        logging.warning(f"Invalid benchmarks manifest entry ignored: {metadata.benchmarks}")
                    continue
                # print(f"row benchmark groups: {m.groups()}")
                # assert all(x is not None for x in m.groups())
                if not any(x in benchmarks for x in m.groups()):
                    continue
            if for_universities and metadata.university not in for_universities:
                continue
            yield metadata


def download_manifest_for_version(
    version: str, dataset: str, download_dir: Path, s3
) -> Path:
    """
    Downloads the manifest file to the download_path as a file named "manifest.csv"

    Args:
        version:
        dataset:
        download_dir:
        s3 (S3.ServiceResource):

    Returns:
    """
    download_path = download_dir / "manifest.csv"
    _manifest_object(version, dataset, s3).download_file(str(download_path))
    return download_path


def _manifest_object(version: str, dataset: str, s3):
    """

    Args:
        version:
        dataset:
        s3 (S3.ServiceResource):

    Returns:
    S3.Object
    """
    return s3.Bucket(__MANIFEST_BUCKET).Object(
        f"public/{version}/{dataset}/manifest.csv"
    )
