# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
Functionality related to parsing the video manifest and storing an in-memory
representation for the download operation.
"""

import csv
import datetime
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Set

import regex

from ego4d.cli.s3path import bucket_and_key_from_path
from ego4d.cli.universities import BUCKET_TO_UNIV

__MANIFEST_BUCKET = "ego4d-consortium-sharing"
__METADATA_FILENAME = "ego4d.json"
__DATASETS_FILENAME = "datasets.csv"


class VideoMetadata:
    """
    Data object that corresponds to a single video entry in a manifest CSV file.
    """

    __FILE_UID_KEY = "file_uid"
    __VIDEO_UID_KEY = "video_uid"
    __S3_LOCATION_KEYS = ["canonical_s3_location", "s3_path"]
    __FILE_TYPE_KEY = "type"
    __BENCHMARKS_KEY = "benchmarks"

    def __init__(self, row: Dict[str, str]):
        # The raw contents of the CSV row
        self.raw_data: Dict[str, str] = dict(row)

        # Unique identifier for the video
        if self.__FILE_UID_KEY in row:
            self.file_download = True
            self.uid: str = row[self.__FILE_UID_KEY]
        else:
            assert (
                self.__VIDEO_UID_KEY in row
            ), "Either file_uid or video_uid must be specified"
            self.file_download = False
            self.uid: str = row[self.__VIDEO_UID_KEY]

        # Path to the video file on AWS S3 (e.g. "s3://bucket/key")
        for x in self.__S3_LOCATION_KEYS:
            if x in row:
                self.s3_path: str = row[x]
                break

        # Name of the S3 bucket that holds the video
        self.s3_bucket: str = None

        # S3 object key for the video
        self.s3_object_key: str = None

        if self.s3_path:
            self.s3_bucket, self.s3_object_key = bucket_and_key_from_path(self.s3_path)
            self.university: str = BUCKET_TO_UNIV.get(self.s3_bucket, "")

        type = row.get(self.__FILE_TYPE_KEY)
        if type in ["mp4", "video"]:
            self.file_download = False
        elif type in ["file", "json"]:
            self.file_download = True
        else:
            # Default to above
            pass

        if self.file_download:
            self.filename_base = os.path.basename(self.s3_path)
        else:
            self.filename_base = f"{self.uid}.mp4"

        benchmarks = row.get(self.__BENCHMARKS_KEY)
        if benchmarks:
            self.benchmarks = regex.sub(r"\s+", "", benchmarks.lower())
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
                b_re = regex.compile(r"\[(\w+)?(?:\|(\w+))*\]", regex.IGNORECASE)
                print(f"Filtering by benchmarks: {benchmarks}")
            else:
                print(
                    "Benchmarks specified but ignored without a benchmarks field in manifest."
                )

        for row in reader:
            metadata = VideoMetadata(row)
            if has_benchmarks:
                if not metadata.benchmarks:
                    continue
                m = b_re.match(metadata.benchmarks)
                if not m:
                    if metadata.benchmarks:
                        logging.warning(
                            f"Invalid benchmarks manifest entry ignored: {metadata.benchmarks}"
                        )
                    continue
                grps = m.captures(1) + m.captures(2)
                cnt_bars = metadata.benchmarks.count("|")
                if cnt_bars > 0:
                    assert len(grps) == (
                        cnt_bars + 1
                    ), f"Invalid benchmarks row: {metadata.benchmarks}"
                else:
                    assert (
                        len(grps) <= 1
                    ), f"Invalid benchmarks row: {metadata.benchmarks}"
                if not any(x in benchmarks for x in grps):
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


def _metadata_object(version: str, s3):
    """
    The primary metadata JSON
    """
    return s3.Bucket(__MANIFEST_BUCKET).Object(
        f"public/{version}/{__METADATA_FILENAME}"
    )


def download_metadata(version: str, download_dir: Path, s3) -> Path:
    """
    Downloads the primary metadata JSON to the download_path
    """
    download_path = download_dir / __METADATA_FILENAME
    if download_path.exists():
        # TODO: Check for file version
        return download_path
    else:
        print("Downloading Ego4D metadata json..")

    _metadata_object(version, s3).download_file(str(download_path))
    return download_path


def _datasets_object(version: str, s3):
    """
    The primary metadata JSON
    """
    return s3.Bucket(__MANIFEST_BUCKET).Object(
        f"public/{version}/{__DATASETS_FILENAME}"
    )


def download_datasets(version: str, download_dir: Path, s3) -> Path:
    """
    Downloads the primary datasets csv to the download_path
    """
    download_path = download_dir / __DATASETS_FILENAME
    if download_path.exists():
        mtime = datetime.datetime.fromtimestamp(
            download_path.stat().st_mtime, tz=datetime.timezone.utc
        )
        delta = (datetime.utcnow() - mtime).totalseconds() / 3600
        if delta < 12:
            print("Bypassing recent datasets.csv..")
            return download_path
    else:
        print("Downloading datasets.csv..")

    _datasets_object(version, s3).download_file(str(download_path))
    return download_path


def print_datasets(version: str, s3) -> None:
    assert version
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            p = download_datasets(version, tmppath, s3)
            if not p.exists():
                logging.error("datasets.csv download failed!")
                print("Download datasets.csv error (defaulting to local)..")
                p = Path(__file__).with_name("datasets.csv")
            with p.open("r") as f:
                rows = csv.DictReader(f)
                print("\nAvailable Ego4D datasets:")
                for row in rows:
                    print(f"   {row['dataset']:<21}\t{row['description']}")
                print()
    except Exception as ex:
        logging.exception(f"Exception retrieving Ego4D datasets: {ex}")
