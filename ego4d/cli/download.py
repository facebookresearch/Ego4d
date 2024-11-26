# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
Functionality related to downloading objects from S3.
"""

import csv
import datetime
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields
from itertools import compress
from pathlib import Path
from typing import (
    Any,
    Callable,
    Collection,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import boto3.session
import botocore.exceptions
from botocore.client import Config
from ego4d.cli import manifest
from ego4d.cli.config import DATASET_FILE_EXTENSIONS, DATASETS_VIDEO, ValidatedConfig
from ego4d.cli.manifest import VideoMetadata
from tqdm import tqdm


__VERSION_ENTRY_FILENAME = "manifest.ver"
SAVE_INTERVAL_S = 15


@dataclass
class VersionEntry:
    uid: str
    version: str
    filename: str


@dataclass
class FileToDownload:
    """Data object that tracks a video to be downloaded to a specific path"""

    filename: str
    download_folder: Path
    uid: str
    s3_bucket: str = None
    s3_object_key: str = None
    s3_object: Any = None
    s3_exists: Optional[bool] = None
    # Size of the file on S3. This should match the file size on disk.
    s3_content_size_bytes: int = 0
    s3_version: str = None
    # Location of the local file downloaded.
    file_path: Optional[Path] = None

    def exists(self) -> bool:
        if not self.s3_object:
            raise RuntimeError(f"No s3_object for exists call: {self.uid}")
        try:
            self.s3_object.load()
            return True
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            if e.response["Error"]["Code"] == "403":
                print(f"Boto 403 Exception For exists: {self.uid} | {self.filename}")
                return False
            print(f"Boto Unexpected Exception For exists: {self.uid} | {self.filename}")
            raise

    def file_version_base(self) -> str:
        if not self.filename:
            raise RuntimeError("Invalid filename for file_version")
        base, ext = os.path.splitext(self.filename)
        if ext not in DATASET_FILE_EXTENSIONS:
            logging.warning(
                f"Unexpected file_version extension: {ext} filename: {self.filename}"
            )
        assert "/" not in base
        return base

    def file_version_name(self, version: str) -> str:
        """Creates a file_version_name for a video that is being downloaded"""
        base = self.file_version_base()
        return f"{base}.{version}"

    def file_version_pattern(self) -> str:
        base = self.file_version_base()
        return f"{base}.*"

    def to_version_entry(self) -> VersionEntry:
        return VersionEntry(
            uid=self.uid, version=self.s3_version, filename=self.filename
        )

    @staticmethod
    def create(video: VideoMetadata, download_folder: Path) -> "FileToDownload":
        assert video.filename_base, f"VideoMetadata missing filename_base: {video.uid}"

        x = FileToDownload(
            filename=video.filename_base,
            download_folder=download_folder,
            uid=video.uid,
        )
        if video.s3_bucket and video.s3_object_key:
            x.s3_bucket = video.s3_bucket
            x.s3_object_key = video.s3_object_key
        return x


@dataclass
class VideoOnDisk:
    """
    Data object representing a video that has been downloaded from S3 to the local hard
    disk.
    """

    # Path to the file on disk
    file_path: Path

    # Size of the file on S3. This should match the file size on disk.
    s3_content_size_bytes: int


def create_output_directory(validated_cfg: ValidatedConfig) -> Path:
    """
    Creates a top level download directory if it does not already exist, and returns
    the Path to the download directory.
    """
    # only use the major for the output dir, i.e. v2_1 => v2
    download_path = validated_cfg.output_directory / f"{validated_cfg.out_version_dir}"
    download_path.mkdir(parents=True, exist_ok=True)
    return download_path


def create_download_directory(validated_cfg: ValidatedConfig, dataset: str) -> Path:
    """
    Creates a download directory for a dataset if it does not already exist, and returns
    the Path to the download directory.
    """
    download_path = (
        validated_cfg.output_directory / f"{validated_cfg.out_version_dir}/{dataset}"
    )
    download_path.mkdir(parents=True, exist_ok=True)
    return download_path


# def list_s3_objects(downloads: Union[Collection[FileToDownload], tqdm], s3):
#     """
#     Takes a list of videos to be downloaded (optionally wrapped by tqdm to show a
#     progress bar) and returns a corresponding list of S3.Objects for each video that is
#     to be downloaded.

#     Args:
#         downloads:
#         s3 (S3.ServiceResource):

#     Returns:
#     List[S3.Object]
#     """
#     for d in downloads:
#         if not d.s3_object_key:
#             continue

#     return [s3.Object(d.s3_bucket, d.s3_object_key) for d in downloads if d.s3_object_key]


def info(msg):
    logging.info(msg)
    print(msg)


def filter_already_downloaded(
    downloads: Iterable[FileToDownload],
    version_entries: List[VersionEntry],
    bypass_version_check: bool = False,
    skip_s3_checks: bool = False,
) -> List[FileToDownload]:
    """
    Takes a collection of files that are to be downloaded and a list of the S3.Objects
    corresponding to the files to download and removes any that have already been
    downloaded.
    """

    def already_downloaded(download: FileToDownload) -> bool:
        assert download.filename
        # file_version_name = download.file_version_name(s3_object.version_id)
        # assert file_version_name

        download.s3_exists = (
            skip_s3_checks or download.exists()
        )  # shortcircuits for a faster initial download
        if not download.s3_exists:
            info(
                f"Missing s3 object (ignored for download): {download.uid} | {download.filename}"
            )
            return False

        file_location = download.download_folder / download.filename
        # file_version_location = download.download_folder / file_version_name
        if not file_location.exists():
            logging.info(f"already_downloaded: missing file: {file_location}")
            return False

        if not download.s3_object:
            logging.error(
                f"filter_already_downloaded: invalid s3 object: {download.uid}"
            )
            return False

        if bypass_version_check:
            return True

        version_entry = next(
            (x for x in version_entries if x.uid == download.uid), None
        )
        if not version_entry:
            info(
                f"already_downloaded: no version entry for existing file: {file_location}"
            )
            return False

        s3_version = download.s3_object.version_id
        if version_entry.version != s3_version:
            info(
                f"filter_already_downloaded: mismatched s3 object version: {download.uid} "
                f" {version_entry.version} v {s3_version} "
            )
            return False

        s3_size = download.s3_object.content_length
        # Return true if the size on S3 matches the file size on disk
        if file_location.stat().st_size != s3_size:
            logging.warning(f"already_downloaded=False for file size: {file_location}")
            return False

        return True

    with ThreadPoolExecutor(max_workers=15) as pool:
        to_download = list(
            tqdm(
                pool.map(
                    lambda x: x.s3_object and not already_downloaded(x) and x.s3_exists,
                    downloads,
                ),
                total=len(downloads),
                unit="file",
            )
        )

    n_filtered = len(downloads) - sum(to_download)
    if n_filtered > 0:
        info(f"Filtered {n_filtered}/{len(downloads)} existing videos for download.")
    else:
        info("No existing videos to filter.")

    return list(compress(downloads, to_download))


def download_all(
    downloads: Collection[FileToDownload],
    entries: List[VersionEntry],
    aws_profile_name: str,
    callback: Callable[[int], None] = None,
    save_callback: Callable[[], None] = None,
) -> List[FileToDownload]:
    thread_data = threading.local()
    lock_update = threading.Lock()
    last_save = datetime.datetime.utcnow()

    def initializer():
        config = Config(
            connect_timeout=120,
            retries={"mode": "standard", "max_attempts": 5},
        )
        thread_data.s3 = boto3.session.Session(profile_name=aws_profile_name).resource(
            "s3",
            config=config,
        )

    def download_video(download: FileToDownload):
        nonlocal last_save

        assert download.filename
        assert download.s3_bucket
        assert download.s3_object_key

        obj = thread_data.s3.Object(download.s3_bucket, download.s3_object_key)
        download.s3_version = obj.version_id

        # file_version_name = download.file_version_name(obj.version_id)
        file_path = download.download_folder / download.filename

        # TODO: Can remove for ship
        # Remove any existing version files
        old_version_files = list(
            download.download_folder.glob(download.file_version_pattern())
        )
        for file in old_version_files:
            # file.unlink()
            os.unlink(str(file))

        # Remove the old video file (if it exists)
        if os.path.exists(file_path):
            # file_path.unlink(missing_ok=True)
            os.unlink(str(file_path))

        # Create an empty file with the version name so that it can be tracked later
        # (download.download_folder / file_version_name).touch()

        try:
            obj.download_file(str(file_path), Callback=callback)

            download.file_path = file_path
            download.s3_content_size_bytes = obj.content_length

            with lock_update:
                now = datetime.datetime.utcnow()
                upsert_version(download, entries)
                if save_callback:
                    delta_s = (now - last_save).total_seconds()
                    if delta_s > SAVE_INTERVAL_S:
                        logging.debug("Incremental save..")
                        save_callback()
                        last_save = now

        except Exception as ex:
            logging.exception(f"S3 Download Exception: {ex}")
            download.file_path = None
            download.s3_content_size_bytes = 0

        return download

    # Note: The download operation will be I/O bound by network bandwidth, not compute
    # bound, so a thread pool executor can be used instead of a process pool executor.
    with ThreadPoolExecutor(max_workers=8, initializer=initializer) as pool:
        results = list(pool.map(download_video, downloads))

    return results


def list_videos_for_download(
    cfg: ValidatedConfig, dataset: str, manifest_path: Path
) -> List[VideoMetadata]:
    """
    Takes a user-supplied configuration and a video manifest from S3 then returns a
    list of all videos from the manifest that the user has requested.

    Args:
        cfg:
        manifest_path:
    """
    generator = manifest.list_videos_in_manifest(
        manifest_path, cfg.benchmarks, cfg.universities
    )

    videos = list(generator)

    if cfg.video_uids:
        if any(x != x.lower() for x in cfg.video_uids):
            raise RuntimeError(
                "ERROR: Upper case uids invalid - please sanity check your inputs: {cfg.video_uids}"
            )
        matches = [x for x in videos if x.uid in cfg.video_uids]
        if dataset in DATASETS_VIDEO or dataset.lower() in DATASETS_VIDEO:
            missing = cfg.video_uids - {v.uid for v in videos}
            if missing:
                raise RuntimeError(
                    "The following requested video UIDs could not be found "
                    f"in the manifest for version: '{cfg.version}' and "
                    f"dataset(s): '{cfg.datasets}':\n"
                    f"{missing}"
                )

            videos = matches
        else:
            if matches:
                print(
                    "ERROR: video_uids not supported for non-video datasets: {[x.uid for x in matches]}"
                )

    return videos


def list_corrupt_files(downloads: Collection[FileToDownload]) -> List[FileToDownload]:
    """
    Returns a list of any downloaded files that appear corrupted.
    """
    return [d for d in downloads if _file_is_corrupt(d)]


def _file_is_corrupt(download: FileToDownload):
    if download.file_path and download.file_path.exists():
        return download.file_path.stat().st_size != download.s3_content_size_bytes

    return True


def load_version_file(download_path: Path) -> List[VersionEntry]:
    file_path = download_path / __VERSION_ENTRY_FILENAME
    if not os.path.exists(file_path):
        return []
    with open(file_path) as f:
        reader = csv.DictReader(f)
        data = list(reader)
        entries = [
            VersionEntry(row["uid"], row["version"], row["filename"]) for row in data
        ]
    return entries


def save_version_file(entries: List[VersionEntry], download_path: Path):
    file_path = download_path / __VERSION_ENTRY_FILENAME
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [field.name for field in fields(VersionEntry)]
        writer = csv.DictWriter(f, fieldnames)
        writer.writeheader()
        writer.writerows(
            {"uid": x.uid, "version": x.version, "filename": x.filename}
            for x in entries
        )


def upsert_version(download: FileToDownload, entries: List[VersionEntry]):
    assert download and download.uid

    if download.s3_content_size_bytes == 0:
        # download failed, just ignore?
        logging.warning("upsert_version: ignoring 0 byte download: " + download.uid)
        return

    matches = [x for x in entries if x.uid == download.uid]
    if len(matches) == 0:
        entries.append(download.to_version_entry())
    else:
        assert (
            len(matches) == 1
        ), f"Multiple version entries for uid invalid: {download.uid}"
        entry = matches[0]
        if download.s3_version:
            entry.version = download.s3_version
        if download.filename != entry.filename:
            logging.error(
                f"Suspect version info: Filename ({download.filename}) changed for existing entry: {entry.filename}"
            )
            entry.filename = download.filename
