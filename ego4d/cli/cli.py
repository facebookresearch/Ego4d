#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
Command line tool to download Ego4D datasets.

Examples:
      python -m ego4d.cli.cli \
        --version="v1" \
        --datasets full_scale annotations \
        --output_directory="~/ego4d_data"
"""
import logging
import os
from typing import List

import boto3
from ego4d.cli.config import (
    Config,
    config_from_args,
    DATASET_PRIMARY,
    DATASETS_VIDEO,
    validate_config,
)
from ego4d.cli.download import (
    create_download_directory,
    create_output_directory,
    download_all,
    FileToDownload,
    filter_already_downloaded,
    list_corrupt_files,
    list_videos_for_download,
    load_version_file,
    save_version_file,
    upsert_version,
)
from ego4d.cli.manifest import download_manifest_for_version, download_metadata
from ego4d.cli.progressbar import DownloadProgressBar
from tqdm import tqdm


def main_cfg(cfg: Config) -> None:

    validated_cfg = validate_config(cfg)

    # This service resource in the default session will be used for general light-weight
    # requests on the main thread, such as downloading the video manifests and getting
    # S3 object metadata
    s3 = boto3.session.Session(profile_name=validated_cfg.aws_profile_name).resource(
        "s3"
    )

    if cfg.video_uids and all(x not in DATASETS_VIDEO for x in validated_cfg.datasets):
        logging.error(
            "ERROR: video_uids specified for non-video datasets (and will be ignored)"
        )

    output_path = create_output_directory(validated_cfg)
    print(f"Download Path: {output_path}\n")

    # Download the primary metadata to the root directory
    if cfg.metadata:
        metadata_path = download_metadata(
            validated_cfg.version,
            validated_cfg.output_directory,
            s3,
        )
        if not metadata_path:
            logging.error("ERROR: Primary Metadata Download Failed")
        else:
            print(f"Ego4D Metadata: {metadata_path}")

    # Download the manifest to the root directory
    if cfg.manifest:
        toplevel_manifest_path = download_manifest_for_version(
            validated_cfg.version,
            DATASET_PRIMARY,
            validated_cfg.output_directory,
            s3,
        )
        if not toplevel_manifest_path:
            logging.error("ERROR: Primary Manifest Download Failed")
            print("ABORT: Primary Manifest Download Failed")
            return

        print(f"Manifest downloaded: {toplevel_manifest_path}")

    downloads: List[FileToDownload] = []
    print("Checking requested datasets and versions...")
    for dataset in validated_cfg.datasets:
        if dataset != dataset.lower():
            dataset = dataset.lower()
            print(f"ERROR: Please specify datasets as lower case.  Assuming: {dataset}")

        download_path = create_download_directory(validated_cfg, dataset)
        print(
            f"Created download directory for version: '{validated_cfg.version}' of "
            f"dataset: '{dataset}' at:\n{download_path}\n"
        )

        manifest_path = download_manifest_for_version(
            validated_cfg.version, dataset, download_path, s3
        )

        version_entries = load_version_file(download_path)

        if validated_cfg.video_uids:
            print(
                "Only downloading a subset of the video files because the "
                "'video_uids' flag has been set on the command line or in the config "
                f"file. A total of {len(validated_cfg.video_uids)} video files will "
                f"be downloaded.\n"
            )
        to_download = list_videos_for_download(validated_cfg, dataset, manifest_path)
        downloads.extend(
            [FileToDownload.create(video, download_path) for video in to_download]
        )

    print("Retrieving object metadata from S3...")
    cnt_invalid = 0
    for x in tqdm(downloads, unit="object"):
        if not x.s3_object_key:
            cnt_invalid += 1
            continue
        x.s3_object = s3.Object(x.s3_bucket, x.s3_object_key)

    if cnt_invalid == len(downloads):
        print("ABORT: All S3 Objects Invalid")
        return
    elif cnt_invalid > 0:
        logging.error(
            f"{cnt_invalid}/{len(downloads)} invalid S3 downloads will be ignored"
        )

    print("Checking if latest file versions are already downloaded...")
    active_downloads = filter_already_downloaded(
        downloads,
        version_entries,
        bypass_version_check=validated_cfg.bypass_version_check,
    )

    missing = [x for x in downloads if not x.s3_exists]
    if len(missing) > 0:
        logging.error(
            f"{len(missing)}/{len(downloads)} missing S3 downloads will be ignored"
        )

    if len(active_downloads) == 0:
        print(
            "The latest versions of all requested videos already exist in the output "
            "directories under:\n"
            f"{validated_cfg.output_directory}"
        )
        exit(0)

    print(f"Downloading {len(active_downloads)}/{len(downloads)}..")

    assert all(x.s3_object for x in active_downloads)

    total_size_bytes = sum(
        x.s3_object.content_length for x in active_downloads if x.s3_object
    )
    if total_size_bytes == 0:
        print(
            "The latest versions of all requested videos already exist in the output "
            "directories under:\n"
            f"{validated_cfg.output_directory}"
        )
        exit(0)

    expected_gb = total_size_bytes / 1024 / 1024 / 1024
    if validated_cfg.assume_yes:
        print(f"Downloading {expected_gb:.1f} GB..")
    else:
        confirm = None
        while confirm is None:
            response = input(
                f"Expected size of downloaded files is "
                f"{expected_gb:.1f} GB. "
                f"Do you want to start the download? ([y]/n) "
            )
            if response.lower() in ["yes", "y", ""]:
                confirm = True
            elif response.lower() in ["no", "n"]:
                print("Aborting the download operation.")
                exit(0)
            else:
                continue

    files = download_all(
        active_downloads,
        aws_profile_name=validated_cfg.aws_profile_name,
        callback=DownloadProgressBar(total_size_bytes=total_size_bytes).update,
    )

    # TODO: Handle/filter? failed download pre-corrupted?

    print("Checking file integrity...")
    corrupted = list_corrupt_files(files)

    if corrupted:
        msg = f"ERROR: {len(corrupted)} files failed download: "
        for x in corrupted:
            msg += f"\t{x.uid}\n"
        logging.error(msg)
        # TODO: retry these downloads?

    # TODO: Only succeeded/non-corrupted files (currently warns in upsert)
    for x in active_downloads:
        upsert_version(x, version_entries)

    save_version_file(version_entries, download_path)

    # TODO: Add logging functionality and store to hidden directory in download
    # folder


def main() -> None:
    config = config_from_args()
    main_cfg(config)


if __name__ == "__main__":
    main()
