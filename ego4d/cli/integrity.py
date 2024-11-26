# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
Verifies integrity of downloaded files
"""

from typing import Collection, List

from ego4d.cli.download import VideoOnDisk


def list_corrupt_files(downloads: Collection[VideoOnDisk]) -> List[VideoOnDisk]:
    """
    Returns a list of any downloaded files that appear corrupted.
    """
    return [d for d in downloads if _file_is_corrupt(d)]


def _file_is_corrupt(download: VideoOnDisk):
    if download.file_path.exists():
        return download.file_path.stat().st_size != download.s3_content_size_bytes

    return True
