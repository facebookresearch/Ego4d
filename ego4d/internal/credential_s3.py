# pyre-unsafe
import logging
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import botocore


@dataclass
class FileInfo:
    key: str
    size: int


def is_file_readable(file_path):
    return os.path.isfile(file_path) and os.access(file_path, os.R_OK)


def parse_bucket_key(key: str) -> Tuple[str, str]:
    # Determine from the key
    assert key and key.startswith(
        "s3://"
    ), f"Invalid key without bucket supplied: {key}"
    m = re.match("^s3://([^/]*)/(.*)$", key)
    assert m, f"Invalid s3:// search key: {key}"
    grp = m.groups()
    assert len(grp) == 2
    return grp[0], grp[1]


class S3Helper:
    _bucket_name = None

    def __init__(self, s3, bucket_name):
        self._bucket_name = bucket_name
        # self._s3
        self._s3 = s3

    @property
    def bucket(self):
        return self._bucket_name

    def ls(self, prefix: str, max_results=-1, **kwargs) -> Tuple[bool, List[FileInfo]]:
        if max_results < 0:
            paginator = self._s3.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self._bucket_name, Prefix=prefix)
            results = []
            for page in pages:
                for f in page["Contents"]:
                    results.append(FileInfo(f["Key"], f["Size"]))
            return False, results
        else:
            # legacy
            ls_result = self._s3.list_objects_v2(
                Bucket=self._bucket_name, Prefix=prefix, MaxKeys=max_results, **kwargs
            )
            return (
                ls_result["IsTruncated"],
                [FileInfo(f["Key"], f["Size"]) for f in ls_result["Contents"]],
            )

    def get_file(self, key: str, local_path: str, **kwargs) -> Optional[str]:
        print(f"Downloading: {key} to {local_path}")
        try:
            self._s3.download_file(
                Bucket=self._bucket_name, Key=key, Filename=local_path, **kwargs
            )
        except botocore.exceptions.ClientError as e:
            # If a client error is thrown, then check that it was a 404 error.
            # If it was a 404 error, then the file does not exist.
            error_code = int(e.response["Error"]["Code"])
            if error_code == 404:
                return None
            else:
                raise e
        return local_path

    def exists(self, key: str, bucket: Optional[str] = None, **kwargs) -> bool:
        if not bucket:
            bucket, key = parse_bucket_key(key)
            assert key and bucket
        try:
            obj = self._s3.get_object(Bucket=bucket, Key=key)
            return bool(obj)
        except botocore.exceptions.ClientError as e:
            logging.exception("AWS get_size exception: ", e)
            raise e
