# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
Functionality for parsing AWS S3 paths.

References:
    https://docs.aws.amazon.com/AmazonS3/latest/userguide/access-bucket-intro.html#accessing-a-bucket-using-S3-format
    https://docs.aws.amazon.com/AmazonS3/latest/userguide/bucketnamingrules.html
    https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-keys.html

"""

import re
from typing import Tuple


__S3_PATH_REGEX = re.compile(r"^s3://(?P<bucket>[^/]*)/(?P<key>.*)$")


def bucket_and_key_from_path(path: str) -> Tuple[str, str]:
    """
    Takes an S3 path (i.e. s3://<bucket-name>/<object-key>) and returns the bucket and
    object key as a string tuple.

    Notes:
        This does not validate whether or not the bucket and key satisfy the naming and
        character requirements imposed by AWS (e.g character limits on bucket names).
    """
    match = __S3_PATH_REGEX.match(path)
    return match.group("bucket"), match.group("key")
