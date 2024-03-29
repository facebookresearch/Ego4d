import os
import random
import time
import traceback
from dataclasses import dataclass
from typing import Any, List, Optional

import boto3
import boto3.session
import botocore.client as bclient
import botocore.exceptions
from iopath import PathManager

S3Client = bclient.BaseClient


@dataclass
class S3FileDesc:
    basename: str
    path: str
    size: int


def _get_location(bucket_name: str) -> str:
    client = boto3.client("s3")
    response = client.get_bucket_location(Bucket=bucket_name)
    return response["LocationConstraint"]


def get_config(
    num_workers: int = 10,
    connect_timeout: int = 180,
    max_attempts: int = 3,
    region_name: Optional[str] = None,
):
    config = bclient.Config(
        region_name=region_name,
        connect_timeout=connect_timeout,
        max_pool_connections=num_workers,
        retries={"mode": "standard", "max_attempts": max_attempts},
    )
    return config


def get_session(profile: Optional[str]):
    return boto3.session.Session(profile_name=profile)


def get_client(
    bucket_name: str,
    num_workers: int = 10,
    connect_timeout: int = 180,
    max_attempts: int = 3,
    profile: Optional[str] = None,
) -> S3Client:
    session = get_session(
        profile=profile,
    )
    return session.client(
        "s3",
        config=get_config(
            region_name=_get_location(bucket_name),
            num_workers=num_workers,
            connect_timeout=connect_timeout,
            max_attempts=max_attempts,
        ),
    )


def get_resource(
    profile: str,
    num_workers: int = 10,
    connect_timeout: int = 180,
    max_attempts: int = 3,
) -> Any:
    session = get_session(profile=profile)
    return session.resource(
        "s3",
        config=get_config(
            region_name=None,
            num_workers=num_workers,
            connect_timeout=connect_timeout,
            max_attempts=max_attempts,
        ),
    )


class StreamPathMgr:
    def __init__(self, expiration_sec: int = 7200):
        self.clients = {}
        self.expr_sec = expiration_sec
        self.cached_paths = {}

    def open(self, path: str) -> str:
        if path in self.cached_paths:
            return self.cached_paths[path]

        if path.startswith("s3"):
            temp = path.split("s3://")[1].split("/")
            bucket_name = temp[0]
            object_name = "/".join(temp[1:])

            client = None
            if bucket_name not in self.clients:
                self.clients[bucket_name] = get_client(bucket_name)

            client = self.clients[bucket_name]
            ret = client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket_name, "Key": object_name},
                ExpiresIn=self.expr_sec,
            )
            self.cached_paths[path] = ret
            return ret
        else:
            return path


def ls_relative(path: str, pathmgr: PathManager) -> List[str]:
    files = pathmgr.ls(path)
    return [f.split("/")[-1] for f in files]


# TOOD(miguelmartin): removeme, duplicate function
def exp_backoff(max_sleep_time_sec=1200, base=2):  # pyre-ignore
    def decorator(func):  # pyre-ignore
        assert base > 1

        def wrapper(*args, **kwargs):  # pyre-ignore
            sleep_t = base
            while True:
                try:
                    ret = func(*args, **kwargs)
                    return ret
                except Exception as e:
                    print(
                        f"WARN: exp_backoff retrying...: {func.__name__}, {traceback.format_exc()}"
                    )
                    time.sleep(sleep_t / base + sleep_t * random.random())
                    sleep_t *= base
                    if sleep_t > max_sleep_time_sec:
                        print(
                            f"ERROR: exp_backoff FAILED: {func.__name__}, {traceback.format_exc()}"
                        )
                        raise e
                    continue

        return wrapper

    return decorator


# NOTE:
# iopath / pathmgr not used as we cannot get an S3 Object for sizing with it
class S3Downloader:
    def __init__(self, profile, num_workers=10, callback=None):
        self.clients = {}
        self.resources = get_resource(profile=profile, num_workers=num_workers)
        self.profile = profile
        self.callback = callback

    def ls(self, path, recursive=False, max_keys=-1) -> List[S3FileDesc]:
        assert path.startswith("s3://")
        temp = path.split("s3://")[1].split("/")
        bucket_name = temp[0]
        object_name = "/".join(temp[1:])
        if bucket_name not in self.clients:
            self.clients[bucket_name] = get_client(
                bucket_name,
                profile=self.profile,
            )  # pyre-ignore

        delim = "/" if not recursive else ""
        client = self.clients[bucket_name]
        if max_keys < 0:
            paginator = client.get_paginator("list_objects_v2")
            pages = paginator.paginate(
                Bucket=bucket_name, Prefix=object_name, Delimiter=delim
            )
            return [
                S3FileDesc(
                    basename=os.path.basename(f["Key"]),
                    path=f"s3://{bucket_name}/{f['Key']}",
                    size=f["Size"],
                )
                for page in pages
                for f in page.get("Contents", [])
            ]
        else:
            assert max_keys <= 1000
            ls_result = client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=object_name,
                MaxKeys=max_keys,
                Delimiter="/",
            )
            if ls_result["KeyCount"] == 0:
                return []
            return [
                S3FileDesc(
                    basename=os.path.basename(f["Key"]),
                    path=f"s3://{bucket_name}/{f['Key']}",
                    size=f["Size"],
                )
                for f in ls_result["Contents"]
            ]

    def copy(self, path: str, out_path: str):
        self.obj(path).download_file(out_path, Callback=self.callback)

    def file_desc(self, path: str) -> Optional[S3FileDesc]:
        o = self.obj(path)
        if o is None:
            return None
        try:
            file_size = o.content_length
            ret = S3FileDesc(
                basename=os.path.basename(o.key),
                path=path,
                size=file_size,
            )
        except botocore.exceptions.ClientError as e:
            # If a client error is thrown, then check that it was a 404 error.
            # If it was a 404 error, then the file does not exist.
            error_code = int(e.response["Error"]["Code"])
            if error_code == 404:
                return None
            else:
                raise e
        return ret

    def obj(self, path):
        assert path.startswith("s3://")
        temp = path.split("s3://")[1].split("/")
        bucket_name = temp[0]
        object_name = "/".join(temp[1:])

        try:
            return self.resources.Object(bucket_name, object_name)
        except botocore.exceptions.ClientError as e:
            # If a client error is thrown, then check that it was a 404 error.
            # If it was a 404 error, then the file does not exist.
            error_code = int(e.response["Error"]["Code"])
            if error_code == 404:
                return None
            else:
                raise e
