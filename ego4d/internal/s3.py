from typing import List

import boto3
import botocore.client as bclient
from iopath import PathManager

DEFAULT_NUM_WORKERS = 32


S3Client = bclient.BaseClient


def _get_location(bucket_name: str) -> str:
    client = boto3.client("s3")
    response = client.get_bucket_location(Bucket=bucket_name)
    return response["LocationConstraint"]


def get_client(
    bucket_name: str,
    num_workers: int,
    connect_timeout: int = 180,
    max_attempts: int = 3,
) -> S3Client:
    return boto3.client(
        "s3",
        config=bclient.Config(
            region_name=_get_location(bucket_name),
            connect_timeout=connect_timeout,
            max_pool_connections=num_workers,
            retries={"total_max_attempts": max_attempts},
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
                self.clients[bucket_name] = get_client(bucket_name, DEFAULT_NUM_WORKERS)

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
