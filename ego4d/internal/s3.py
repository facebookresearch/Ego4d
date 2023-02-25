from typing import Any

import boto3
import botocore.client as bclient

DEFAULT_NUM_WORKERS = 32


def _get_location(bucket_name: str) -> str:
    client = boto3.client("s3")
    response = client.get_bucket_location(Bucket=bucket_name)
    return response["LocationConstraint"]


def get_client(
    bucket_name: str,
    num_workers: int,
    connect_timeout: int = 180,
    max_attempts: int = 3,
) -> Any:
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
    def __init__(self):
        self.clients = {}

    def open(self, path: str, expiration_sec: int = 7200) -> str:
        if path.startswith("s3"):
            temp = path.split("s3://")[1].split("/")
            bucket_name = temp[0]
            object_name = "/".join(temp[1:])

            client = None
            if bucket_name not in self.clients:
                self.clients[bucket_name] = get_client(bucket_name, DEFAULT_NUM_WORKERS)

            client = self.clients[bucket_name]
            return client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket_name, "Key": object_name},
                ExpiresIn=expiration_sec,
            )
        else:
            return path
