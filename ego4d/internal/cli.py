#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
Command line tool to download Ego4D datasets.

Examples:
      python -m ego4d.internal.cli \
        -i "s3://ego4d-unict/metadata_v7" \
        -mf "s3://ego4d-consortium-sharing/internal/standard_metadata_v10/" \
        -ed "error_details" \
        -es "error_summary" \
"""
import boto3
import botocore.client as bclient
from ego4d.cli.universities import UNIV_TO_BUCKET
from ego4d.internal.config import (
    Config,
    config_from_args,
    meta_path,
    unis,
    validate_config,
)
from ego4d.internal.validate import validate_all


def _get_location(bucket_name: str) -> str:
    client = boto3.client("s3")
    response = client.get_bucket_location(Bucket=bucket_name)
    return response["LocationConstraint"]


def main_cfg(cfg: Config) -> None:

    validated_cfg = validate_config(cfg)

    # This service resource in the default session will be used for general light-weight
    # requests on the main thread, such as downloading the video manifests and getting
    # S3 object metadata
    if cfg.validate_all:
        for u in unis:
            bucket = UNIV_TO_BUCKET[u]
            path = f"s3://{bucket}/{meta_path[u]}"
            s3 = boto3.client(
                "s3",
                config=bclient.Config(
                    region_name=_get_location(bucket),
                    connect_timeout=180,
                    max_pool_connections=validated_cfg.num_workers,
                    retries={"total_max_attempts": 3},
                ),
            )
            validate_all(
                path,
                s3,
                validated_cfg.metadata_folder,
                validated_cfg.error_details_name,
                validated_cfg.error_summary_name,
                validated_cfg.num_workers,
                validated_cfg.expiry_time_sec,
            )
    else:
        input_dir = validated_cfg.input_directory
        assert "s3://" in input_dir
        bucket = input_dir.split("://")[1].split("/")[0]

        s3 = boto3.client(
            "s3",
            config=bclient.Config(
                region_name=_get_location(bucket),
                connect_timeout=180,
                max_pool_connections=validated_cfg.num_workers,
                retries={"total_max_attempts": 3},
            ),
        )
        validate_all(
            input_dir,
            s3,
            validated_cfg.metadata_folder,
            validated_cfg.error_details_name,
            validated_cfg.error_summary_name,
            validated_cfg.num_workers,
            validated_cfg.expiry_time_sec,
        )


def main() -> None:
    config = config_from_args()
    main_cfg(config)


if __name__ == "__main__":
    main()
