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
from ego4d.internal.s3 import get_client
from ego4d.internal.validation.config import (
    Config,
    config_from_args,
    meta_path,
    unis,
    validate_config,
)
from ego4d.internal.validation.validate import validate_all


def main_cfg(cfg: Config) -> None:

    validated_cfg = validate_config(cfg)

    # This service resource in the default session will be used for general light-weight
    # requests on the main thread, such as downloading the video manifests and getting
    # S3 object metadata
    if cfg.validate_all:
        for u in unis:
            bucket = UNIV_TO_BUCKET[u]
            path = f"s3://{bucket}/{meta_path[u]}"
            s3 = get_client(bucket, validated_cfg.num_workers)
            validate_all(
                path,
                s3,
                u,
                validated_cfg.released_video_path,
                validated_cfg.metadata_folder,
                validated_cfg.error_details_name,
                validated_cfg.error_summary_name,
                validated_cfg.num_workers,
                validated_cfg.expiry_time_sec,
            )
    else:
        input_dir = validated_cfg.input_directory
        u = validated_cfg.input_university

        s3 = None
        if "s3://" in input_dir:
            bucket = input_dir.split("://")[1].split("/")[0]
            s3 = get_client(
                bucket_name=bucket,
                num_workers=validated_cfg.num_workers,
                connect_timeout=validated_cfg.expiry_time_sec,
                max_attempts=10,
            )

        validate_all(
            input_dir,
            s3,
            validated_cfg.input_university,
            validated_cfg.released_video_path,
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
