#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
Command line tool to download Ego4D datasets.

Examples:
      python -m ego4d.internal.cli \
        -i "s3://ego4d-unict/metadata_v7" \
        -mf "./ego4d/internal/standard_metadata_v10" \
        -ed "error_details" \
        -es "error_summary" \
"""
import boto3
from ego4d.cli.universities import UNIV_TO_BUCKET
from ego4d.internal.config import (
    Config,
    config_from_args,
    meta_path,
    unis,
    validate_config,
)
from ego4d.internal.validate import validate_all


def main_cfg(cfg: Config) -> None:

    validated_cfg = validate_config(cfg)

    # This service resource in the default session will be used for general light-weight
    # requests on the main thread, such as downloading the video manifests and getting
    # S3 object metadata
    # s3 = boto3.session.Session(profile_name=validated_cfg.aws_profile_name).resource(
    #     "s3"
    # )
    s3 = boto3.client("s3")
    if cfg.validate_all:
        for u in unis:
            path = f"s3://{UNIV_TO_BUCKET[u]}/{meta_path[u]}"
            validate_all(
                path,
                s3,
                validated_cfg.metadata_folder,
                validated_cfg.error_details_name,
                validated_cfg.error_summary_name,
            )
    else:
        validate_all(
            validated_cfg.input_directory,
            s3,
            validated_cfg.metadata_folder,
            validated_cfg.error_details_name,
            validated_cfg.error_summary_name,
        )


def main() -> None:
    config = config_from_args()
    main_cfg(config)


if __name__ == "__main__":
    main()
