#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
Command line tool to download Ego4D datasets.

Examples:
      python ego4d/internal/validation/cli.py \
        -i "s3://ego4d-unict/metadata_v7" \
        -mf ego4d/internal/validation/standard_metadata/ego4d

      python ego4d/internal/validation/cli.py \
        -i "s3://ego4d-georgiatech/metadata_v5" \
        -mf ego4d/internal/validation/standard_metadata/ego4d \
        -u georgiatech \
        -o errors
"""

import boto3
import botocore.client as bclient
from ego4d.cli.universities import UNIV_TO_BUCKET
from ego4d.internal.s3 import get_client
from ego4d.internal.validation.config import Config, config_from_args, validate_config
from ego4d.internal.validation.validate import run_validation


def main_cfg(cfg: Config) -> None:
    validated_cfg = validate_config(cfg)
    run_validation(
        manifest_dir=validated_cfg.input_directory,
        input_university=validated_cfg.input_university,
        released_video_path=validated_cfg.released_video_path,
        standard_metadata_folder=validated_cfg.metadata_folder,
        output_dir=validated_cfg.output_dir,
        num_workers=validated_cfg.num_workers,
        expiry_time_sec=validated_cfg.expiry_time_sec,
        version=validated_cfg.version,
        skip_mp4_check=validated_cfg.skip_mp4_check,
    )


def main() -> None:
    config = config_from_args()
    main_cfg(config)


if __name__ == "__main__":
    main()
