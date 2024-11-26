# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
A data object for storing user options, along with utilities for parsing input options
from command line flags and a configuration file.
"""

import argparse
import datetime
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ego4d.cli.universities import BUCKET_TO_UNIV, UNIV_TO_BUCKET


@dataclass
class ValidatedConfig:
    """
    Data object that stores validated user-supplied configuration options for a video
    download operation.
    """

    input_directory: str
    validate_all: bool
    metadata_folder: str
    input_university: str
    output_dir: str
    aws_profile_name: str
    num_workers: int
    expiry_time_sec: int
    version: str
    skip_mp4_check: bool
    released_video_path: Optional[str] = None


@dataclass
class Config:
    """
    Data object that stores the user-supplied configuration options for a video download
    operation.
    """

    input_directory: str
    validate_all: bool
    metadata_folder: str
    num_workers: int
    expiry_time_sec: int
    aws_profile_name: str
    version: str
    skip_mp4_check: bool
    released_video_path: Optional[str] = None
    input_university: Optional[str] = None
    output_dir: Optional[str] = None


def validate_config(cfg: Config) -> ValidatedConfig:
    """

    Args:
        cfg: A user-supplied configuration for the download operation
    Returns:
        A ValidatedConfig if all user-supplied options appear valid
    """

    def _maybe_fix_s3_folder(path: str) -> str:
        if path.startswith("s3") and not path.endswith("/"):
            return path + "/"
        return path

    if cfg.input_directory.startswith("s3://"):
        if cfg.input_university is not None:
            raise AssertionError("please do not provide a university name")

        bucket = cfg.input_directory.split("s3://")[1].split("/")[0]
        cfg.input_university = BUCKET_TO_UNIV[bucket]

    if cfg.input_university not in set(UNIV_TO_BUCKET.keys()):
        raise AssertionError(f"{cfg.input_university} is not a valid university")

    if cfg.aws_profile_name != "default":
        raise AssertionError(
            "aws profile!=default: other profiles not currently supported"
        )

    if cfg.output_dir is None:
        time_now = f"{datetime.datetime.now().strftime('%d%m%y_%H%M%S')}"
        cfg.output_dir = f"s3://ego4d-consortium-sharing/internal/validation/{cfg.input_university}/{time_now}/"  # noqa
        print(f"Using output directory: {cfg.output_dir}")

    if cfg.metadata_folder is None:
        cfg.metadata_folder = (
            "ego4d/internal/validation/standard_metadata/egoexo"
            if cfg.version == "egoexo"
            else "ego4d/internal/validation/standard_metadata/ego4d/"
        )

    return ValidatedConfig(
        input_directory=_maybe_fix_s3_folder(cfg.input_directory),
        validate_all=bool(cfg.validate_all),
        metadata_folder=_maybe_fix_s3_folder(cfg.metadata_folder),
        released_video_path=cfg.released_video_path,
        input_university=cfg.input_university,
        output_dir=_maybe_fix_s3_folder(cfg.output_dir),
        aws_profile_name=cfg.aws_profile_name,
        num_workers=cfg.num_workers,
        expiry_time_sec=cfg.expiry_time_sec,
        version=cfg.version,
        skip_mp4_check=cfg.skip_mp4_check,
    )


def config_from_args(args=None) -> Config:
    """
    Parses command line flags and returns a Config object with corresponding values from
    the flags.
    """
    # Parser for a configuration file
    json_parser = argparse.ArgumentParser(
        description="Command line tool to download Ego4D datasets from Amazon S3"
    )

    json_parser.add_argument(
        "--config_path",
        type=Path,
        help="Local path to a config JSON file. If specified, the flags will be read "
        "from this file instead of the command line.",
    )

    args, remaining = json_parser.parse_known_args(args=args)

    # Parser for command line flags other than the configuration file
    flag_parser = argparse.ArgumentParser(add_help=False)

    # required_flags = {"output_directory"}
    flag_parser.add_argument(
        "-i",
        "--input_directory",
        help="The S3 path where the university's video metadata is stored",
    )

    flag_parser.add_argument(
        "-a",
        "--all",
        default=False,
        help="validate all files in S3",
        dest="validate_all",
    )
    flag_parser.add_argument(
        "-mf",
        "--metadata_folder",
        help="The S3 path where the device/component_type/scenario metadata is stored",
        default=None,
    )
    flag_parser.add_argument(
        "-rp",
        "--released_video_path",
        help="The path where released_videos file is stored",
        default=None,
    )
    flag_parser.add_argument(
        "-u",
        "--input_university",
        help="The university name we're checking data on",
        default=None,
    )
    flag_parser.add_argument(
        "-o",
        "--output_dir",
        help="output directory",
    )
    flag_parser.add_argument(
        "-nw",
        "--num_workers",
        type=int,
        help="number of workers",
        default=10,
    )
    flag_parser.add_argument(
        "-expiry",
        "--expiry_time_sec",
        type=int,
        help="default expiry for presigned URLs (in seconds)",
        default=3600,
    )
    flag_parser.add_argument(
        "--aws_profile_name",
        help="Defaults to 'default'. Specifies the AWS profile name from "
        "~/.aws/credentials to use for the download",
        default="default",
    )
    flag_parser.add_argument(
        "--version",
        help="EgoExo validation or Ego4D validation?",
        default="egoexo",
    )
    flag_parser.add_argument(
        "--skip_mp4_check",
        default=False,
        help="skip checking mp4 files with ffprobe",
        action="store_true",
    )

    # Use the values in the config file, but set them as defaults to flag_parser so they
    # can be overridden by command line flags
    if args.config_path:
        with open(args.config_path.expanduser()) as f:
            config_contents = json.load(f)
            flag_parser.set_defaults(**config_contents)

    parsed_args = flag_parser.parse_args(remaining)

    flags = {k: v for k, v in vars(parsed_args).items()}

    # Note: Since the flags from the config file are being used as default argparse
    # values, we can't set required=True. Doing so would mean that the user couldn't
    # leave them unspecified when invoking the CLI.
    config = Config(**flags)

    return config
