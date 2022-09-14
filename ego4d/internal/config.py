# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
A data object for storing user options, along with utilities for parsing input options
from command line flags and a configuration file.
"""
import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Set

import boto3.session
from botocore.exceptions import ProfileNotFound
from ego4d.cli.universities import UNIV_TO_BUCKET

unis = [
    "unict",
    "cmu",
    "iiith",
    "minnesota",
    "utokyo",
    "kaust",
    "indiana",
    "nus",
    "bristol",
    "georgiatech",
    "frl_track_1_public",
    "cmu_africa",
    "uniandes",
]
meta_path = [
    "metadata_v5",
    "metadata_v27",
    "2021-08-29_august_fixed",
    "metadata_v10",
    "metadata_v3",
    "metadata_v7",
    "metadata-v4",
    "metadata_v1.2",
    "metadata-v3",
    "metadata_v3",
    "track1/metadata_v0",
    "metadata_v3",
    "metadata_v0",
]


@dataclass
class ValidatedConfig:
    """
    Data object that stores validated user-supplied configuration options for a video
    download operation.
    """

    input_directory: str
    validate_all: bool
    metadata_folder: str
    error_details_name: str
    error_summary_name: str
    aws_profile_name: str
    universities: Set[str]
    num_workers: int
    expiry_time_sec: int


@dataclass
class Config:
    """
    Data object that stores the user-supplied configuration options for a video download
    operation.
    """

    input_directory: str
    validate_all: bool
    metadata_folder: str
    error_details_name: str
    error_summary_name: str
    num_workers: int
    expiry_time_sec: int
    aws_profile_name: str
    universities: List[str] = field(default_factory=list)


def validate_config(cfg: Config) -> ValidatedConfig:
    """

    Args:
        cfg: A user-supplied configuration for the download operation
    Returns:
        A ValidatedConfig if all user-supplied options appear valid
    """

    try:
        boto3.session.Session(profile_name=cfg.aws_profile_name)
    except ProfileNotFound:
        raise RuntimeError(f"Could not find AWS profile '{cfg.aws_profile_name}'.")

    return ValidatedConfig(
        input_directory=cfg.input_directory,
        validate_all=bool(cfg.validate_all),
        metadata_folder=cfg.metadata_folder,
        error_details_name=cfg.error_details_name,
        error_summary_name=cfg.error_summary_name,
        aws_profile_name=cfg.aws_profile_name,
        universities=set(cfg.universities) if cfg.universities else {},
        num_workers=cfg.num_workers,
        expiry_time_sec=cfg.expiry_time_sec,
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
    )
    flag_parser.add_argument(
        "-ed", "--error_details_name", help="output file name for error details"
    )
    flag_parser.add_argument(
        "-es", "--error_summary_name", help="output file name for error summary"
    )
    flag_parser.add_argument(
        "-nw",
        "--num_workers",
        type=int,
        help="number of workers",
        default=25,
        required=False,
    )
    flag_parser.add_argument(
        "-expiry",
        "--expiry_time_sec",
        type=int,
        help="default expiry for presigned URLs (in seconds)",
        default=3600,
        required=False,
    )
    flag_parser.add_argument(
        "--aws_profile_name",
        help="Defaults to 'default'. Specifies the AWS profile name from "
        "~/.aws/credentials to use for the download",
        default="default",
    )
    flag_parser.add_argument(
        "--universities",
        nargs="+",
        choices=UNIV_TO_BUCKET.keys(),
        help="List of university IDs. If specified, only UIDs from the S3 buckets "
        "belonging to the listed universities will be downloaded. A full list of "
        "university IDs can be found in the ego4d/cli/universities.py file.",
    )

    # Use the values in the config file, but set them as defaults to flag_parser so they
    # can be overridden by command line flags
    if args.config_path:
        with open(args.config_path.expanduser()) as f:
            config_contents = json.load(f)
            flag_parser.set_defaults(**config_contents)

    parsed_args = flag_parser.parse_args(remaining)

    flags = {k: v for k, v in vars(parsed_args).items() if v is not None}

    # Note: Since the flags from the config file are being used as default argparse
    # values, we can't set required=True. Doing so would mean that the user couldn't
    # leave them unspecified when invoking the CLI.

    config = Config(**flags)

    # We need to check the universities values here since they might have been set
    # through the JSON config file, in which case the argparse checks won't validate
    # them
    universities = set(UNIV_TO_BUCKET.keys())
    unrecognized = set(config.universities) - universities
    if unrecognized:
        raise RuntimeError(
            f"Unrecognized universities: {unrecognized}. Please choose "
            f"from: {universities}"
        )

    return config
