# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
A data object for storing user options, along with utilities for parsing input options
from command line flags and a configuration file.
"""

import argparse
import json
from pathlib import Path
from typing import List, NamedTuple, Set, Union

import boto3.session
from botocore.exceptions import ProfileNotFound
from ego4d.cli.universities import UNIV_TO_BUCKET


VERSION_DEFAULT = "v2_1"
DATASET_PRIMARY = "full_scale"
DATASETS_VIDEO = ["full_scale", "clips", "components/videos", "video_540ss"]
DATASETS_FILE = [
    "annotations",
    "viz",
    "imu",
    "gaze",
    "3d",
    "3d_scan_keypoints",
    "3d_scans",
    "av_models",
    "social_test",
    "vq2d_models",
    "vq2d_detections",
    "sta_models",
    "lta_models",
    "slowfast8x8_r101_k400",
    "egotracks",
    "paco_annotations",
    "paco_frames",
]
DATASETS_ALL = DATASETS_VIDEO + DATASETS_FILE
DATASET_FILE_EXTENSIONS = [
    ".mp4",
    ".json",
    ".jsonl",
    ".jpg",
    ".txt",
    ".csv",
    ".pt",
    ".tar",
    ".tgz",
    ".ckpt",
    ".pth",
    ".zip",
]


class ValidatedConfig(NamedTuple):
    """
    Data object that stores validated user-supplied configuration options for a video
    download operation.
    """

    output_directory: Path
    assume_yes: bool
    version: str
    datasets: Set[str]
    benchmarks: Set[str]
    aws_profile_name: str
    metadata: bool
    manifest: bool
    bypass_version_check: bool
    skip_s3_checks: bool
    video_uids: Set[str]
    universities: Set[str]
    annotations: Union[bool, Set[str]]
    list_datasets: bool
    manifest_override_path: str

    @property
    def out_version_dir(self) -> str:
        base_version = self.version.split("_")[0]
        return base_version


class Config(NamedTuple):
    """
    Data object that stores the user-supplied configuration options for a video download
    operation.
    """

    output_directory: str = None
    assume_yes: bool = False
    version: str = VERSION_DEFAULT
    datasets: List[str] = []
    benchmarks: Set[str] = []
    aws_profile_name: str = "default"
    metadata: bool = False
    manifest: bool = False
    bypass_version_check: bool = False
    skip_s3_checks: bool = False
    video_uids: List[str] = []
    universities: List[str] = []
    annotations: Union[bool, List[str]] = True
    list_datasets: bool = False
    manifest_override_path: str = None


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
        output_directory=(
            Path(cfg.output_directory).expanduser() if cfg.output_directory else None
        ),
        version=cfg.version,
        datasets=set(cfg.datasets),
        benchmarks=set(cfg.benchmarks),
        aws_profile_name=cfg.aws_profile_name,
        manifest=cfg.manifest,
        metadata=cfg.metadata,
        video_uids=set(cfg.video_uids) if cfg.video_uids else {},
        universities=set(cfg.universities) if cfg.universities else {},
        assume_yes=bool(cfg.assume_yes),
        bypass_version_check=cfg.bypass_version_check,
        skip_s3_checks=cfg.skip_s3_checks,
        annotations=cfg.annotations if isinstance(cfg.annotations, bool) else set(),
        list_datasets=cfg.list_datasets,
        manifest_override_path=cfg.manifest_override_path,
    )


def config_from_args(args=None) -> Config:
    """
    Parses command line flags and returns a Config object with corresponding values from
    the flags.
    """
    # Parser for a configuration file
    json_parser = argparse.ArgumentParser(add_help=False)

    json_parser.add_argument(
        "--config_path",
        type=Path,
        help="Local path to a config JSON file. If specified, the flags will be read "
        "from this file instead of the command line.",
    )

    json_args, remaining = json_parser.parse_known_args(args=args)

    # Parser for command line flags other than the configuration file
    flag_parser = argparse.ArgumentParser(
        description="Command line tool to download Ego4D datasets from Amazon S3",
        add_help=True,
    )

    flag_parser.add_argument(
        "-o",
        "--output_directory",
        help="A local path where the downloaded files and metadata will be stored",
    )
    flag_parser.add_argument(
        "--datasets",
        nargs="+",
        help="The datasets to download: 'full_scale', 'annotations', etc."
        "\nThe datasets will be stored in folders in the output directory with the name "
        "of the dataset (e.g. output_dir/full_scale/).",
        default=["annotations"],
    )
    flag_parser.add_argument(
        "--benchmarks",
        nargs="*",
        help="The benchmarks to download dataset subsets of: 'EM', 'FHO', 'AV'",
    )
    flag_parser.add_argument(
        "--annotations",
        nargs="*",
        help="(Currently unsupported) If passed without arguments, downloads all annotations.  "
        "Otherwise, a list of specific annotations to pass, e.g. narration, fho, moments, vq, nlq, av",
    )
    flag_parser.add_argument(
        "--metadata",
        dest="metadata",
        action="store_true",
        help="Download the primary ego4d.json to the folder root.  (Default: True)",
    )
    flag_parser.add_argument(
        "--no-metadata",
        dest="metadata",
        action="store_false",
        help="Bypass downloading the primary ego4d.json.",
    )
    flag_parser.set_defaults(metadata=True)
    flag_parser.add_argument(
        "--manifest",
        const=True,
        action="store_const",
        help="Downloads the video manifest. (True by default, only relevant if you want only the manifest.)",
    )
    # flag_parser.add_argument(
    #     "--viz",
    #     const=True,
    #     action="store_const",
    #     help="Downloads the local visualization dataset. (Convenience option equivalent to including viz in datasets.)",
    # )
    flag_parser.add_argument(
        "--bypass-existing",
        const=True,
        action="store_const",
        dest="bypass_version_check",
        help="Bypass existing files without checking file versions/sizes.",
    )
    flag_parser.add_argument(
        "--skip-s3-checks",
        const=True,
        action="store_const",
        dest="skip_s3_checks",
        help="Skips initial check that files are present on S3 before downloading. Download size estimates are disabled with this flag.",
    )
    flag_parser.add_argument(
        "--version",
        help="A version identifier - i.e. 'v1', 'v2', 'v2_1'",
        default=VERSION_DEFAULT,
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
    flag_parser.add_argument(
        "-y",
        "--yes",
        default=False,
        const=True,
        action="store_const",
        dest="assume_yes",
        help="If this flag is set, then the CLI will not show a prompt asking the user "
        "to confirm the download. This is so that the tool can be used as part of "
        "shell scripts.",
    )
    video_uid_group = flag_parser.add_mutually_exclusive_group()
    video_uid_group.add_argument(
        "--video_uids",
        nargs="+",
        help="List of video UIDs to be downloaded. If not specified, all relevant UIDs "
        "will be downloaded.",
    )
    video_uid_group.add_argument(
        "--video_uid_file",
        help="Path to a whitespace delimited file that contains a list of UIDs. "
        "Mutually exclusive with the video_uids flag.",
    )
    flag_parser.add_argument(
        "--manifest-override-path",
        help="Override the manifest with the specified path.  Only relevant if exactly "
        "1 dataset is specified",
    )

    flag_parser.add_argument(
        "--list-datasets",
        dest="list_datasets",
        action="store_true",
        help="List the available datasets",
    )

    # Use the values in the config file, but set them as defaults to flag_parser so they
    # can be overridden by command line flags
    if json_args.config_path:
        with open(json_args.config_path.expanduser()) as f:
            config_contents = json.load(f)
            flag_parser.set_defaults(**config_contents)

    parsed_args = flag_parser.parse_args(remaining)
    if parsed_args.datasets is None and parsed_args.annotations is None:
        raise RuntimeError(
            f"Please specify either datasets, annotations or manifest for a minimal download set."
        )

    unknown_datasets = [x for x in parsed_args.datasets if x not in DATASETS_ALL]
    if unknown_datasets:
        print(
            f"Warning: Non-standard Dataset Specfied (Allowed, will attempt download): {unknown_datasets}"
        )

    # if parsed_args.viz:
    #     if parsed_args.datasets:
    #         if "viz" not in parsed_args.datasets:
    #             print("Adding viz to datasets..")
    #             parsed_args.datasets.append("viz")
    #     else:
    #         parsed_args.datasets = ["viz"]
    #     del parsed_args.viz

    help_cmd = False
    if parsed_args.list_datasets:
        help_cmd = True

    flags = {k: v for k, v in vars(parsed_args).items() if v is not None}

    required_flags = {"output_directory"} if not help_cmd else set()
    missing = required_flags - flags.keys()
    if missing:
        raise RuntimeError(f"Missing required flags: {missing}")

    # Read video_uid_file if it is specified and put the UIDs in the "video_uids" key
    if "video_uid_file" in flags:
        if "video_uids" in flags:
            raise RuntimeError(
                "argument --video_uid_file: not allowed with argument" "--video_uids"
            )

        uids_str = Path(flags.pop("video_uid_file")).expanduser().read_text()
        flags["video_uids"] = uids_str.split()

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
