# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import json
import tempfile

import pytest

from .config import config_from_args


def test_invalid_flags():
    with pytest.raises(RuntimeError):
        # missing required flag output_directory
        config_from_args(["--version=v1", "--datasets", "full_scale"])

    # with pytest.raises(RuntimeError):
    #     # missing required flag datasets
    #     config_from_args(["--version=v1", "--output_directory=~/test"])

    with pytest.raises(SystemExit):
        # invalid university choice
        config_from_args(
            [
                "--datasets",
                "full_scale",
                "--output_directory=~/test",
                "--version=v1",
                "--universities",
                "cmu",
                "fake",
            ]
        )

    with pytest.raises(SystemExit):
        # unrecognized flag
        config_from_args(
            [
                "--datasets",
                "full_scale",
                "--output_directory=~/test",
                "--version=v1",
                "--fake_flag=123",
            ]
        )

    with pytest.raises(SystemExit):
        # unrecognized flag
        config_from_args(
            [
                "--datasets",
                "full_scale",
                "--output_directory=~/test",
                "--version=v1",
                "--video_uids",
                "123",
                "456",
                "--video_uid_file=~/uid_file",
            ]
        )


def test_valid_flags():
    c = config_from_args(
        [
            "--datasets",
            "full_scale",
            "--output_directory=~/test",
            "--version=v1",
            "--universities",
            "cmu",
            "indiana",
            "--aws_profile_name=ego4d",
            "--video_uids",
            "123",
            "456",
            "789",
            "--metadata",
        ]
    )

    assert c.version == "v1"
    assert c.datasets == ["full_scale"]
    assert c.universities == ["cmu", "indiana"]
    assert c.output_directory == "~/test"
    assert c.aws_profile_name == "ego4d"
    assert c.video_uids == ["123", "456", "789"]
    assert c.metadata == True


def test_version_flags():
    c = config_from_args(
        [
            "--datasets",
            "full_scale",
            "--output_directory=~/test",
            "--version=v2",
            "--universities",
            "cmu",
            "indiana",
            "--aws_profile_name=ego4d",
            "--video_uids",
            "123",
            "456",
            "789",
            "--metadata",
        ]
    )

    assert c.version == "v2"
    assert c.datasets == ["full_scale"]
    assert c.universities == ["cmu", "indiana"]
    assert c.output_directory == "~/test"
    assert c.aws_profile_name == "ego4d"
    assert c.video_uids == ["123", "456", "789"]
    assert c.metadata == True


def test_video_uid_file():
    uids = "123 456\n789"

    with tempfile.NamedTemporaryFile(mode="w") as f:
        f.write(uids)
        f.flush()

        c = config_from_args(
            [
                "--datasets",
                "full_scale",
                "--output_directory=~/test",
                "--version=v1",
                f"--video_uid_file={f.name}",
            ]
        )

        assert c.video_uids == ["123", "456", "789"]


def test_json_file():
    conf = {
        "version": "v1",
        "datasets": ["full_scale"],
        "universities": ["cmu", "indiana"],
        "output_directory": "~/test",
        "aws_profile_name": "ego4d",
        "video_uids": ["123", "456", "789"],
    }

    with tempfile.NamedTemporaryFile(mode="w") as f:
        json.dump(conf, f)
        f.flush()

        # JSON values should be loaded into config
        c = config_from_args([f"--config_path={f.name}"])
        assert c.version == "v1"
        assert c.datasets == ["full_scale"]
        assert c.universities == ["cmu", "indiana"]
        assert c.output_directory == "~/test"
        assert c.aws_profile_name == "ego4d"
        assert c.video_uids == ["123", "456", "789"]

        # JSON values should be overridden by flags that are also specified on the
        # command line
        c = config_from_args(
            [
                f"--config_path={f.name}",
                "--output_directory=~/test2",
                "--video_uids",
                "000",
                "000",
            ]
        )
        assert c.version == "v1"
        assert c.datasets == ["full_scale"]
        assert c.universities == ["cmu", "indiana"]
        assert c.output_directory == "~/test2"
        assert c.aws_profile_name == "ego4d"
        assert c.video_uids == ["000", "000"]

        # If video_uid_file specified on command line, but video_uids is specified in
        # JSON file, then CLI should error
        with pytest.raises(RuntimeError):
            config_from_args(
                [
                    f"--config_path={f.name}",
                    "--output_directory=~/test2",
                    "--video_uid_file=~/uid_file",
                ]
            )
