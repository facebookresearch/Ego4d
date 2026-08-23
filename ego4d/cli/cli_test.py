# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

from unittest.mock import patch

from .cli import DownloadFailedError, main


@patch("ego4d.cli.cli.main_cfg", side_effect=DownloadFailedError("incomplete"))
@patch("ego4d.cli.cli.config_from_args", return_value=object())
def test_retry_exhaustion_returns_nonzero(_, __):
    assert main() == 1
