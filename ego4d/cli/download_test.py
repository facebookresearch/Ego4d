# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

from unittest.mock import Mock, patch

from .download import (
    download_all_with_retries,
    FileToDownload,
    filter_already_downloaded,
    VersionEntry,
)


@patch("ego4d.cli.download.time.sleep")
@patch("ego4d.cli.download.list_corrupt_files", return_value=[])
@patch("ego4d.cli.download.download_all")
def test_successful_download_does_not_retry(download_all, _, sleep):
    completed = Mock()
    download_all.return_value = [completed]

    failed = download_all_with_retries(
        [completed], [], "ego4d", download_retries=3, retry_delay_seconds=5
    )

    assert failed == []
    assert download_all.call_count == 1
    sleep.assert_not_called()


@patch("ego4d.cli.download.time.sleep")
@patch("ego4d.cli.download.list_corrupt_files")
@patch("ego4d.cli.download.download_all")
def test_failure_followed_by_success_retries_only_failed(
    download_all, list_corrupt_files, sleep
):
    completed = Mock()
    failed_then_completed = Mock()
    download_all.side_effect = [
        [completed, failed_then_completed],
        [failed_then_completed],
    ]
    list_corrupt_files.side_effect = [[failed_then_completed], []]
    callback = Mock()

    failed = download_all_with_retries(
        [completed, failed_then_completed],
        [],
        "ego4d",
        download_retries=3,
        retry_delay_seconds=5,
        callback=callback,
    )

    assert failed == []
    assert download_all.call_args_list[0].args[0] == [completed, failed_then_completed]
    assert download_all.call_args_list[1].args[0] == [failed_then_completed]
    assert download_all.call_args_list[0].kwargs["callback"] is callback
    assert download_all.call_args_list[1].kwargs["callback"] is None
    sleep.assert_called_once_with(5)


@patch("ego4d.cli.download.time.sleep")
@patch("ego4d.cli.download.list_corrupt_files")
@patch("ego4d.cli.download.download_all")
def test_retry_exhaustion_returns_failed_downloads(
    download_all, list_corrupt_files, sleep
):
    failed_download = Mock()
    download_all.return_value = [failed_download]
    list_corrupt_files.return_value = [failed_download]

    failed = download_all_with_retries(
        [failed_download], [], "ego4d", download_retries=2, retry_delay_seconds=1
    )

    assert failed == [failed_download]
    assert download_all.call_count == 3
    assert sleep.call_count == 2


@patch("ego4d.cli.download.time.sleep")
@patch("ego4d.cli.download.list_corrupt_files")
@patch("ego4d.cli.download.download_all")
def test_zero_retries_makes_one_attempt(download_all, list_corrupt_files, sleep):
    failed_download = Mock()
    download_all.return_value = [failed_download]
    list_corrupt_files.return_value = [failed_download]

    failed = download_all_with_retries(
        [failed_download], [], "ego4d", download_retries=0, retry_delay_seconds=5
    )

    assert failed == [failed_download]
    assert download_all.call_count == 1
    sleep.assert_not_called()


@patch("ego4d.cli.download.download_all")
def test_integrity_size_mismatch_is_retried(download_all, tmp_path):
    file_path = tmp_path / "clip.mp4"
    download = FileToDownload("clip.mp4", tmp_path, "clip")
    download.file_path = file_path
    download.s3_content_size_bytes = 4

    def download_side_effect(downloads, *args, **kwargs):
        file_path.write_bytes(b"x" if download_all.call_count == 1 else b"xxxx")
        return list(downloads)

    download_all.side_effect = download_side_effect

    failed = download_all_with_retries(
        [download], [], "ego4d", download_retries=1, retry_delay_seconds=0
    )

    assert failed == []
    assert download_all.call_count == 2


def test_existing_version_and_size_matching_file_is_skipped(tmp_path):
    file_path = tmp_path / "clip.mp4"
    file_path.write_bytes(b"data")
    s3_object = Mock(version_id="s3-version", content_length=4)
    download = FileToDownload("clip.mp4", tmp_path, "clip", s3_object=s3_object)

    active = filter_already_downloaded(
        [download], [VersionEntry("clip", "s3-version", "clip.mp4")]
    )

    assert active == []
    s3_object.load.assert_called_once_with()
