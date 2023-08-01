"""
Example usage:

"""
import argparse
import os
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Tuple, TypeVar

from ego4d.cli.progressbar import DownloadProgressBar

from ego4d.internal.download.types import manifest_loads, PathSpecification
from ego4d.internal.s3 import S3Downloader
from iopath.common.file_io import PathManager
from iopath.common.s3 import S3PathHandler
from tqdm.auto import tqdm

T = TypeVar("T")
U = TypeVar("U")


def map_all(
    values: List[T],
    map_fn: Callable[[Optional[S3Downloader], T], Tuple[U, T, Optional[str]]],
    num_workers: int,
    s3_profile: Optional[str],
    needs_downloader: bool,
    progress_on_bytes: bool,
    total_bytes: Optional[int],
) -> Tuple[List[Tuple[T, U]], List[Tuple[T, str]]]:
    # NOTE: boto3 is not thread safe
    thread_data = threading.local()
    callback = None

    def initializer():
        if needs_downloader:
            thread_data.downloader = S3Downloader(
                s3_profile,
                num_workers=num_workers,
                callback=callback,
            )
        else:
            thread_data.downloader = None

    def wrap_map(x):
        return map_fn(thread_data.downloader, x)

    failures = []
    ret = []
    iterator_wrapper = tqdm
    if progress_on_bytes:
        progress = DownloadProgressBar(total_bytes)
        iterator_wrapper = lambda x, total: x
        callback = progress.update if progress is not None else None

    with ThreadPoolExecutor(max_workers=num_workers, initializer=initializer) as pool:
        for value, key, err in iterator_wrapper(
            pool.map(wrap_map, values), total=len(values)
        ):
            if value is None:
                failures.append((key, err))
            else:
                ret.append((key, value))
    return ret, failures


def main(args):
    base_dir = args.base_dir
    release_name = args.release_name
    parts = set(args.parts)
    out_dir = args.out_dir
    num_workers = args.num_workers
    s3_profile = args.s3_profile
    yes = args.yes
    uids = set(args.uids) if args.uids is not None else None

    # TODO: remove iopath dependency
    pathmgr = PathManager()
    pathmgr.register_handler(S3PathHandler(profile=s3_profile))

    def check_file(
        downloader: Optional[S3Downloader],
        path_expected_size: Tuple[PathSpecification, int],
    ) -> Tuple[Optional[int], Tuple[PathSpecification, int], Optional[str]]:
        assert downloader is None
        path, expected_size = path_expected_size
        out_path = os.path.join(out_dir, path.relative_path)
        try:
            if not os.path.exists(out_path):
                return 0, path_expected_size, None
            size = os.path.getsize(out_path)
            ok = size == expected_size
            return size if ok else 0, path_expected_size, None
        except Exception:
            return None, path_expected_size, traceback.format_exc()

    def get_size(
        downloader: Optional[S3Downloader], path: PathSpecification
    ) -> Tuple[Optional[int], PathSpecification, Optional[str]]:
        assert downloader is not None
        try:
            if not path.source_path.startswith("s3://"):
                return None, path, "path is not an S3 path"
            return downloader.obj(path.source_path).content_length, path, None
        except Exception:
            return None, path, traceback.format_exc()

    def download(
        downloader: Optional[S3Downloader],
        path_size_pair: Tuple[PathSpecification, int],
    ) -> Tuple[Optional[int], Tuple[PathSpecification, int], Optional[str]]:
        assert downloader is not None
        path, _ = path_size_pair
        try:
            out_path = os.path.join(out_dir, path.relative_path)
            downloader.copy(path.source_path, out_path)
            size, _, err = check_file(None, path_size_pair)
            return size, path_size_pair, err
        except Exception:
            return None, path_size_pair, traceback.format_exc()

    os.makedirs(out_dir, exist_ok=True)
    assert os.path.isdir(out_dir), f"output dir {out_dir} is not a directory"

    release_dir = os.path.join(base_dir, release_name)
    if not release_dir.endswith("/"):
        release_dir += "/"

    manifests = []
    for part in parts:
        manifest_path = os.path.join(release_dir, part, "manifest.json")
        assert pathmgr.exists(
            manifest_path
        ), f"{part} does not have a manifest path (looking at {manifest_path})"
        manifests.append(manifest_loads(pathmgr.open(manifest_path).read()))

    all_paths = []
    for m in manifests:
        for x in m:
            if uids is not None and x.uid not in uids:
                continue
            all_paths.extend(x.paths)

    print("Determining what to download ...")
    path_size_pairs, s3_stat_failures = map_all(
        all_paths,
        map_fn=get_size,
        num_workers=num_workers,
        s3_profile=s3_profile,
        needs_downloader=True,
        progress_on_bytes=False,
        total_bytes=None,
    )
    s3_zero_sizes = [path for path, size in path_size_pairs if size == 0]
    success_path_size_pairs = [
        (path, size) for path, size in path_size_pairs if size != 0
    ]
    all_s3_stat_failures = len(s3_zero_sizes) + len(s3_stat_failures)
    if all_s3_stat_failures > 0:
        print(
            f"WARN: failed to get stats for {all_s3_stat_failures} files, will skip. [zero_size={len(s3_zero_sizes)}, exceptions={all_s3_stat_failures}]"
        )
    total_size_bytes = sum(x for _, x in path_size_pairs if x is not None)
    total_size_gib = total_size_bytes / 1024**3

    print("Checking current download status ...")
    curr_paths, curr_stats_failures = map_all(
        success_path_size_pairs,
        map_fn=check_file,  # pyre-ignore
        num_workers=num_workers,
        s3_profile=s3_profile,
        needs_downloader=False,
        progress_on_bytes=False,
        total_bytes=None,
    )
    # TODO: refactor such that we can tell the number of files to be updated
    existing_paths = {
        path_size_pair
        for path_size_pair, actual_size in curr_paths
        if actual_size is not None and actual_size > 0
    }
    existing_gib = sum(
        size / 1024**3 for _, size in existing_paths if size is not None
    )
    existing_len = len(existing_paths)
    progress_percent = existing_gib / total_size_gib

    print(
        f"Downloaded: {progress_percent:.3%} = {existing_gib:.3f}GiB / {total_size_gib:.3f}GiB ({existing_len} / {len(path_size_pairs)} files) downloaded"
    )
    if len(curr_stats_failures) > 0:
        print(f"WARN: {len(curr_stats_failures)} failed to stat")

    ps_to_dl = {
        path: size
        for path, size in success_path_size_pairs
        if (path, size) not in existing_paths
    }
    if len(ps_to_dl) == 0 and not args.force:
        print("Everything has been downloaded. Bye.")
        sys.exit(0)

    if args.force:
        print("Forcing everything to be re-downloaded ...")
        ps_to_dl = {path: size for path, size in success_path_size_pairs}

    assert all(x is not None for x in ps_to_dl.values())
    expected_gb = sum(x / 1024**3 for x in ps_to_dl.values() if x is not None)
    confirm = False
    if not yes:
        response = input(
            f"Expected size of downloaded files is "
            f"{expected_gb:.1f} GiB ({len(ps_to_dl)} files). "
            f"Do you want to start the download? ([y]/n): "
        )
        if response.lower() in ["yes", "y", ""]:
            confirm = True
        else:
            confirm = False
    else:
        confirm = True

    if not confirm:
        print("Aborting...")
        sys.exit(0)

    print("Preparing output directories ...")
    all_out_dirs = {
        os.path.join(out_dir, os.path.dirname(x.relative_path)) for x in all_paths
    }
    for x in tqdm(all_out_dirs):
        os.makedirs(x, exist_ok=True)

    print("Downloading ...")
    assert all(size is not None for size in ps_to_dl.values())
    paths_to_fetch = [
        (path, size) for path, size in ps_to_dl.items() if size is not None
    ]
    total_bytes_to_fetch = sum(size for _, size in paths_to_fetch)
    print(f"Fetching: {total_bytes_to_fetch/1024**3:.3f}GiB")
    paths_downloaded, dl_failures = map_all(
        paths_to_fetch,  # pyre-ignore
        map_fn=download,  # pyre-ignore
        num_workers=num_workers,
        s3_profile=s3_profile,
        needs_downloader=True,
        progress_on_bytes=True,
        total_bytes=total_bytes_to_fetch,
    )

    integrity_errs = [path for path, size in paths_downloaded if size == 0]
    if len(dl_failures) > 0:
        num_all_failures = len(dl_failures) + len(integrity_errs)
        print(
            f"WARN: failed to fetch {num_all_failures} files [integrity={len(integrity_errs)}, exceptions={len(dl_failures)}]"
        )
        print("Please retry the download (... returning with error code 2)")
        sys.exit(2)
    print(flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        usage="""
    EgoExo downloader CLI

    Simple usage:
        python ego4d/internal/download/cli.py -o <out_dir>

    Advanced usage examples:
        - Download point clouds and annotations 
            python ego4d/internal/download/cli.py -o <out_dir> --parts annotations point_cloud -y
        - Download VRS files for a capture
            python ego4d/internal/download/cli.py -o <out_dir> --parts capture_raw_vrs --uids <uid1>

"""
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        default=None,
        help="Where to download the data",
        required=True,
    )
    parser.add_argument(
        "--parts",
        type=str,
        nargs="+",
        default=["metadata", "captures", "takes", "trajectory", "annotations"],
        help="""
What parts of the dataset to download, one of {metadata, takes, trajectory, point_cloud, annotations, capture_raw_vrs, capture_raw_stitched_videos}.

By default the following parts will be downloaded: {metadata, captures, takes, trajectory, annotations}.

Example usage: --parts annotations point_cloud
""",
    )
    parser.add_argument(
        "--uids",
        type=str,
        nargs="+",
        default=None,
        help="what uids to filter for takes or captures",
    )
    parser.add_argument(
        "--num_workers",
        type=str,
        default=15,
        help="number of workers to perform download ops",
    )
    parser.add_argument(
        "--release_name", type=str, default="dev", help="name of the release"
    )
    parser.add_argument(
        "-y",
        "--yes",
        default=False,
        help="don't prompt to confirm",
        action="store_true",
    )
    parser.add_argument(
        "--force",
        default=False,
        help="force a download of all files",
        action="store_true",
    )
    parser.add_argument(
        "--s3_profile",
        type=str,
        default="default",
        help="profile to use for S3",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="s3://ego4d-consortium-sharing/egoexo/releases/",
        help="base directory for download (ADVANCED usage, do not change unless you know what you're doing)",
    )
    args = parser.parse_args()
    main(args)
