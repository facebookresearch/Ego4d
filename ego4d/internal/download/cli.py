import argparse
import os
import sys
import threading
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Tuple, TypeVar

from ego4d.cli.progressbar import DownloadProgressBar
from ego4d.cli.universities import UNIV_TO_BUCKET

from ego4d.internal.download.manifest import (
    manifest_loads,
    ManifestEntry,
    PathSpecification,
)
from ego4d.internal.s3 import S3Downloader
from iopath.common.file_io import PathManager
from iopath.common.s3 import S3PathHandler
from tqdm.auto import tqdm

T = TypeVar("T")
U = TypeVar("U")


def _path_ok(p: PathSpecification, args) -> bool:
    ok = True
    if (
        args.uids is not None
        and p.uids is not None
        and len(set(p.uids) & args.uids) == 0
    ):
        ok = False
    if (
        args.views is not None
        and p.views is not None
        and len(set(p.views) & args.views) == 0
    ):
        ok = False
    if (
        args.universities is not None
        and p.universities is not None
        and len(set(p.universities) & args.universities) == 0
    ):
        ok = False
    return ok


def _manifest_ok(m: ManifestEntry, args) -> bool:
    ok = True
    if (
        args.splits is not None
        and m.splits is not None
        and len(set(m.splits) & args.splits) == 0
    ):
        ok = False

    if (
        args.benchmarks is not None
        and m.benchmarks is not None
        and len(set(m.benchmarks) & args.benchmarks) == 0
    ):
        ok = False
    return ok


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
    release_name = args.release
    parts = set(args.parts)
    out_dir = args.out_dir
    num_workers = args.num_workers
    s3_profile = args.s3_profile
    yes = args.yes
    args.uids = set(args.uids) if args.uids is not None else None
    args.benchmarks = set(args.benchmarks) if args.benchmarks is not None else None
    args.splits = set(args.splits) if args.splits is not None else None
    args.views = set(args.views) if args.views is not None else None
    args.universities = (
        set(args.universities) if args.universities is not None else None
    )

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

    num_paths = 0
    all_paths = []
    for part in parts:
        manifest_path = os.path.join(release_dir, part, "manifest.json")
        assert pathmgr.exists(
            manifest_path
        ), f"{part} does not have a manifest path (looking at {manifest_path})"
        ms = manifest_loads(pathmgr.open(manifest_path).read())
        for m in ms:
            num_paths += len(m.paths)
            if not _manifest_ok(m, args):
                continue

            all_paths.extend([p for p in m.paths if _path_ok(p, args)])

    if num_paths != len(all_paths):
        print(f"Filtered {num_paths} -> {len(all_paths)} files")

    # TODO: pre-cache this such that it is faster
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
        print(
            "*** BUCKET FAILURES ***: ",
            {
                path.source_path.split("s3://")[1].split("/")[0]
                for path, _ in s3_stat_failures
            },
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

    if args.delete:
        print("Scanning for files to delete ...")
        files_that_exist = []
        for (dirpath, _, filenames) in os.walk(out_dir):
            files_that_exist.extend([os.path.join(dirpath, f) for f in filenames])

        files_that_exist = set(files_that_exist)
        files_to_delete = files_that_exist - {
            os.path.join(out_dir, x[0].relative_path) for x in success_path_size_pairs
        }
        print(
            f"Deleting: {len(files_to_delete)} files ({len(files_that_exist)} total files)"
        )
        for f in tqdm(files_to_delete):
            os.remove(f)

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


def create_arg_parse(script_name: str, base_dir: str, release_name: str):
    parser = argparse.ArgumentParser(
        usage=f"""
    EgoExo downloader CLI

    Simple usage:
        {script_name} -o <out_dir>

    Advanced usage examples:
        - Download point clouds and annotations 
            {script_name} -o <out_dir> --parts annotations point_cloud -y
        - Download VRS files for a capture
            {script_name} -o <out_dir> --parts capture_raw_vrs --uids <uid1>

"""
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        default=None,
        help="Which folder to download the data to",
        required=True,
    )
    parser.add_argument(
        "--parts",
        type=str,
        nargs="+",
        default=["metadata", "captures", "takes", "trajectory", "annotations"],
        help="""
What parts of the dataset to download, one of: {metadata, annotations, takes, captures, trajectory, eye_gaze, point_cloud, capture_raw_stitched_videos, capture_raw_vrs, ego_pose_pseudo_gt}

By default the following parts will be downloaded: {metadata, captures, takes, trajectory, annotations}.

Example usage: --parts annotations point_cloud eye_gaze
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
        "--benchmarks",
        type=str,
        nargs="+",
        default=None,
        help="""
Data relating to particular benchmarks will be only included. Valid benchmark
values are: keystep, proficiency, proficiency_demonstration, proficiency_demonstrator, relations, translation, correspondence, atomic_action_descriptions, egopose
""",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=None,
        help="""
Data relating to the train/val/test dataset splits. Valid values are {train, val, test}.
""",
    )
    parser.add_argument(
        "--views",
        type=str,
        nargs="+",
        default=None,
        help="""
Data relating to a particular view. Valid values are {ego, exo}.
""",
    )
    parser.add_argument(
        "-u",
        "--universities",
        type=str,
        nargs="+",
        default=None,
        help=f"""
Data relating to a particular university. Valid values are:
{", ".join(UNIV_TO_BUCKET.keys())}
""",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=15,
        help="number of workers to perform download ops",
    )
    parser.add_argument(
        "-y",
        "--yes",
        default=False,
        help="don't prompt to confirm",
        action="store_true",
    )
    parser.add_argument(
        "-d",
        "--delete",
        default=False,
        help="delete unused files in the directory",
        action="store_true",
    )
    parser.add_argument(
        "--force",
        default=False,
        help="force a download of all files",
        action="store_true",
    )
    parser.add_argument(
        "--release", type=str, default=release_name, help="name/version of the release"
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
        default=base_dir,
        help="base directory for download (ADVANCED usage, do not change unless you know what you're doing)",
    )
    return parser


def internal_main():
    parser = create_arg_parse(
        script_name="egoexo_internal",
        release_name="dev",
        base_dir="s3://ego4d-consortium-sharing/egoexo/releases/",
    )
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    internal_main()
