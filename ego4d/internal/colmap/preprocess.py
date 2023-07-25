import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import av

import boto3
import cv2
import hydra
import numpy as np
import pandas as pd

from ego4d.internal.s3 import StreamPathMgr

from iopath.common.file_io import PathManager
from iopath.common.s3 import S3PathHandler

from omegaconf import OmegaConf

from tqdm.auto import tqdm

stream_pathmgr = StreamPathMgr()

pathmgr = PathManager()
pathmgr.register_handler(S3PathHandler(profile="default"))


VRS_BIN = "/private/home/miguelmartin/repos/vrs/build/tools/vrs/vrs"
COLMAP_BIN = "/private/home/miguelmartin/repos/colmap/build/src/exe/colmap"


@dataclass
class ColmapConfig:
    in_metadata_path: Optional[str]
    in_videos: Optional[Dict[str, str]]  # device_id -> path
    output_dir: str
    download_video_files: bool

    sync_exo_views: bool
    rot_mode: int

    camera_model: str
    include_aria: bool

    mobile_frames: Optional[List[int]] = None
    aria_frames: Optional[Dict[str, List[int]]] = None
    aria_walkthrough_start_sec: Optional[float] = None
    aria_walkthrough_end_sec: Optional[float] = None
    aria_use_sync_info: bool = False

    exo_frames: Optional[Dict[str, List[int]]] = None
    exo_from_frame: Optional[int] = None
    exo_to_frame: Optional[int] = None
    frame_rate: Optional[float] = 1 / 2
    name: Optional[str] = None
    aria_fps: float = 30
    exo_fps: Optional[float] = None
    mobile_fps: Optional[float] = None

    run_colmap: bool = False
    colmap_bin: Optional[str] = None
    vrs_bin: Optional[str] = None

    force_download: Optional[bool] = False
    take_id: Optional[str] = None
    video_source: Optional[str] = None


def get_uniq_cache_root_dir(config) -> str:
    assert (
        config.video_source is not None and config.take_id is not None
    ), "need video source and take_id to set unique output dir for COLMAP data"
    assert config.output_dir is not None, "need cache root dir"
    return os.path.join(
        config.output_dir, "cache", f"{config.video_source}_{config.take_id}"
    )


def get_timesync_path(config) -> str:
    return os.path.join(get_uniq_cache_root_dir(config), "timesync.csv")


def pilot_video_metadata(uni_name: str, take_id: str) -> str:
    return f"s3://ego4d-consortium-sharing/internal/egoexo_pilot/{uni_name}/{take_id}/metadata.json"  # noqa


def get_colmap_data_dir(config) -> str:
    assert (
        config.video_source is not None and config.take_id is not None
    ), "need video source and take_id to set unique output dir for COLMAP data"
    name = config.name
    if name is None:
        name = f"{config.camera_model}_s{int(config.sync_exo_views)}_r{config.rot_mode}_a{int(config.include_aria)}_fr{config.frame_rate:.3f}"  # noqa
    return os.path.join(
        config.output_dir, "output", f"{config.video_source}_{config.take_id}", name
    )


def frames_for_region(start_frame: int, end_frame: int, skip_frames: int) -> List[int]:
    # [start, end]
    return list(range(start_frame, end_frame + 1))[0::skip_frames]


def get_fps(fp: str) -> float:
    with av.open(stream_pathmgr.open(fp)) as cont:
        return float(cont.streams[0].average_rate)


def produce_colmap_script(
    config: ColmapConfig,
    skip_frame_gen: bool = False,
) -> ColmapConfig:
    vrs_bin = config.vrs_bin
    colmap_bin = config.colmap_bin
    if vrs_bin is None:
        vrs_bin = VRS_BIN
    if colmap_bin is None:
        colmap_bin = COLMAP_BIN

    if not os.path.exists(colmap_bin) and config.run_colmap:
        raise AssertionError(f"colmap binary does not exist: {colmap_bin}")

    if not os.path.exists(vrs_bin):
        raise AssertionError(f"vrs binary does not exist: {colmap_bin}")

    print("Getting data ready...")
    _get_data_ready(config, config.force_download, vrs_bin, skip_frame_gen)
    print(f"Writing colmap script using {colmap_bin}")
    _write_colmap_file(config, colmap_bin)
    print(f"Please run {os.path.join(get_colmap_data_dir(config), 'run_colmap.sh')}")
    return config


def _get_data_ready(
    config: ColmapConfig, force_download: bool, vrs_bin: str, skip_frame_gen: bool
):
    by_dev_id = _download_data(config, force_download)
    if not skip_frame_gen:
        _extract_frames(config, by_dev_id, vrs_bin)


def download_andor_generate_streams(
    metadata: Dict[str, Any],
    download_video_files: bool,
    force_download: bool,
    output_dir: str,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    if metadata["timesync_csv_path"] is not None:
        local_timesync_csv = os.path.join(output_dir, "timesync.csv")
        if metadata["timesync_csv_path"].startswith("s3://"):
            _download_s3_path(metadata["timesync_csv_path"], local_timesync_csv)
        else:
            # local path
            shutil.copy2(metadata["timesync_csv_path"], local_timesync_csv)

    by_dev_id = {}
    for v in tqdm(metadata["videos"]):
        by_dev_id[v["device_id"]] = v
        s3_path = v["s3_path"]
        if s3_path.startswith("s3://") and (
            download_video_files or "aria" in v["device_id"]
        ):
            local_path = os.path.join(output_dir, v["device_id"])
            by_dev_id[v["device_id"]]["local_path"] = local_path
            if os.path.exists(local_path) and not force_download:
                continue
            print(f"Fetching {s3_path} to {local_path}")
            _download_s3_path(s3_path, local_path)
        else:
            by_dev_id[v["device_id"]]["local_path"] = s3_path
    return by_dev_id


def _download_data(config: ColmapConfig, force_download: bool) -> Dict[str, Any]:
    assert config.output_dir is not None, "Please set the output directory"

    if config.in_metadata_path is None:
        assert (
            config.in_videos is not None
        ), "require input video paths if metdata path is None"
        assert not config.sync_exo_views, "cannot sync videos without metadata"

        print("WARNING: using input video paths. This should only be used for TESTING")
        meta = {"videos": [], "timesync_csv_path": None}
        for dev_id, path in config.in_videos.items():
            meta["videos"].append(
                {
                    "device_id": dev_id,
                    "s3_path": path,  # may actually be a local path
                }
            )
    else:
        assert (
            config.in_videos is None
        ), "cannot use input videos if metadata is not None"
        meta = json.load(pathmgr.open(config.in_metadata_path))
        if config.take_id is None or config.video_source is None:
            config.take_id = meta["take_id"]
            config.video_source = meta["video_source"]

    output_dir = get_uniq_cache_root_dir(config)
    return download_andor_generate_streams(
        metadata=meta,
        output_dir=output_dir,
        download_video_files=config.download_video_files,
        force_download=force_download,
    )


def _extract_frames(config, by_dev_id, vrs_bin):
    frame_out_dir = os.path.join(get_colmap_data_dir(config), "frames")
    shutil.rmtree(frame_out_dir, ignore_errors=True)
    os.makedirs(frame_out_dir, exist_ok=True)

    aria_keys = [k for k in by_dev_id.keys() if "aria" in k]
    exo_keys = [k for k in by_dev_id.keys() if "cam" in k]

    synced_df = None
    if config.sync_exo_views or (
        config.aria_walkthrough_start_sec is None
        and config.aria_walkthrough_end_sec is None
        and config.in_metadata_path is not None
    ):
        assert (
            config.aria_walkthrough_start_sec is None
        ), "If not providing end sec of walkthrough time, please set start to none for timesync option"  # noqa
        timesync_df = pd.read_csv(get_timesync_path(config))
        cam01_idx = timesync_df.cam01_global_time.first_valid_index()
        cam02_idx = timesync_df.cam02_global_time.first_valid_index()
        cam03_idx = timesync_df.cam03_global_time.first_valid_index()
        cam04_idx = timesync_df.cam04_global_time.first_valid_index()
        fsf_idx = max(cam01_idx, cam02_idx, cam03_idx, cam04_idx)  # first synced frame
        synced_df = timesync_df.iloc[fsf_idx:]

    if config.include_aria:
        if config.aria_frames is None:
            config.aria_frames = {k: None for k in aria_keys}
        print("aria used", aria_keys)
        for k in aria_keys:
            print(k, config.aria_walkthrough_start_sec, config.aria_walkthrough_end_sec)
            with tempfile.TemporaryDirectory() as tempdir:
                aria_frame_path = os.path.join(tempdir, "aria01_frames")
                os.makedirs(aria_frame_path, exist_ok=True)
                extract_aria_frames(
                    vrs_bin=vrs_bin,
                    aria_vrs_file_path=by_dev_id[k]["local_path"],
                    out_dir=aria_frame_path,
                    synced_df=synced_df,
                    from_point=config.aria_walkthrough_start_sec,
                    to_point=config.aria_walkthrough_end_sec,
                )
                skip_aria = (
                    int(np.round(config.aria_fps * config.frame_rate))
                    if config.aria_fps is not None and config.frame_rate is not None
                    else None
                )
                subsample_aria_frames(
                    aria_frame_path,
                    skip_aria,
                    frame_out_dir,
                    config.aria_frames[k],
                    config,
                    aria_key=k,
                )

    exo_fps = get_fps(by_dev_id["cam01"]["local_path"])
    pts_by_key = {}
    if config.sync_exo_views:
        assert set(exo_keys) == {"cam01", "cam02", "cam03", "cam04"}, (
            "for sync_mode, exo cameras needed: {cam01, cam02, cam03, cam04}, but got "
            + f"{set(exo_keys)}"
        )
        f1 = config.exo_from_frame
        f2 = config.exo_to_frame
        skip_num_frames = int(np.round(exo_fps * config.frame_rate))
        if config.exo_frames is not None:
            raise AssertionError("cannot use exo_frames in sync mode")
        cam01_pts = [
            int(x) for x in synced_df.cam01_pts.iloc[f1:f2:skip_num_frames].tolist()
        ]
        cam02_pts = [
            int(x) for x in synced_df.cam02_pts.iloc[f1:f2:skip_num_frames].tolist()
        ]
        cam03_pts = [
            int(x) for x in synced_df.cam03_pts.iloc[f1:f2:skip_num_frames].tolist()
        ]
        cam04_pts = [
            int(x) for x in synced_df.cam04_pts.iloc[f1:f2:skip_num_frames].tolist()
        ]
        pts_by_key["cam01"] = cam01_pts
        pts_by_key["cam02"] = cam02_pts
        pts_by_key["cam03"] = cam03_pts
        pts_by_key["cam04"] = cam04_pts
    else:
        if config.exo_from_frame is not None:
            assert (
                config.exo_frames is None
            ), "if given exo_from_frame and exo_to_frame - you cannot supply frames"
            assert config.exo_to_frame is not None, "need to_frame"
            assert config.frame_rate is not None, "need frame rate"
            f1 = config.exo_from_frame
            f2 = config.exo_to_frame
            skip_num_frames = int(np.round(exo_fps * config.frame_rate))

            frames = list(range(f1, f2))[::skip_num_frames]
            config.exo_frames = {k: frames for k in exo_keys}

        for k in exo_keys:
            assert config.exo_frames[k] is not None, f"exo_frames have frames for {k}"
            print(f"{k}_frames=\n{config.exo_frames[k]}")
            pts = get_all_pts(by_dev_id[k]["local_path"], idx_set=config.exo_frames[k])
            pts_by_key[k] = pts

    mobile_frames = config.mobile_frames
    if mobile_frames is not None:
        mobile_pts = get_all_pts(by_dev_id["mobile"]["local_path"], mobile_frames)
    else:
        mobile_pts = get_all_pts(by_dev_id["mobile"]["local_path"])
        mobile_fps = (
            get_fps(by_dev_id["mobile"]["local_path"])
            if config.mobile_fps is None
            else config.mobile_fps
        )
        msf = int(np.round(mobile_fps * config.frame_rate))

        nf = len(mobile_pts)
        config.mobile_frames = list(range(0, nf))[::msf]
        mobile_pts = get_all_pts(
            by_dev_id["mobile"]["local_path"], config.mobile_frames
        )

    pts_by_key["mobile"] = mobile_pts
    for dev_name, pts_set in tqdm(pts_by_key.items()):
        assert issorted(pts_set)
        print(dev_name, len(pts_set))
        frames = get_frames_list(by_dev_id[dev_name]["local_path"], pts_set)

        save_off_frames(
            frames, pts_set, f"{frame_out_dir}/{dev_name}", config.rot_mode == 2
        )


def _write_colmap_file(config: ColmapConfig, colmap_bin: str):
    colmap_bin_str = "${1:-" + colmap_bin + "}"
    gpu_idx_str = "${2:-0}"
    script = f"""
COLMAP_BIN={colmap_bin_str}
GPU_IDX={gpu_idx_str}
echo "Using $COLMAP_BIN"
echo "You can provide your own COLMAP binary by providing it as the first arg to this script"

echo "Using GPU: $GPU_IDX"
echo "You can provide -1 for CPU or another index as the second arg to this script"

SCRIPT_DIR=$(dirname -- $0)
DB_PATH=$SCRIPT_DIR/colmap.db
IN_FRAME_DIR=$SCRIPT_DIR/frames
MODEL_DIR=$SCRIPT_DIR/colmap_model
UNDIST_DIR=$SCRIPT_DIR/undistorted_images

mkdir $MODEL_DIR
mkdir $UNDIST_DIR


$COLMAP_BIN feature_extractor \\
   --database_path $DB_PATH \\
   --image_path $IN_FRAME_DIR \\
   --SiftExtraction.gpu_index $GPU_IDX \\
   --ImageReader.single_camera_per_folder 1 \\
   --ImageReader.camera_mode {config.camera_model}

$COLMAP_BIN exhaustive_matcher --database_path $DB_PATH

$COLMAP_BIN mapper \\
    --database_path $DB_PATH \\
    --image_path $IN_FRAME_DIR \\
    --output_path $MODEL_DIR \\
    --Mapper.ba_global_pba_gpu_index $GPU_IDX

$COLMAP_BIN model_converter \
    --input_path $MODEL_DIR/0 \
    --output_path $MODEL_DIR/0 \
    --output_type TXT

$COLMAP_BIN image_undistorter \\
    --image_path $IN_FRAME_DIR \\
    --input_path $MODEL_DIR/0 \\
    --output_path $SCRIPT_DIR \\
    --output_type COLMAP \\
    --max_image_size 2000

$COLMAP_BIN model_analyzer --path $MODEL_DIR/0 > $SCRIPT_DIR/analysis_0.txt
"""
    with open(os.path.join(get_colmap_data_dir(config), "run_colmap.sh"), "w") as out_f:
        out_f.write(script)

    with open(os.path.join(get_colmap_data_dir(config), "config.yaml"), "w") as out_f:
        out_f.write(OmegaConf.to_yaml(config))
    print("Config=")
    print(OmegaConf.to_yaml(config))


def issorted(xs):
    return sorted(xs) == xs


def get_frames_iter(video_path, pts_set):
    min_pts = min(pts_set)
    max_pts = max(pts_set)

    with av.open(stream_pathmgr.open(video_path)) as cont:
        v_stream = cont.streams.video[0]
        cont.seek(min_pts, stream=v_stream)
        for frame in cont.decode(video=0):
            if frame.pts > max_pts:
                break
            if frame.pts not in pts_set:
                continue
            if isinstance(frame, av.VideoFrame):
                x = frame.to_ndarray(format="rgb24")
                yield frame.pts, x


def get_frames_list(video_path, pts_set):
    min_pts = min(pts_set)
    max_pts = max(pts_set)

    frame_data = {}
    with av.open(stream_pathmgr.open(video_path)) as cont:
        v_stream = cont.streams.video[0]
        cont.seek(min_pts, stream=v_stream)
        for frame in cont.decode(video=0):
            if frame.pts > max_pts:
                break
            if frame.pts not in pts_set:
                continue
            if isinstance(frame, av.VideoFrame):
                x = frame.to_ndarray(format="rgb24")
                frame_data[frame.pts] = x

    assert len(frame_data) == len(pts_set)
    return frame_data


def save_off_frames(frames, pts_set, out_dir, rot90):
    os.makedirs(out_dir, exist_ok=True)
    for idx, pts in tqdm(enumerate(pts_set), total=len(pts_set)):
        frame_data = frames[pts]
        out_path = os.path.join(out_dir, f"{idx}.jpg")

        img = cv2.cvtColor(frame_data, cv2.COLOR_RGB2BGR)
        if rot90:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(out_path, img)


def get_all_pts(video_path, idx_set=None):
    with av.open(stream_pathmgr.open(video_path)) as cont:
        frame_pts = []
        idx = 0
        for frame in cont.demux(video=0):
            if frame.pts is None:
                continue
            if idx_set is not None and idx not in idx_set:
                idx += 1
                continue
            frame_pts.append(frame.pts)
            idx += 1
        return frame_pts


def extract_aria_frames(
    vrs_bin: str,
    aria_vrs_file_path: str,
    out_dir: str,
    synced_df: Optional[pd.DataFrame],
    stream_name="214-1",
    from_point: Optional[float] = None,
    to_point: Optional[float] = None,
):
    if synced_df is not None:
        assert from_point is None and to_point is None
        # TODO: get a better estimate for this
        t_end = synced_df["aria01_1201-2_capture_timestamp_ns"].iloc[0] / 10**9
        # !$vrs extract-images $data_path/aria01.mp4 + "214-1
        # " --before $t_end --to $input_aria_frames
        cmd = [
            vrs_bin,
            "extract-images",
            aria_vrs_file_path,
            "+",
            stream_name,
            "--before",
            f"{t_end}",
            "--to",
            out_dir,
        ]
        print("Running:")
        print(" ".join(cmd))
        subprocess.run(cmd)
    else:
        assert (
            to_point is not None or from_point is not None
        ), "will not exact all aria frames, please specify a region"
        cmd = [
            vrs_bin,
            "extract-images",
            aria_vrs_file_path,
            "+",
            stream_name,
        ]
        if to_point is not None:
            cmd += [
                "--before",
                f"{to_point}",
            ]
        if from_point is not None:
            cmd += ["--after", f"{from_point}"]

        cmd += ["--to", out_dir]
        print("Running:")
        print(" ".join(cmd))
        subprocess.run(cmd)


def subsample_aria_frames(
    input_aria_frame_dir,
    skip_num_frames,
    frame_out_dir,
    aria_frames_subset,
    config,
    aria_key,
):
    aria_frames = os.listdir(input_aria_frame_dir)
    out_frame_dir = os.path.join(frame_out_dir, "aria01")
    shutil.rmtree(out_frame_dir, ignore_errors=True)
    os.makedirs(out_frame_dir, exist_ok=True)

    f_idx = aria_frames_subset
    if f_idx is None:
        f_idx = list(range(len(aria_frames)))[::skip_num_frames]

    config.aria_frames[aria_key] = f_idx

    for idx in f_idx:
        f = aria_frames[idx]
        full_path = os.path.join(input_aria_frame_dir, f)
        out_path = os.path.join(out_frame_dir, f)
        if config.rot_mode == 1:
            img = cv2.imread(full_path)
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(out_path, img)
        else:
            shutil.copy(full_path, out_path)


def _download_s3_path(s3_path: str, out_path: str):
    assert s3_path.startswith("s3://")
    s3 = boto3.client("s3")
    no_s3_path = s3_path.split("s3://")[1]
    temp = no_s3_path.split("/")
    bucket_name = temp[0]
    key = "/".join(temp[1:])
    s3.download_file(bucket_name, key, out_path)


def create_video(video_path, pts_set, out_path, fps=30, codec="h264", flip_90=True):
    """
    This is used to construct a video, useful for feeding into EasyMocap
    """
    frames = get_frames_iter(video_path, pts_set)
    print("Received frames")

    with av.open(stream_pathmgr.open(video_path)) as in_container:
        in_video_stream = in_container.streams.video[0]
        width, height = in_video_stream.width, in_video_stream.height
        if width % 2 != 0:
            width -= 1
        elif height % 2 != 0:
            height -= 1
        pix_fmt = in_video_stream.pix_fmt

    v_frame = 0
    with av.open(out_path, mode="w") as out_cont:
        video_stream = out_cont.add_stream(codec, fps)
        if flip_90:
            video_stream.width = height
            video_stream.height = width
        else:
            video_stream.width = width
            video_stream.height = height
        video_stream.pix_fmt = pix_fmt
        out_cont.start_encoding()
        for _, frame in frames:
            img = frame[0:height, 0:width, :]
            if flip_90:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

            copy_frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in video_stream.encode(copy_frame):
                out_cont.mux(packet)
                v_frame += 1
    print(f"encoded {v_frame} frames, from given {len(pts_set)} frames")
    return v_frame


@hydra.main(config_path="configs", config_name=None)
def run_colmap_preprocess(config: ColmapConfig):
    print(f"Working Directory {os.getcwd()} - please make sure you're using abs paths")
    ret = produce_colmap_script(config)
    if config.run_colmap:
        subprocess.run(
            ["bash", os.path.join(get_colmap_data_dir(config), "run_colmap.sh")]
        )
    return ret


if __name__ == "__main__":
    run_colmap_preprocess()  # pyre-ignore
