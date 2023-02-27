import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional

import av

import boto3
import cv2
import hydra
import numpy as np
import pandas as pd

from ego4d.internal.s3 import StreamPathMgr

from omegaconf import OmegaConf

from tqdm.auto import tqdm

pathmgr = StreamPathMgr()


VRS_BIN = "/private/home/miguelmartin/repos/vrs/build/tools/vrs/vrs"
COLMAP_BIN = "/private/home/miguelmartin/repos/colmap/build/src/exe/colmap"


@dataclass
class ColmapConfig:
    # TODO: more COLMAP hparams
    uni_name: str
    take_id: str
    data_dir: str

    use_gpu: bool
    sync_exo_views: bool
    rot_mode: int

    camera_model: str
    include_aria: bool

    exo1_frames: Optional[List[int]] = None
    exo2_frames: Optional[List[int]] = None
    exo3_frames: Optional[List[int]] = None
    exo4_frames: Optional[List[int]] = None
    mobile_frames: Optional[List[int]] = None
    aria_frames: Optional[List[int]] = None
    aria_last_walkthrough_sec: Optional[float] = None
    aria_use_sync_info: bool = False

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
    force_download: bool = False
    download_video_files: bool = False  # if false then use presigned urls


def get_video_data_dir(config) -> str:
    return os.path.join(config.data_dir, f"{config.uni_name}_{config.take_id}")


def get_timesync_path(config) -> str:
    return os.path.join(get_video_data_dir(config), "timesync.csv")


def get_colmap_data_dir(config) -> str:
    name = config.name
    if name is None:
        name = f"g{int(config.use_gpu)}_s{int(config.sync_exo_views)}_r{config.rot_mode}_a{int(config.include_aria)}_fr{config.frame_rate:.3f}"
    return os.path.join(
        config.data_dir, "colmap", f"{config.uni_name}_{config.take_id}", name
    )


def frames_for_region(start_frame: int, end_frame: int, skip_frames: int) -> List[int]:
    # [start, end]
    return list(range(start_frame, end_frame + 1))[0::skip_frames]


def get_fps(fp: str) -> float:
    with av.open(pathmgr.open(fp)) as cont:
        return float(cont.streams[0].average_rate)


def produce_colmap_script(
    config: ColmapConfig,
    skip_frame_gen: bool = False,
):
    vrs_bin = config.vrs_bin
    colmap_bin = config.colmap_bin
    if vrs_bin is None:
        vrs_bin = VRS_BIN
    if colmap_bin is None:
        colmap_bin = COLMAP_BIN

    print("Getting data ready...")
    _get_data_ready(config, config.force_download, vrs_bin, skip_frame_gen)
    print(f"Writing colmap script using {colmap_bin}")
    _write_colmap_file(config, colmap_bin)
    print(f"Please run {os.path.join(get_colmap_data_dir(config), 'run_colmap.sh')}")


def _get_data_ready(
    config: ColmapConfig, force_download: bool, vrs_bin: str, skip_frame_gen: bool
):
    by_dev_id = _download_data(config, force_download)
    if not skip_frame_gen:
        _extract_frames(config, by_dev_id, vrs_bin)


def _download_data(config, force_download):
    os.makedirs(get_video_data_dir(config), exist_ok=True)

    metadata_path_local = os.path.join(get_video_data_dir(config), "metadata.json")
    _get_file_from_s3(
        f"s3://ego4d-consortium-sharing/internal/egoexo_pilot/{config.uni_name}/{config.take_id}/metadata.json",
        metadata_path_local,
    )
    meta = json.load(open(metadata_path_local))

    timesync_dl_path = meta["timesync_csv_path"]
    if timesync_dl_path is not None:
        print("timesync path=", timesync_dl_path)
        _get_file_from_s3(timesync_dl_path, get_timesync_path(config))

    by_dev_id = {}
    for v in tqdm(meta["videos"]):
        by_dev_id[v["device_id"]] = v
        s3_path = v["s3_path"]
        if config.download_video_files or "aria" in v["device_id"]:
            path = os.path.join(get_video_data_dir(config), v["device_id"])
            by_dev_id[v["device_id"]]["local_path"] = path
            if os.path.exists(path) and not force_download:
                continue
            print(f"Fetching {s3_path} to {path}")
            _get_file_from_s3(s3_path, path)
        else:
            by_dev_id[v["device_id"]]["local_path"] = s3_path
    return by_dev_id


def _extract_frames(config, by_dev_id, vrs_bin):
    frame_out_dir = os.path.join(get_colmap_data_dir(config), "frames")
    shutil.rmtree(frame_out_dir, ignore_errors=True)
    os.makedirs(frame_out_dir, exist_ok=True)

    synced_df = None
    if config.sync_exo_views or config.aria_last_walkthrough_sec is None:
        timesync_df = pd.read_csv(get_timesync_path(config))
        cam01_idx = timesync_df.cam01_global_time.first_valid_index()
        cam02_idx = timesync_df.cam02_global_time.first_valid_index()
        cam03_idx = timesync_df.cam03_global_time.first_valid_index()
        cam04_idx = timesync_df.cam04_global_time.first_valid_index()
        fsf_idx = max(cam01_idx, cam02_idx, cam03_idx, cam04_idx)  # first synced frame
        synced_df = timesync_df.iloc[fsf_idx:]

    if config.include_aria:
        with tempfile.TemporaryDirectory() as tempdir:
            aria_frame_path = os.path.join(tempdir, "aria01_frames")
            os.makedirs(aria_frame_path, exist_ok=True)
            extract_aria_frames(
                vrs_bin=vrs_bin,
                aria_vrs_file_path=by_dev_id["aria01"]["local_path"],
                out_dir=aria_frame_path,
                synced_df=synced_df,
                until_point=config.aria_last_walkthrough_sec,
            )
            skip_aria = (
                int(np.round(config.aria_fps * config.frame_rate))
                if config.aria_fps is not None and config.frame_rate is not None
                else None
            )
            subsample_aria_frames(
                aria_frame_path, skip_aria, frame_out_dir, config.aria_frames, config
            )

    exo_fps = get_fps(by_dev_id["cam01"]["local_path"])
    if config.sync_exo_views:
        f1 = config.exo_from_frame
        f2 = config.exo_to_frame
        skip_num_frames = int(np.round(exo_fps * config.frame_rate))
        if config.exo1_frames is not None:
            raise AssertionError("cannot use exo1_frames in sync mode")
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
    else:
        if config.exo_from_frame is not None:
            assert (
                config.exo1_frames is None
                and config.exo2_frames is None
                and config.exo3_frames is None
                and config.exo4_frames is None
            ), "if given exo_from_frame and exo_to_frame - you cannot supply frames"
            assert config.exo_to_frame is not None, "need to_frame"
            assert config.frame_rate is not None, "need frame rate"
            f1 = config.exo_from_frame
            f2 = config.exo_to_frame
            skip_num_frames = int(np.round(exo_fps * config.frame_rate))

            frames = list(range(f1, f2))[::skip_num_frames]
            config.exo1_frames = frames
            config.exo2_frames = frames
            config.exo3_frames = frames
            config.exo4_frames = frames
        elif config.exo1_frames is not None:
            assert (
                config.exo1_frames is not None
                and config.exo2_frames is not None
                and config.exo3_frames is not None
                and config.exo4_frames is not None
            ), "if given exo1_frames, assume you to give all other frames"

        print(f"exo1_frames=\n{config.exo1_frames}")
        print(f"exo2_frames=\n{config.exo2_frames}")
        print(f"exo3_frames=\n{config.exo3_frames}")
        print(f"exo4_frames=\n{config.exo4_frames}")
        cam01_pts = get_all_pts(
            by_dev_id["cam01"]["local_path"], idx_set=config.exo1_frames
        )
        cam02_pts = get_all_pts(
            by_dev_id["cam02"]["local_path"], idx_set=config.exo2_frames
        )
        cam03_pts = get_all_pts(
            by_dev_id["cam03"]["local_path"], idx_set=config.exo3_frames
        )
        cam04_pts = get_all_pts(
            by_dev_id["cam04"]["local_path"], idx_set=config.exo4_frames
        )

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

    assert issorted(cam01_pts)
    assert issorted(cam02_pts)
    assert issorted(cam03_pts)
    assert issorted(cam04_pts)
    assert issorted(mobile_pts)

    devices = ["cam01", "cam02", "cam03", "cam04", "mobile"]
    pts_data = [cam01_pts, cam02_pts, cam03_pts, cam04_pts, mobile_pts]
    for dev_name, pts_set in tqdm(zip(devices, pts_data)):
        print(dev_name, len(pts_set))
        frames = get_frames_list(by_dev_id[dev_name]["local_path"], pts_set)
        save_off_frames(
            frames, pts_set, f"{frame_out_dir}/{dev_name}", config.rot_mode == 2
        )


def _write_colmap_file(config: ColmapConfig, colmap_bin: str):
    gpu_idx = 0 if config.use_gpu else -1
    colmap_bin_str = "${1:-" + colmap_bin + "}"
    script = f"""
COLMAP_BIN={colmap_bin_str}
echo "Using $COLMAP_BIN"
echo "You can provide your own COLMAP binary by providing it as an arg to this script"

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
   --SiftExtraction.gpu_index {gpu_idx} \\
   --ImageReader.single_camera_per_folder 1 \\
   --ImageReader.camera_mode {config.camera_model}

$COLMAP_BIN exhaustive_matcher --database_path $DB_PATH

$COLMAP_BIN mapper \\
    --database_path $DB_PATH \\
    --image_path $IN_FRAME_DIR \\
    --output_path $MODEL_DIR \\
    --Mapper.ba_global_pba_gpu_index {gpu_idx}

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

    with av.open(pathmgr.open(video_path)) as cont:
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
    with av.open(pathmgr.open(video_path)) as cont:
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
    with av.open(pathmgr.open(video_path)) as cont:
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
    until_point: Optional[float] = None,
):
    if synced_df is not None and until_point is None:
        t_end = synced_df["aria01_1201-2_capture_timestamp_ns"].iloc[0] / 10**9
        # !$vrs extract-images $data_path/aria01.mp4 + "214-1" --before $t_end --to $input_aria_frames
        subprocess.run(
            [
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
        )
    else:
        assert until_point is not None, "will not exact all aria frames"
        t_end = until_point
        subprocess.run(
            [
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
        )


def subsample_aria_frames(
    input_aria_frame_dir, skip_num_frames, frame_out_dir, aria_frames_subset, config
):
    aria_frames = os.listdir(input_aria_frame_dir)
    out_frame_dir = os.path.join(frame_out_dir, "aria01")
    shutil.rmtree(out_frame_dir, ignore_errors=True)
    os.makedirs(out_frame_dir, exist_ok=True)

    f_idx = aria_frames_subset
    if f_idx is None:
        f_idx = list(range(len(aria_frames)))[::skip_num_frames]

    config.aria_frames = f_idx

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


def _get_file_from_s3(path: str, out_path: str):
    s3 = boto3.client("s3")
    no_s3_path = path.split("s3://")[1]
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

    with av.open(pathmgr.open(video_path)) as in_container:
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
        for _, frame in tqdm(frames, total=len(pts_set)):
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
    produce_colmap_script(config)
    if config.run_colmap:
        subprocess.run(
            ["bash", os.path.join(get_colmap_data_dir(config), "run_colmap.sh")]
        )


if __name__ == "__main__":
    run_colmap_preprocess()  # pyre-ignore
