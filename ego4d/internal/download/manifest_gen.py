"""
This scripts generates the listing of files that the CLI will download.

It does so by partitioning a list of ManifestEntry's into parts. See the
`manifests` dict below; the keys are the names of each part of the dataset.

This should only be used internally by the Eng Team of the EgoExo dataset.
"""

import json
import os
from collections import defaultdict

from ego4d.internal.download.manifest import (
    manifest_dumps,
    ManifestEntry,
    PathSpecification,
)

from ego4d.internal.s3 import S3Downloader
from iopath.common.file_io import PathManager
from iopath.common.s3 import S3PathHandler
from tqdm.auto import tqdm

pathmgr = PathManager()  # for downloading files
pathmgr.register_handler(S3PathHandler(profile="default"))

base_release_dir = "egoexo-public/v1/"
release_dir = f"s3://ego4d-consortium-sharing/{base_release_dir}"
internal_release_dir = (
    "s3://ego4d-consortium-sharing/egoexo/releases/public_internal/v1/"
)


manifests = {
    "metadata": [],
    "annotations": [],
    "takes": [],
    "captures": [],
    "take_trajectory": [],
    "take_point_cloud": [],
    "take_vrs": [],  # TODO
    "capture_trajectory": [],
    "capture_eye_gaze": [],
    "capture_point_cloud": [],
    "downscaled_takes/448": [],
    "features/omnivore_video": [],
    # TODO: later date
    # "ego_pose_pseudo_gt": [],
    # "narrate_and_act_transc": [],
}

egoexo = {
    "released_takes": os.path.join(internal_release_dir, "released_takes.json"),
    "takes": os.path.join(internal_release_dir, "takes.json"),
    "takes_dropped": os.path.join(internal_release_dir, "takes_dropped.json"),
    "captures": os.path.join(internal_release_dir, "captures.json"),
    "physical_setting": os.path.join(internal_release_dir, "physical_setting.json"),
    "participants": os.path.join(internal_release_dir, "participants.json"),
    "visual_objects": os.path.join(internal_release_dir, "visual_objects.json"),
}

downloader = S3Downloader("default")
print("Downloading metadata")
for k, out_path in tqdm(egoexo.items()):
    s3_path = os.path.join(release_dir, f"{k}.json")
    if not pathmgr.exists(s3_path):
        continue
    # print(s3_path, out_path)
    paths = [
        PathSpecification(
            s3_path,
            f"{k}.json",
        )
    ]
    manifests["metadata"].append(
        ManifestEntry(
            uid=k,
            paths=paths,
        )
    )

for k, v in egoexo.items():
    if pathmgr.exists(v):
        egoexo[k] = json.load(pathmgr.open(v))
    else:
        egoexo[k] = []

s3_buckets = set()
for c in tqdm(egoexo["captures"]):
    s3_bucket = c["_s3_root_dir"].split("/")[2]
    s3_buckets.add(s3_bucket)

take_by_take_uid = {x["take_uid"]: x for x in egoexo["takes"] + egoexo["takes_dropped"]}
take_name_to_uid = {
    t["take_name"]: t["take_uid"] for t in (egoexo["takes"] + egoexo["takes_dropped"])
}

capture_cam_id_to_is_ego = {
    (c["capture_uid"], cam["cam_id"]): cam["is_ego"]
    for c in egoexo["captures"]
    for cam in c["cameras"]
}

split_file_path = os.path.join(release_dir, "annotations/splits.json")
split_data = json.load(pathmgr.open(split_file_path))
take_uid_to_splits = {k: [v] for k, v in split_data["take_uid_to_split"].items()}

capture_uid_to_splits = defaultdict(set)
for take_uid, split in split_data["take_uid_to_split"].items():
    if take_uid not in take_by_take_uid:
        print("Missing", take_uid)
        continue
    capture_uid = take_by_take_uid[take_uid]["capture"]["capture_uid"]
    capture_uid_to_splits[capture_uid].add(split.lower())

take_uid_to_benchmarks = defaultdict(list)
take_uid_to_benchmarks.update(split_data.get("take_uid_to_benchmark", {}))

s3_buckets = {x["university_id"] for x in egoexo["takes"] if x["take_uid"] in egoexo["released_takes"]}

manifests["downscaled_takes/448"] = []
if "downscaled_takes/448" in manifests:
    by_take = defaultdict(list)
    for bucket in s3_buckets:
        ds_base_dir = os.path.join(f"s3://{bucket}/{base_release_dir}", "downscaled_takes/448/")
        print(ds_base_dir)
        for bn, path in downloader.ls(ds_base_dir, recursive=True):
            take_name = path.split("downscaled_takes/")[1].split("/")[1]
            if take_name not in take_name_to_uid:
                continue
            take_uid = take_name_to_uid[take_name]
            if take_uid not in egoexo["released_takes"]:
                continue
            by_take[take_uid].append(
                PathSpecification(
                    source_path=path,
                    relative_path=f"takes/{take_name}/frame_aligned_videos/downscaled/448/{bn}",
                    file_type="mp4",
                    views=None,  # TODO
                )
            )

    for take_uid, paths in by_take.items():
        manifests["downscaled_takes/448"].append(
            ManifestEntry(
                uid=take_uid,
                paths=paths,
                benchmarks=take_uid_to_benchmarks.get(take_uid, None),
                splits=take_uid_to_splits.get(take_uid, None),
            )
        )

path = None
take_uid = None
take_name = None
by_take_and_dir = {}
for bucket in tqdm(s3_buckets):
    base_dir = os.path.join(f"s3://{bucket}/{base_release_dir}", "takes/")
    for bn, path in downloader.ls(base_dir, recursive=True):
        take_name = path.split("takes/")[1].split("/")[0]
        if take_name not in take_name_to_uid:
            continue
        take_uid = take_name_to_uid[take_name]
        if take_uid not in egoexo["released_takes"]:
            continue
        dirname = os.path.basename(os.path.dirname(path))
        key = (take_uid, dirname)
        if key not in by_take_and_dir:
            by_take_and_dir[key] = []
        by_take_and_dir[key].append((bn, path))


for manifest_key in ["takes", "takes_dropped"]:
    if manifest_key not in manifests:
        continue

    for t in tqdm(egoexo[manifest_key]):
        take_uid = t["take_uid"]
        if take_uid not in egoexo["released_takes"]:
            continue

        root_dir = t["root_dir"]
        paths = []
        for streams in t["frame_aligned_videos"].values():
            for vid in streams.values():
                if vid["_s3_path"] is None:
                    continue
                uid = vid["clip_uid"]
                vid_k = (t["capture_uid"], vid.get("cam_id"))
                is_ego = capture_cam_id_to_is_ego.get(vid_k)
                views = None
                if is_ego is not None:
                    views = ["ego"] if is_ego else ["exo"]
                paths.append(
                    PathSpecification(
                        source_path=vid["_s3_path"],
                        relative_path=os.path.join(root_dir, vid["relative_path"]),
                        views=views,
                        universities=[t["university_name"]],
                        file_type="mp4",
                    )
                )

        # add vrs path (no rgb stream)
        if t["has_trimmed_vrs"]:
            sp = t["_vrs_s3_path"]
            assert sp is not None
            paths.append(
                PathSpecification(
                    source_path=sp,
                    relative_path=os.path.join(root_dir, t["vrs_relative_path"]),
                    views=["ego"],
                    universities=[t["university_name"]],
                    file_type="vrs",
                )
            )
            for d in ["trajectory", "eye_gaze"]:
                d_paths = []
                pc_paths = []
                key = (take_uid, d)
                if key not in by_take_and_dir:
                    print("Skipping", key)
                    continue
                for bn, p in by_take_and_dir[key]:
                    ext = os.path.splitext(bn)[-1]
                    if "semidense" in bn:
                        pc_paths.append(
                            PathSpecification(
                                source_path=p,
                                relative_path=os.path.join(root_dir, d, bn),
                                views=None,
                                universities=[t["university_name"]],
                                file_type=ext[1:]
                            )
                        )
                    else:
                        d_paths.append(
                            PathSpecification(
                                source_path=p,
                                relative_path=os.path.join(root_dir, d, bn),
                                views=None,
                                universities=[t["university_name"]],
                                file_type=ext[1:]
                            )
                        )

                paths.extend(pc_paths)
                paths.extend(d_paths)
                if len(pc_paths) > 0:
                    manifests["take_point_cloud"].append(
                        ManifestEntry(
                            uid=take_uid,
                            paths=pc_paths,
                            benchmarks=take_uid_to_benchmarks.get(take_uid, None),
                            splits=take_uid_to_splits.get(take_uid, None),
                        )
                    )
                
                if d != "eye_gaze":
                    assert d == "trajectory"
                    manifests["take_trajectory"].append(
                        ManifestEntry(
                            uid=take_uid,
                            paths=d_paths,
                            benchmarks=take_uid_to_benchmarks.get(take_uid, None),
                            splits=take_uid_to_splits.get(take_uid, None),
                        )
                    )

            manifests["take_vrs"].append(
                ManifestEntry(
                    uid=take_uid,
                    paths=[PathSpecification(
                        source_path=t["_vrs_all_streams_s3_path"],
                        relative_path=os.path.join(root_dir, t["vrs_all_streams_relative_path"]),
                        views=["ego"],
                        universities=[t["university_name"]],
                        file_type="vrs",
                    )],
                    benchmarks=take_uid_to_benchmarks.get(take_uid, None),
                    splits=take_uid_to_splits.get(take_uid, None),
                )
            )


        manifests[manifest_key].append(
            ManifestEntry(
                uid=take_uid,
                paths=paths,
                benchmarks=take_uid_to_benchmarks.get(take_uid, None),
                splits=take_uid_to_splits.get(take_uid, None),
            )
        )


for c in tqdm(egoexo["captures"]):
    s3_bucket = c["_s3_root_dir"].split("/")[2]
    root_dir = c["root_dir"]
    traj_fp = c["_trajectory_s3_dir"]
    post_fp = c["_postsurvery_s3_path"]
    eye_gaze_fp = c["_gaze_s3_dir"]
    timesync_fp = c["_timesync_s3_path"]
    ts_dir = os.path.join(c["_s3_root_dir"], "timesync")
    traj_files = downloader.ls(traj_fp + "/", max_keys=1000) if traj_fp else []
    eye_gaze_files = (
        downloader.ls(eye_gaze_fp + "/", max_keys=1000) if eye_gaze_fp else []
    )
    ts_files = downloader.ls(ts_dir + "/", max_keys=1000)
    universities = [c["university_name"]]

    traj_paths = []
    point_cloud_paths = []
    for bn, s3_path in traj_files:
        ext = os.path.splitext(bn)[-1]
        assert ext.startswith(".")
        if "semidense" in bn:
            point_cloud_paths.append(
                PathSpecification(
                    source_path=s3_path,
                    relative_path=os.path.join(root_dir, "trajectory", bn),
                    universities=universities,
                    file_type=ext[1:],
                    views=None,
                )
            )
        else:
            traj_paths.append(
                PathSpecification(
                    source_path=s3_path,
                    relative_path=os.path.join(root_dir, "trajectory", bn),
                    universities=universities,
                    file_type=ext[1:],
                    views=None,
                )
            )

    capture_uid = c["capture_uid"]
    manifests["capture_trajectory"].append(
        ManifestEntry(
            uid=capture_uid,
            paths=traj_paths,
            benchmarks=take_uid_to_benchmarks[capture_uid],
            splits=capture_uid_to_splits[capture_uid],
        )
    )
    manifests["capture_point_cloud"].append(
        ManifestEntry(
            uid=capture_uid,
            paths=point_cloud_paths,
            benchmarks=take_uid_to_benchmarks[capture_uid],
            splits=capture_uid_to_splits[capture_uid],
        )
    )
    eye_gaze_paths = []
    for bn, s3_path in eye_gaze_files:
        ext = os.path.splitext(bn)[-1]
        assert ext.startswith(".")
        eye_gaze_paths.append(
            PathSpecification(
                source_path=s3_path,
                relative_path=os.path.join(root_dir, "eye_gaze", bn),
                universities=universities,
                file_type=ext[1:],
                views=None,
            )
        )
    manifests["capture_eye_gaze"].append(
        ManifestEntry(
            uid=capture_uid,
            paths=eye_gaze_paths,
            benchmarks=take_uid_to_benchmarks[capture_uid],
            splits=capture_uid_to_splits[capture_uid],
        )
    )

    capture_paths = []
    if post_fp is not None:
        capture_paths.append(
            PathSpecification(
                source_path=post_fp,
                relative_path=os.path.join(root_dir, "post_surveys.csv"),
                universities=universities,
                file_type="csv",
                views=None,
            )
        )

    if timesync_fp is not None:
        capture_paths.append(
            PathSpecification(
                source_path=timesync_fp,
                relative_path=os.path.join(root_dir, "timesync.csv"),
                universities=universities,
                file_type="csv",
                views=None,
            )
        )
    for bn, s3_path in ts_files:
        ext = os.path.splitext(bn)[-1]
        assert ext.startswith(".")
        capture_paths.append(
            PathSpecification(
                source_path=s3_path,
                relative_path=os.path.join(root_dir, "timesync", f"{bn}"),
                universities=universities,
                file_type=ext[1:],
                views=None,
            )
        )

    manifests["captures"].append(
        ManifestEntry(
            uid=c["capture_uid"],
            paths=capture_paths,
            benchmarks=take_uid_to_benchmarks[capture_uid],
            splits=capture_uid_to_splits[capture_uid],
        )
    )


if "annotations" in manifests:
    manifests["annotations"] = []
    annotations = downloader.ls(os.path.join(release_dir, "annotations/"))
    for bn, s3_path in annotations:
        if len(bn) == 0:
            continue
        if bn == "manifest.json":
            continue

        print(bn)
        benchmarks = []
        benchmark_name = "_".join(bn.split("_")[0:-1])
        benchmarks = [benchmark_name] if len(benchmark_name) > 0 else None
        if "proficiency" in benchmark_name:
            benchmarks.append("proficiency")

        manifests["annotations"].append(
            ManifestEntry(
                uid="_".join(bn.split("_")[:-1]),
                paths=[
                    PathSpecification(
                        source_path=s3_path,
                        relative_path=f"annotations/{bn}",
                    )
                ],
                benchmarks=benchmarks,
            )
        )

    egopose_part_subdirs = {
        "annotations": {
            "subdirs": [
                "annotation",
                "camera_pose",
            ],
            "take_uid_as_key": False,
        },
        "ego_pose_pseudo_gt": {
            "subdirs": ["automatic"],
            "take_uid_as_key": True,
        },
    }
    egopose_base_dir = os.path.join(release_dir, "annotations/ego_pose_latest/")
    for body_type in ["body", "hand"]:
        subdir = os.path.join(egopose_base_dir, body_type)

        annotation_files = []
        for manifest_key, metadata in egopose_part_subdirs.items():

            for ann_dir in metadata["subdirs"]:
                relative_subdir = os.path.join(
                    "annotations/ego_pose/", body_type, ann_dir
                )
                ann_files = downloader.ls(os.path.join(subdir, ann_dir + "/"))
                for bn, s3_path in ann_files:
                    if len(bn) == 0:
                        continue
                    if bn == "manifest.json":
                        continue
                    take_uid = os.path.splitext(bn)[0]
                    uid = take_uid
                    manifests[manifest_key].append(
                        ManifestEntry(
                            # e.g.
                            # - ego_pose/body/camera_pose
                            # - ego_pose/body/annotation
                            # - ego_pose/hand/camera_pose
                            # - ego_pose/hand/annotation
                            name="/".join(["ego_pose", body_type, ann_dir]),
                            uid=uid,
                            benchmarks=[
                                "egopose",
                                "ego_pose",
                                f"{body_type}_pose",
                                f"ego_{body_type}_pose",
                            ],
                            splits=take_uid_to_splits.get(take_uid, None),
                            paths=[
                                PathSpecification(
                                    source_path=s3_path,
                                    relative_path=f"{relative_subdir}/{bn}",
                                )
                            ],
                        )
                    )

if "features/omnivore_video" in manifests:
    for feature_name in ["omnivore_video"]:
        feature_files = downloader.ls(
            os.path.join(release_dir, "features/", feature_name + "/")
        )
        by_take = defaultdict(list)
        for file_name, path in feature_files:
            if file_name.endswith("yaml"):
                manifests[f"features/{feature_name}"].append(
                    ManifestEntry(
                        uid="omnivore_video_config",
                        paths=[
                            PathSpecification(
                                source_path=path,
                                relative_path=f"features/{feature_name}/{file_name}",
                            )
                        ],
                    )
                )
                continue

            if file_name == "" or file_name == "manifest.json":
                continue
            take_uid, cam_id, stream_id = file_name.split("_")
            if take_uid not in egoexo["released_takes"]:
                continue

            t = take_by_take_uid[take_uid]
            vid_k = (t["capture_uid"], vid.get("cam_id"))
            is_ego = capture_cam_id_to_is_ego.get(vid_k)
            views = None
            if is_ego is not None:
                views = ["ego"] if is_ego else ["exo"]
            by_take[take_uid].append(
                PathSpecification(
                    source_path=path,
                    relative_path=f"features/{feature_name}/{file_name}",
                    views=views,
                    universities=[t["university_name"]],
                    file_type="pt",
                )
            )

        for take_uid, paths in by_take.items():
            manifests[f"features/{feature_name}"].append(
                ManifestEntry(
                    uid=take_uid,
                    paths=paths,
                    benchmarks=take_uid_to_benchmarks.get(take_uid, None),
                    splits=take_uid_to_splits.get(take_uid, None),
                )
            )

na_transc = downloader.ls(
    os.path.join(release_dir, "annotations/narrate_and_act_transc/"), recursive=True
)
by_take = defaultdict(list)
for file_name, path in na_transc:
    if file_name == ".DS_Store":
        continue
    tn = os.path.splitext(file_name)[0]
    if tn not in take_name_to_uid:
        continue
    take_uid = take_name_to_uid[tn]
    by_take[take_uid].append(
        PathSpecification(
            source_path=path,
            relative_path=f"annotations/narrate_and_act_transc/{tn}/{file_name}",
            views=None,
            universities=None,  # TODO
        )
    )

if "narrate_and_act_transc" in manifests:
    for take_uid, paths in by_take.items():
        manifests["narrate_and_act_transc"].append(
            ManifestEntry(
                uid=take_uid,
                paths=paths,
                benchmarks=take_uid_to_benchmarks.get(take_uid, None),
                splits=take_uid_to_splits.get(take_uid, None),
            )
        )


all_bs = set()
for k, v in manifests.items():
    out_dir = os.path.join(release_dir, f"{k}/")
    # out_dir = os.path.join(f"s3://ego4d-consortium-sharing/egoexo-public/v1_test", f"{k}/")
    manifest_file = os.path.join(out_dir, "manifest.json")
    for m in v:
        for b in m.benchmarks or []:
            all_bs.add(b)

    print(k, len(v))
    print(f"will output to: {out_dir}")
    print(f"{manifest_file}")
    pathmgr.mkdirs(out_dir, exist_ok=True)
    pathmgr.rm(manifest_file)
    with pathmgr.open(manifest_file, "w") as out_f:
        json_data = manifest_dumps(v)
        out_f.write(json_data)
