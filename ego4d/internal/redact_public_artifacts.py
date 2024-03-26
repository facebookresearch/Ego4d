import json
import os
import shutil

from tqdm.auto import tqdm

release_out_base_dir = "egoexo-public/v2"
public_takes = json.load(open("/large_experiments/egoexo/v2/takes.json"))
take_by_name = {x["take_name"]: x for x in public_takes}
take_by_uid = {x["take_uid"]: x for x in public_takes}

feature_name = "maws_clip_2b"
feature_in_f = f"/checkpoint/miguelmartin/egoexo_features/{feature_name}"
feature_out_f = f"/checkpoint/miguelmartin/egoexo_features/{feature_name}_public"
downscaled_takes_in_f = "/checkpoint/miguelmartin/egoexo/v2/downscaled_takes/takes/"
downscaled_takes_out_f = (
    "/checkpoint/miguelmartin/egoexo/v2/downscaled_takes/takes_by_uni/"
)

features = os.listdir(feature_in_f)
public_features = []
covered_takes = set()
for f in features:
    if f.endswith("pt"):
        uid = f.split("_")[0]
        if uid in take_by_uid:
            public_features.append((uid, f))
            covered_takes.add(uid)
    else:
        public_features.append((None, f))

print(
    f"""Feature Stats: 
# features = {len(features)}
# public features = {len(public_features)}
missing public = {len(take_by_uid.keys() - covered_takes)}

feature in path = {feature_in_f}
feature out path = {feature_out_f}
"""
)

if False:
    os.makedirs(feature_out_f, exist_ok=True)
    for take_uid, f in tqdm(public_features):
        src = os.path.join(feature_in_f, f)
        dst = os.path.join(feature_out_f, f)
        if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src):
            shutil.copy(src, dst)

print("please run:")
print(
    f"aws s3 sync {feature_out_f} s3://ego4d-consortium-sharing/{release_out_base_dir}/features/{feature_name}/"
)
print()

fs = os.listdir(downscaled_takes_in_f)
fs_to_copy = []
for f in fs:
    if f not in take_by_name:
        continue
    fs_to_copy.append(f)

set(take_by_name) - set(fs_to_copy)

buckets = set()
for f in tqdm(fs_to_copy):
    take = take_by_name[f]
    bucket = take["university_id"]
    buckets.add(bucket)

    src = os.path.join(downscaled_takes_in_f, f)
    dst = os.path.join(downscaled_takes_out_f, bucket, f)
    if os.path.exists(dst):
        continue
    shutil.copytree(src, dst)

with open("downscaled_takes_sync.sh", "w") as out_f:
    out_f.write("#/bin/bash\n")
    for b in buckets:
        p = os.path.join(downscaled_takes_out_f, b)
        out_f.write(
            f"aws s3 sync {p} s3://{b}/egoexo-public/v2/downscaled_takes/448/\n"
        )
