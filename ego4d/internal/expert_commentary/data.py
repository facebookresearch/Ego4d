import json
import os
import shutil
import tempfile
import zipfile
from typing import List

from tqdm.auto import tqdm

RAW_COMMENTARY_ROOT = "/checkpoint/miguelmartin/expert_commentary/raw_data"
RAW_EXTRACTED_COMM_ROOT = (
    "/checkpoint/miguelmartin/expert_commentary/raw_data_extracted"
)


def extract_commentaries(
    input_dir: str = RAW_COMMENTARY_ROOT, output_dir: str = RAW_EXTRACTED_COMM_ROOT
):
    seen_dirs = dict()
    merge_dirs = 0
    for root, _, files in tqdm(os.walk(input_dir)):
        if "merge" in root.lower():
            merge_dirs += 1
            print("merge root")
        for file in files:
            print(file)
            with tempfile.TemporaryDirectory() as tempdir:
                if file == "data.json":  # already unextracted
                    extract_path = root
                elif file.endswith(".zip"):  # in expected zip form
                    zip_path = os.path.join(root, file)
                    extract_path = f"{tempdir}/temp"
                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        zip_ref.extractall(extract_path)
                else:
                    continue

                data = json.load(open(os.path.join(extract_path, "data.json")))
                copy_to_path = None
                for i in range(100):
                    uid_path = f"{data['user_id']}_{data['video_name']}_{i}"
                    copy_to_path = os.path.join(output_dir, uid_path)
                    if copy_to_path not in seen_dirs:
                        break
                assert copy_to_path and copy_to_path not in seen_dirs
                seen_dirs[copy_to_path] = True
                if os.path.exists(copy_to_path):
                    continue
                shutil.copytree(extract_path, copy_to_path)


def load_raw_commentaries(raw_extracted_dir: str) -> List[str]:
    result = []
    for root, _, files in os.walk(raw_extracted_dir):
        for file in files:
            if file == "data.json":
                result.append(root)
                break
    return sorted(result)
