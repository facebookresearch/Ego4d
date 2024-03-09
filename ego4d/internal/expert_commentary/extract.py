import os
import json
import shutil
import tempfile
import zipfile

from tqdm.auto import tqdm

RAW_COMMENTARY_ROOT = "/checkpoint/miguelmartin/expert_commentary/231211/raw_data_do_not_sync"
RAW_EXTRACTED_COMM_ROOT = (
    "/checkpoint/miguelmartin/expert_commentary/231211/raw_data_extracted"
)

def extract_commentaries(
    input_dir: str = RAW_COMMENTARY_ROOT,
    output_dir: str = RAW_EXTRACTED_COMM_ROOT,
):
    """
    NOTE: INTERNAL USAGE ONLY
    """
    seen_dirs = {}
    merge_dirs = 0
    bad_dirs = []
    for root, _, files in tqdm(os.walk(input_dir)):
        if "merge" in root.lower():
            merge_dirs += 1
        for file in files:
            with tempfile.TemporaryDirectory() as tempdir:
                if file == "data.json":  # already unextracted
                    extract_path = root
                elif file.endswith(".zip"):  # in expected zip form
                    zip_path = os.path.join(root, file)
                    extract_path = f"{tempdir}/temp"
                    try:
                        with zipfile.ZipFile(zip_path, "r") as zip_ref:
                            zip_ref.extractall(extract_path)
                    except zipfile.BadZipfile:
                        print(f"file: {file} is not a zip")
                        bad_dirs.append(file)
                        continue
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
                    shutil.rmtree(copy_to_path)
                shutil.copytree(extract_path, copy_to_path)
    return bad_dirs


if __name__ == "__main__":
    bad_dirs = extract_commentaries()
    print("Bad dirs= ", len(bad_dirs))
