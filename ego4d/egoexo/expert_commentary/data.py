import json
import os
from typing import List

from tqdm.auto import tqdm


def load_all_raw_commentaries(raw_extracted_dir: str) -> List[str]:
    result = [os.path.join(raw_extracted_dir, x) for x in os.listdir(raw_extracted_dir)]
    return sorted(result)


def load_uniq_commentaries(raw_extracted_dir: str) -> List[str]:
    result = {}
    for root in tqdm(os.listdir(raw_extracted_dir)):
        data = json.load(open(os.path.join(raw_extracted_dir, root, "data.json")))
        key = (data["user_id"], data["video_name"])
        if key in result:
            ds_curr = result[key]["data"]["ds"]
            if ds_curr < data["ds"]:
                result[key] = {
                    "dir": os.path.join(raw_extracted_dir, root),
                    "data": data,
                }
        else:
            result[key] = {
                "dir": os.path.join(raw_extracted_dir, root),
                "data": data,
            }
    return [v["dir"] for v in result.values()]


