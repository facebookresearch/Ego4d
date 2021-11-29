# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
Information about contributing universities.
"""

UNIV_TO_BUCKET = {
    "bristol": "ego4d-bristol",
    "cmu": "ego4d-cmu",
    "frl_track_1_public": "ego4d-consortium-sharing",
    "georgiatech": "ego4d-georgiatech",
    "iiith": "ego4d-iiith",
    "indiana": "ego4d-indiana",
    "kaust": "ego4d-kaust",
    "minnesota": "ego4d-minnesota",
    "nus": "ego4d-speac",
    "unict": "ego4d-unict",
    "utokyo": "ego4d-utokyo",
    "uniandes": "ego4d-university-sa",
    "cmu_africa": "ego4d-universityaf",
}

BUCKET_TO_UNIV = {v: k for k, v in UNIV_TO_BUCKET.items()}
