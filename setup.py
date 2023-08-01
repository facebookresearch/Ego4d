#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

from setuptools import find_packages, setup

setup(
    name="ego4d",
    version="1.4.1",
    author="FAIR",
    author_email="info@ego4d-data.org",
    description="Ego4D Dataset CLI",
    url="https://github.com/facebookresearch/Ego4d/",
    install_requires=[
        "boto3",
        "tqdm",
        "regex",
    ],
    tests_require=[
        "pytest",
        "moto",
    ],
    packages=find_packages(exclude=("tests", "tests.*")),
    entry_points={
        "console_scripts": [
            "ego4d=ego4d.cli.cli:main",
            "ego4d_internal = ego4d.internal.cli:main",
        ],
    },
)
