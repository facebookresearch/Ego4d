#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

from setuptools import find_packages, setup

setup(
    name="ego4d",
    version="1.0",
    author="FAIR",
    url="https://github.com/facebookresearch/Ego4d",
    install_requires=[
        "boto3",
        "tqdm",
        # "av",
        # "torch",
        # "torchvision",
        # "pytorch_lightning",
        # "matplotlib",
        # "simplejson",
        # "matplotlib",
        # "pandas",
    ],
    tests_require=[
        "pytest",
        "moto",
    ],
    packages=find_packages(exclude=("configs", "tests")),
)
