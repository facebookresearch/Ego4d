# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import boto3
from moto import mock_s3


@mock_s3
def test_s3():
    s3 = boto3.resource("s3")
    print(s3)
