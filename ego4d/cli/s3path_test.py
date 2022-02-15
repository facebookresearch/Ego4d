# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .s3path import bucket_and_key_from_path


def test_key_without_slashes():
    b, k = bucket_and_key_from_path("s3://bucket/key")
    assert b == "bucket"
    assert k == "key"


def test_key_with_slashes():
    b, k = bucket_and_key_from_path("s3://bucket/object/key")
    assert b == "bucket"
    assert k == "object/key"


def test_bucket_with_special_chars():
    b, k = bucket_and_key_from_path("s3://bucket-with.chars/object/key")
    assert b == "bucket-with.chars"
    assert k == "object/key"


def test_key_with_extension():
    b, k = bucket_and_key_from_path("s3://bucket-name/object/key.json")
    assert b == "bucket-name"
    assert k == "object/key.json"


def test_key_path_with_special_chars():
    b, k = bucket_and_key_from_path("s3://bucket-name/object$-/data/folder/key.json")
    assert b == "bucket-name"
    assert k == "object$-/data/folder/key.json"
