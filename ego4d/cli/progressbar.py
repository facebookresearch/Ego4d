# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import threading
from typing import Optional

import tqdm


class DownloadProgressBar:
    """
    Thread-safe progress bar for tracking downloads based on bytes.
    """

    def __init__(self, total_size_bytes: Optional[int]):
        self.__tqdm = tqdm.tqdm(
            total=total_size_bytes if total_size_bytes else None,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        )
        self.__lock = threading.Lock()

    def update(self, num_bytes: int) -> None:
        with self.__lock:
            self.__tqdm.update(n=num_bytes)
