# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import threading

import tqdm


class DownloadProgressBar:
    """
    Thread-safe progress bar for tracking downloads.
    """

    def __init__(self, total_size_bytes: int):
        self.__tqdm = tqdm.tqdm(
            # Uses size in bits since most people know their internet bandwidth in
            # Megabits/sec
            total=total_size_bytes * 8,
            unit="b",
            unit_scale=True,
            unit_divisor=1000 * 1000,
        )
        self.__lock = threading.Lock()

    def update(self, num_bytes: int) -> None:
        with self.__lock:
            self.__tqdm.update(n=num_bytes * 8)
