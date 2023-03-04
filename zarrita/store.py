from typing import List, Optional, Tuple

import fsspec
import numpy as np


class Store:
    def multi_get(
        self, keys: List[Tuple[str, Optional[Tuple[int, int]]]]
    ) -> List[Optional[bytes]]:
        return [self.get(key, byte_range) for key, byte_range in keys]

    def get(
        self, key: str, byte_range: Optional[Tuple[int, int]] = None
    ) -> Optional[bytes]:
        raise NotImplementedError

    def multi_set(
        self, key_values: List[Tuple[str, bytes, Optional[Tuple[int, int]]]]
    ) -> None:
        for key, value, byte_range in key_values:
            self.set(key, value, byte_range)

    def set(
        self, key: str, value: bytes, byte_range: Optional[Tuple[int, int]] = None
    ) -> None:
        raise NotImplemented

    def delete(self, key: str) -> None:
        raise NotImplemented


class FileSystemStore(Store):
    def __init__(self, url: str, **storage_options):
        assert isinstance(url, str)

        # instantiate file system
        fs, root = fsspec.core.url_to_fs(url, **storage_options)
        self.fs = fs
        self.root = root.rstrip("/")

    def multi_get(
        self, keys: List[Tuple[str, Optional[Tuple[int, int]]]]
    ) -> List[Optional[bytes]]:
        return [self.get(key, byte_range) for key, byte_range in keys]

    def get(
        self, key: str, byte_range: Optional[Tuple[int, int]] = None
    ) -> Optional[bytes]:
        assert isinstance(key, str)
        path = f"{self.root}/{key}"

        try:
            value = (
                self.fs.cat_file(path, byte_range[0], byte_range[1])
                if byte_range
                else self.fs.cat_file(path)
            )
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return None

        return value

    def multi_set(
        self, key_values: List[Tuple[str, bytes, Optional[Tuple[int, int]]]]
    ) -> None:
        for key, value, byte_range in key_values:
            self.set(key, value, byte_range)

    def set(
        self, key: str, value: bytes, byte_range: Optional[Tuple[int, int]] = None
    ) -> None:
        assert isinstance(key, str)
        path = f"{self.root}/{key}"

        # ensure parent folder exists
        self.fs.mkdirs(self.fs._parent(path), exist_ok=True)

        # write data
        if byte_range:
            with self.fs.open(path, "r+b") as f:
                f.seek(byte_range[0])
                f.write(value)
        else:
            with self.fs.open(path, "wb") as f:
                f.write(value)

    def delete(self, key: str) -> None:
        path = f"{self.root}/{key}"
        if self.fs.exists(path):
            self.fs.rm(path)
