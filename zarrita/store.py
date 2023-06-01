import asyncio
from typing import List, Optional, Tuple

import fsspec
from fsspec.implementations.local import LocalFileSystem


class Store:
    async def multi_get_async(
        self, keys: List[Tuple[str, Optional[Tuple[int, int]]]]
    ) -> List[Optional[bytes]]:
        return await asyncio.gather(
            *[self.get_async(key, byte_range) for key, byte_range in keys]
        )

    async def get_async(
        self, key: str, byte_range: Optional[Tuple[int, int]] = None
    ) -> Optional[bytes]:
        raise NotImplementedError

    async def multi_set_async(
        self, key_values: List[Tuple[str, bytes, Optional[Tuple[int, int]]]]
    ) -> None:
        await asyncio.gather(
            *[
                self.set_async(key, value, byte_range)
                for key, value, byte_range in key_values
            ]
        )

    async def set_async(
        self, key: str, value: bytes, byte_range: Optional[Tuple[int, int]] = None
    ) -> None:
        raise NotImplementedError

    async def delete_async(self, key: str) -> None:
        raise NotImplementedError


class AsyncLocalFileSystem:
    fs: LocalFileSystem

    async_impl = True

    def __init__(self, fs: LocalFileSystem):
        self.fs = fs

    async def cat_file(self, *args, **kwargs):
        return self.fs.cat_file(*args, **kwargs)

    async def makedirs(self, *args, **kwargs):
        return self.fs.makedirs(*args, **kwargs)

    async def exists(self, *args, **kwargs):
        return self.fs.exists(*args, **kwargs)

    async def rm(self, *args, **kwargs):
        return self.fs.rm(*args, **kwargs)

    async def _parent(self, *args, **kwargs):
        return self.fs._parent(*args, **kwargs)

    async def write_bytes(self, *args, **kwargs):
        return self.fs.write_bytes(*args, **kwargs)

    def open(self, *args, **kwargs):
        return self.fs.open(*args, **kwargs)


class FileSystemStore(Store):
    def __init__(self, url: str, **storage_options):
        assert isinstance(url, str)

        # instantiate file system
        fs, root = fsspec.core.url_to_fs(
            url, auto_mkdir=True, asynchronous=True, **storage_options
        )
        if isinstance(fs, LocalFileSystem):
            fs = AsyncLocalFileSystem(fs)
        self.fs = fs
        self.root = root.rstrip("/")

    async def get_async(
        self, key: str, byte_range: Optional[Tuple[int, int]] = None
    ) -> Optional[bytes]:
        assert isinstance(key, str)
        path = f"{self.root}/{key}"

        try:
            value = await (
                self.fs.cat_file(path, byte_range[0], byte_range[1])
                if byte_range
                else self.fs.cat_file(path)
            )
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return None

        return value

    async def set_async(
        self, key: str, value: bytes, byte_range: Optional[Tuple[int, int]] = None
    ) -> None:
        assert isinstance(key, str)
        path = f"{self.root}/{key}"

        # write data
        if byte_range:
            with self.fs.open(path, "r+b") as f:
                f.seek(byte_range[0])
                f.write(value)
        else:
            await self.fs.write_bytes(path, value)

    async def delete_async(self, key: str) -> None:
        path = f"{self.root}/{key}"
        if await self.fs.exists(path):
            await self.fs.rm(path)
