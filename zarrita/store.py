import asyncio
import io
from pathlib import Path
from typing import List, Optional, Tuple, Union

import fsspec

from zarrita.common import BytesLike, to_thread


class Store:
    async def multi_get_async(
        self, keys: List[Tuple[str, Optional[Tuple[int, int]]]]
    ) -> List[Optional[BytesLike]]:
        return await asyncio.gather(
            *[self.get_async(key, byte_range) for key, byte_range in keys]
        )

    async def get_async(
        self, key: str, byte_range: Optional[Tuple[int, int]] = None
    ) -> Optional[BytesLike]:
        raise NotImplementedError

    async def multi_set_async(
        self, key_values: List[Tuple[str, BytesLike, Optional[Tuple[int, int]]]]
    ) -> None:
        await asyncio.gather(
            *[
                self.set_async(key, value, byte_range)
                for key, value, byte_range in key_values
            ]
        )

    async def set_async(
        self, key: str, value: BytesLike, byte_range: Optional[Tuple[int, int]] = None
    ) -> None:
        raise NotImplementedError

    async def delete_async(self, key: str) -> None:
        raise NotImplementedError

    async def exists_async(self, key: str) -> bool:
        raise NotImplementedError


class LocalStore(Store):
    root: Path
    auto_mkdir: bool

    def __init__(self, root: Union[Path, str], auto_mkdir: bool = True):
        if isinstance(root, str):
            root = Path(root)
        assert isinstance(root, Path)

        self.root = root
        self.auto_mkdir = auto_mkdir

    def _cat_file(
        self, path: Path, start: Optional[int] = None, end: Optional[int] = None
    ) -> BytesLike:
        if start is None and end is None:
            return path.read_bytes()
        with path.open("rb") as f:
            size = f.seek(0, io.SEEK_END)
            if start is not None:
                if start >= 0:
                    f.seek(start)
                else:
                    f.seek(max(0, size + start))
            if end is not None:
                if end < 0:
                    end = size + end
                return f.read(end - f.tell())
            return f.read()

    def _put_file(
        self,
        path: Path,
        value: BytesLike,
        start: Optional[int] = None,
    ):
        if self.auto_mkdir:
            path.parent.mkdir(parents=True, exist_ok=True)
        if start is not None:
            with path.open("r+b") as f:
                f.seek(start)
                f.write(value)
        else:
            return path.write_bytes(value)

    async def get_async(
        self, key: str, byte_range: Optional[Tuple[int, int]] = None
    ) -> Optional[BytesLike]:
        assert isinstance(key, str)
        path = self.root / key

        try:
            value = await (
                to_thread(self._cat_file, path, byte_range[0], byte_range[1])
                if byte_range is not None
                else to_thread(self._cat_file, path)
            )
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return None

        return value

    async def set_async(
        self, key: str, value: BytesLike, byte_range: Optional[Tuple[int, int]] = None
    ) -> None:
        assert isinstance(key, str)
        path = self.root / key

        if byte_range is not None:
            await to_thread(self._put_file, path, value, byte_range[0])
        else:
            await to_thread(self._put_file, path, value)

    async def delete_async(self, key: str) -> None:
        path = self.root / key
        await to_thread(path.unlink, True)

    async def exists_async(self, key: str) -> bool:
        path = self.root / key
        return await to_thread(path.exists)


class RemoteStore(Store):
    def __init__(self, url: str, **storage_options):
        assert isinstance(url, str)

        # instantiate file system
        fs, root = fsspec.core.url_to_fs(
            url, auto_mkdir=True, asynchronous=True, **storage_options
        )
        assert fs.__class__.async_impl, "FileSystem needs to support async operations."
        self.fs = fs
        self.root = root.rstrip("/")

    async def get_async(
        self, key: str, byte_range: Optional[Tuple[int, int]] = None
    ) -> Optional[BytesLike]:
        assert isinstance(key, str)
        path = f"{self.root}/{key}"

        try:
            value = await (
                self.fs._cat_file(path, byte_range[0], byte_range[1])
                if byte_range
                else self.fs._cat_file(path)
            )
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return None

        return value

    async def set_async(
        self, key: str, value: BytesLike, byte_range: Optional[Tuple[int, int]] = None
    ) -> None:
        assert isinstance(key, str)
        path = f"{self.root}/{key}"

        # write data
        if byte_range:
            with self.fs._open(path, "r+b") as f:
                f.seek(byte_range[0])
                f.write(value)
        else:
            await self.fs._write_bytes(path, value)

    async def delete_async(self, key: str) -> None:
        path = f"{self.root}/{key}"
        if await self.fs._exists(path):
            await self.fs._rm(path)

    async def exists_async(self, key: str) -> bool:
        path = f"{self.root}/{key}"
        return await self.fs._exists(path)
