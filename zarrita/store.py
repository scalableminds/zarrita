from abc import abstractmethod
from typing import List, Optional, Tuple, Union

import fsspec
import numpy as np
from attrs import define


class ValueHandle:
    @abstractmethod
    def __setitem__(
        self,
        selection: Union[slice, Tuple[slice, ...]],
        value: "ValueHandle",
    ):
        pass

    @abstractmethod
    def __getitem__(self, selection: Union[slice, Tuple[slice, ...]]) -> "ValueHandle":
        pass

    @abstractmethod
    def tobytes(self) -> Optional[bytes]:
        pass

    @abstractmethod
    def toarray(self, shape: Optional[Tuple[int, ...]] = None) -> Optional[np.ndarray]:
        pass


class FileHandle(ValueHandle):
    store: "Store"
    path: str

    def __init__(self, store: "Store", path: str):
        super().__init__()
        self.store = store
        self.path = path

    def __setitem__(
        self,
        selection: Union[slice, Tuple[slice, ...]],
        value: ValueHandle,
    ):
        assert isinstance(selection, slice)
        if selection.start is None and selection.stop is None:
            buf = value.tobytes()
            if buf:
                self.store.set(self.path, buf)
            else:
                self.store.delete(self.path)
        else:
            buf = value.tobytes()
            if buf:
                self.store.set(self.path, buf, (selection.start, selection.stop))

    def __getitem__(self, selection: Union[slice, Tuple[slice, ...]]) -> ValueHandle:
        assert isinstance(selection, slice)
        buf = self.store.get(self.path, (selection.start, selection.stop))
        if buf is not None:
            return BufferHandle(buf)
        return NoneHandle()

    def tobytes(self) -> Optional[bytes]:
        return self.store.get(self.path)

    def toarray(self, shape: Optional[Tuple[int, ...]] = None) -> Optional[np.ndarray]:
        buf = self.tobytes()
        if buf is not None:
            return np.frombuffer(buf)
        return None


class BufferHandle(ValueHandle):
    buf: bytes

    def __init__(self, buf: bytes) -> None:
        super().__init__()
        self.buf = buf

    def __setitem__(
        self,
        selection: Union[slice, Tuple[slice, ...]],
        value: ValueHandle,
    ):
        assert isinstance(selection, slice)
        buf = value.tobytes()
        if buf:
            if selection.start is None and selection.stop is None:
                self.buf = buf
            else:
                tmp = bytearray(self.buf)
                tmp[selection] = buf
                self.buf = bytes(tmp)

    def __getitem__(self, selection: Union[slice, Tuple[slice, ...]]) -> ValueHandle:
        assert isinstance(selection, slice)
        return BufferHandle(self.buf[selection])

    def tobytes(self) -> Optional[bytes]:
        return self.buf

    def toarray(self, shape: Optional[Tuple[int, ...]] = None) -> Optional[np.ndarray]:
        return np.frombuffer(self.buf)


class ArrayHandle(ValueHandle):
    buf: np.ndarray

    def __init__(self, buf: np.ndarray) -> None:
        super().__init__()
        self.buf = buf

    def __setitem__(
        self,
        selection: Union[slice, Tuple[slice, ...]],
        value: ValueHandle,
    ):
        self.buf[selection] = value.toarray()

    def __getitem__(self, selection: Union[slice, Tuple[slice, ...]]) -> ValueHandle:
        return ArrayHandle(self.buf[selection])

    def tobytes(self) -> Optional[bytes]:
        return self.buf.tobytes()

    def toarray(self, shape: Optional[Tuple[int, ...]] = None) -> Optional[np.ndarray]:
        return self.buf


class NoneHandle(ValueHandle):
    def __setitem__(
        self, selection: Union[slice, Tuple[slice, ...]], value: ValueHandle
    ):
        raise NotImplemented

    def __getitem__(self, selection: Union[slice, Tuple[slice, ...]]) -> ValueHandle:
        return self

    def tobytes(self) -> Optional[bytes]:
        return None

    def toarray(self, shape: Optional[Tuple[int, ...]] = None) -> Optional[np.ndarray]:
        return None


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
    # TODO ultimately replace this with the fsspec FSMap class, but for now roll
    # our own implementation in order to be able to add some extra methods for
    # listing keys.

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
        # noinspection PyProtectedMember
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
        self.fs.rm(path)
