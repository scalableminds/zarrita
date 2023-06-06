from abc import abstractmethod
from typing import Optional, Tuple

import numpy as np

from zarrita.common import BytesLike
from zarrita.store import Store


# ValueHandle abstracts over files, byte buffer, and arrays.
# Makes it easy pass around as references with lazy reading (esp. for files).
# Facilitates implementations of codec which either operate on arrays or bytes.
class ValueHandle:
    @abstractmethod
    async def set_async(
        self,
        value: "ValueHandle",
    ):
        pass

    @abstractmethod
    def __getitem__(self, selection: slice) -> "ValueHandle":
        pass

    @abstractmethod
    async def tobytes(self) -> Optional[bytes]:
        pass

    @abstractmethod
    async def toarray(
        self, shape: Optional[Tuple[int, ...]] = None
    ) -> Optional[np.ndarray]:
        pass


class FileValueHandle(ValueHandle):
    store: "Store"
    path: str
    selection: Optional[slice] = None

    def __init__(self, store: "Store", path: str):
        super().__init__()
        self.store = store
        self.path = path

    def __getitem__(self, selection: slice) -> ValueHandle:
        out = FileValueHandle(self.store, self.path)
        out.selection = selection
        return out

    async def set_async(
        self,
        value: ValueHandle,
    ):
        buf = await value.tobytes()
        if buf:
            await self.store.set_async(self.path, buf)
        else:
            await self.store.delete_async(self.path)

    async def tobytes(self) -> Optional[bytes]:
        if self.selection:
            return await self.store.get_async(
                self.path, (self.selection.start, self.selection.stop)
            )
        return await self.store.get_async(self.path)

    async def toarray(
        self, shape: Optional[Tuple[int, ...]] = None
    ) -> Optional[np.ndarray]:
        buf = await self.tobytes()
        if buf is not None:
            return np.frombuffer(buf)
        return None


class BufferValueHandle(ValueHandle):
    buf: BytesLike

    def __init__(self, buf: BytesLike) -> None:
        self.buf = buf

    async def set_async(
        self,
        value: ValueHandle,
    ):
        buf = await value.tobytes()
        assert buf is not None
        self.buf = buf

    def __getitem__(self, selection: slice) -> "ValueHandle":
        return BufferValueHandle(self.buf[selection])

    async def tobytes(self) -> Optional[bytes]:
        return self.buf

    async def toarray(
        self, shape: Optional[Tuple[int, ...]] = None
    ) -> Optional[np.ndarray]:
        return np.frombuffer(self.buf)


class ArrayValueHandle(ValueHandle):
    array: np.ndarray

    def __init__(self, array: np.ndarray) -> None:
        super().__init__()
        self.array = array

    async def set_async(
        self,
        value: ValueHandle,
    ):
        array = await value.toarray()
        assert array is not None
        self.array = array

    def __getitem__(self, selection: slice) -> "ValueHandle":
        return ArrayValueHandle(self.array[selection])

    async def tobytes(self) -> Optional[bytes]:
        array = self.array
        if not array.flags.c_contiguous and not array.flags.f_contiguous:
            array = array.copy(order="A")
        return array.tobytes()

    async def toarray(
        self, shape: Optional[Tuple[int, ...]] = None
    ) -> Optional[np.ndarray]:
        return self.array


class NoneValueHandle(ValueHandle):
    async def set_async(self, value: ValueHandle):
        raise NotImplementedError

    def __getitem__(self, _selection: slice) -> "ValueHandle":
        return self

    async def tobytes(self) -> Optional[bytes]:
        return None

    async def toarray(
        self, shape: Optional[Tuple[int, ...]] = None
    ) -> Optional[np.ndarray]:
        return None
