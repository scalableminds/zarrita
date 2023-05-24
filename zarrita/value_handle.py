from abc import abstractmethod
from typing import Optional, Tuple, Union

import numpy as np

from zarrita.store import Store


# ValueHandle abstracts over files, byte buffer, and arrays.
# Makes it easy pass around as references with lazy reading (esp. for files).
# Facilitates implementations of codec which either operate on arrays or bytes.
class ValueHandle:
    @abstractmethod
    async def set_async(
        self,
        selection: Union[slice, Tuple[slice, ...]],
        value: "ValueHandle",
    ):
        pass

    @abstractmethod
    async def get_async(
        self, selection: Union[slice, Tuple[slice, ...]]
    ) -> "ValueHandle":
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

    def __init__(self, store: "Store", path: str):
        super().__init__()
        self.store = store
        self.path = path

    async def set_async(
        self,
        selection: Union[slice, Tuple[slice, ...]],
        value: ValueHandle,
    ):
        assert isinstance(selection, slice)
        if selection.start is None and selection.stop is None:
            buf = await value.tobytes()
            if buf:
                await self.store.set_async(self.path, buf)
            else:
                await self.store.delete_async(self.path)
        else:
            buf = await value.tobytes()
            if buf:
                await self.store.set_async(self.path, buf, (selection.start, selection.stop))

    async def get_async(
        self, selection: Union[slice, Tuple[slice, ...]]
    ) -> ValueHandle:
        assert isinstance(selection, slice)
        buf = await self.store.get_async(self.path, (selection.start, selection.stop))
        if buf is not None:
            return BufferValueHandle(buf)
        return NoneValueHandle()

    async def tobytes(self) -> Optional[bytes]:
        return await self.store.get_async(self.path)

    async def toarray(
        self, shape: Optional[Tuple[int, ...]] = None
    ) -> Optional[np.ndarray]:
        buf = await self.tobytes()
        if buf is not None:
            return np.frombuffer(buf)
        return None


class BufferValueHandle(ValueHandle):
    buf: bytes

    def __init__(self, buf: bytes) -> None:
        super().__init__()
        self.buf = buf

    async def set_async(
        self,
        selection: Union[slice, Tuple[slice, ...]],
        value: ValueHandle,
    ):
        assert isinstance(selection, slice)
        buf = await value.tobytes()
        if buf:
            if selection.start is None and selection.stop is None:
                self.buf = buf
            else:
                tmp = bytearray(self.buf)
                tmp[selection] = buf
                self.buf = bytes(tmp)

    async def get_async(
        self, selection: Union[slice, Tuple[slice, ...]]
    ) -> ValueHandle:
        assert isinstance(selection, slice)
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
        selection: Union[slice, Tuple[slice, ...]],
        value: ValueHandle,
    ):
        self.array[selection] = await value.toarray()

    async def get_async(
        self, selection: Union[slice, Tuple[slice, ...]]
    ) -> ValueHandle:
        return ArrayValueHandle(self.array[selection])

    async def tobytes(self) -> Optional[bytes]:
        array = self.array
        if not array.flags.c_contiguous and not array.flags.f_contiguous:
            array = array.copy(order="K")
        return array.tobytes()

    async def toarray(
        self, shape: Optional[Tuple[int, ...]] = None
    ) -> Optional[np.ndarray]:
        return self.array


class NoneValueHandle(ValueHandle):
    async def set_async(
        self, selection: Union[slice, Tuple[slice, ...]], value: ValueHandle
    ):
        raise NotImplementedError

    async def get_async(
        self, selection: Union[slice, Tuple[slice, ...]]
    ) -> ValueHandle:
        return self

    async def tobytes(self) -> Optional[bytes]:
        return None

    async def toarray(
        self, shape: Optional[Tuple[int, ...]] = None
    ) -> Optional[np.ndarray]:
        return None
