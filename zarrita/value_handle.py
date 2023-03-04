from abc import abstractmethod
from typing import Optional, Tuple, Union

import numpy as np

from zarrita.store import Store


# ValueHandle abstracts over files, byte buffer, and arrays.
# Makes it easy pass around as references with lazy reading (esp. for files).
# Facilitates implementations of codec which either operate on arrays or bytes.
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
    array: np.ndarray

    def __init__(self, array: np.ndarray) -> None:
        super().__init__()
        self.array = array

    def __setitem__(
        self,
        selection: Union[slice, Tuple[slice, ...]],
        value: ValueHandle,
    ):
        self.array[selection] = value.toarray()

    def __getitem__(self, selection: Union[slice, Tuple[slice, ...]]) -> ValueHandle:
        return ArrayHandle(self.array[selection])

    def tobytes(self) -> Optional[bytes]:
        array = self.array
        if not array.flags.c_contiguous and not array.flags.f_contiguous:
            array = array.copy(order="K")
        return array.tobytes()

    def toarray(self, shape: Optional[Tuple[int, ...]] = None) -> Optional[np.ndarray]:
        return self.array


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
