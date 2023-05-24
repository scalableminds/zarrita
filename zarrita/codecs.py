from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    List,
    Literal,
    Tuple,
    Union,
)

import numpy as np
from attr import asdict, frozen
from numcodecs.blosc import Blosc
from numcodecs.gzip import GZip

from zarrita.sharding import ShardingCodecConfigurationMetadata, ShardingCodecMetadata
from zarrita.value_handle import (
    ArrayValueHandle,
    BufferValueHandle,
    NoneValueHandle,
    ValueHandle,
)

if TYPE_CHECKING:
    from zarrita.array import CoreArrayMetadata


def _needs_bytes(
    f: Callable[
        [Any, bytes, Tuple[slice, ...], "CoreArrayMetadata"],
        Awaitable[ValueHandle],
    ]
) -> Callable[
    [Any, ValueHandle, Tuple[slice, ...], "CoreArrayMetadata"],
    Awaitable[ValueHandle],
]:
    async def inner(
        _self,
        value: ValueHandle,
        selection: Tuple[slice, ...],
        array_metadata: "CoreArrayMetadata",
    ) -> ValueHandle:
        buf = await value.tobytes()
        if buf is None:
            return NoneValueHandle()
        return await f(_self, buf, selection, array_metadata)

    return inner


def _needs_array(
    f: Callable[
        [Any, np.ndarray, Tuple[slice, ...], "CoreArrayMetadata"],
        Awaitable[ValueHandle],
    ]
) -> Callable[
    [Any, ValueHandle, Tuple[slice, ...], "CoreArrayMetadata"],
    Awaitable[ValueHandle],
]:
    async def inner(
        _self,
        value: ValueHandle,
        selection: Tuple[slice, ...],
        array_metadata: "CoreArrayMetadata",
    ) -> ValueHandle:
        array = await value.toarray()
        if array is None:
            return NoneValueHandle()
        return await f(_self, array, selection, array_metadata)

    return inner


@frozen
class BloscCodecConfigurationMetadata:
    cname: Literal["lz4", "lz4hc", "blosclz", "zstd", "snappy", "zlib"] = "zstd"
    clevel: int = 5
    shuffle: Literal[0, 1, 2, -1] = 0
    blocksize: int = 0


@frozen
class BloscCodecMetadata:
    configuration: BloscCodecConfigurationMetadata
    name: Literal["blosc"] = "blosc"

    @_needs_bytes
    async def decode(
        self,
        buf: bytes,
        _selection: Tuple[slice, ...],
        _array_metadata: "CoreArrayMetadata",
    ) -> ValueHandle:
        return BufferValueHandle(
            Blosc.from_config(asdict(self.configuration)).decode(buf)
        )

    @_needs_array
    async def encode(
        self,
        chunk: np.ndarray,
        _selection: Tuple[slice, ...],
        _array_metadata: "CoreArrayMetadata",
    ) -> ValueHandle:
        if not chunk.flags.c_contiguous and not chunk.flags.f_contiguous:
            chunk = chunk.copy(order="K")
        return BufferValueHandle(
            Blosc.from_config(asdict(self.configuration)).encode(chunk)
        )


@frozen
class EndianCodecConfigurationMetadata:
    endian: Literal["big", "little"] = "little"


@frozen
class EndianCodecMetadata:
    configuration: EndianCodecConfigurationMetadata
    name: Literal["endian"] = "endian"

    def _get_byteorder(self, array: np.ndarray) -> Literal["big", "little"]:
        if array.dtype.byteorder == "<":
            return "little"
        elif array.dtype.byteorder == ">":
            return "big"
        else:
            import sys

            return sys.byteorder

    @_needs_array
    async def decode(
        self,
        chunk: np.ndarray,
        _selection: Tuple[slice, ...],
        _array_metadata: "CoreArrayMetadata",
    ) -> ValueHandle:
        byteorder = self._get_byteorder(chunk)
        if self.configuration.endian != byteorder:
            chunk = chunk.view(dtype=chunk.dtype.newbyteorder(byteorder))
        return ArrayValueHandle(chunk)

    @_needs_array
    async def encode(
        self,
        chunk: np.ndarray,
        _selection: Tuple[slice, ...],
        _array_metadata: "CoreArrayMetadata",
    ) -> ValueHandle:
        byteorder = self._get_byteorder(chunk)
        if self.configuration.endian != byteorder:
            chunk = chunk.view(dtype=chunk.dtype.newbyteorder(byteorder))
        return ArrayValueHandle(chunk)


@frozen
class TransposeCodecConfigurationMetadata:
    order: Union[Literal["C", "F"], Tuple[int, ...]] = "C"


@frozen
class TransposeCodecMetadata:
    configuration: TransposeCodecConfigurationMetadata
    name: Literal["transpose"] = "transpose"

    @_needs_array
    async def decode(
        self,
        chunk: np.ndarray,
        _selection: Tuple[slice, ...],
        array_metadata: "CoreArrayMetadata",
    ) -> ValueHandle:
        new_order = self.configuration.order
        chunk = chunk.view(np.dtype(array_metadata.data_type.value))
        if isinstance(new_order, tuple):
            chunk = chunk.transpose(new_order)
        else:
            chunk = chunk.reshape(
                array_metadata.chunk_shape,
                order=new_order,
            )
        return ArrayValueHandle(chunk)

    @_needs_array
    async def encode(
        self,
        chunk: np.ndarray,
        _selection: Tuple[slice, ...],
        _array_metadata: "CoreArrayMetadata",
    ) -> ValueHandle:
        new_order = self.configuration.order
        if isinstance(new_order, tuple):
            chunk = chunk.reshape(-1, order="C")
        else:
            chunk = chunk.reshape(-1, order=new_order)
        return ArrayValueHandle(chunk)


@frozen
class GzipCodecConfigurationMetadata:
    level: int = 5


@frozen
class GzipCodecMetadata:
    configuration: GzipCodecConfigurationMetadata
    name: Literal["gzip"] = "gzip"

    @_needs_bytes
    async def decode(
        self,
        buf: bytes,
        _selection: Tuple[slice, ...],
        _array_metadata: "CoreArrayMetadata",
    ) -> ValueHandle:
        return BufferValueHandle(GZip(self.configuration.level).decode(buf))

    @_needs_bytes
    async def encode(
        self,
        buf: bytes,
        _selection: Tuple[slice, ...],
        _array_metadata: "CoreArrayMetadata",
    ) -> ValueHandle:
        return BufferValueHandle(GZip(self.configuration.level).encode(buf))


CodecMetadata = Union[
    BloscCodecMetadata,
    EndianCodecMetadata,
    TransposeCodecMetadata,
    GzipCodecMetadata,
    ShardingCodecMetadata,
]


def blosc_codec(
    cname: Literal["lz4", "lz4hc", "blosclz", "zstd", "snappy", "zlib"] = "zstd",
    clevel: int = 5,
    shuffle: Literal[0, 1, 2, -1] = 0,
    blocksize: int = 0,
) -> BloscCodecMetadata:
    return BloscCodecMetadata(
        configuration=BloscCodecConfigurationMetadata(
            cname=cname, clevel=clevel, shuffle=shuffle, blocksize=blocksize
        )
    )


def endian_codec(endian: Literal["big", "little"]) -> EndianCodecMetadata:
    return EndianCodecMetadata(configuration=EndianCodecConfigurationMetadata(endian))


def transpose_codec(
    order: Union[Tuple[int, ...], Literal["C", "F"]]
) -> TransposeCodecMetadata:
    return TransposeCodecMetadata(
        configuration=TransposeCodecConfigurationMetadata(order)
    )


def gzip_codec(level: int = 5) -> GzipCodecMetadata:
    return GzipCodecMetadata(configuration=GzipCodecConfigurationMetadata(level))


def sharding_codec(
    chunk_shape: Tuple[int, ...], codecs: List[CodecMetadata] = []
) -> ShardingCodecMetadata:
    return ShardingCodecMetadata(
        configuration=ShardingCodecConfigurationMetadata(chunk_shape, codecs)
    )
