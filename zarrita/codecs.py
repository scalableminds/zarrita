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
        [Any, bytes, "CoreArrayMetadata"],
        Awaitable[ValueHandle],
    ]
) -> Callable[[Any, ValueHandle, "CoreArrayMetadata"], Awaitable[ValueHandle],]:
    async def inner(
        _self,
        value: ValueHandle,
        array_metadata: "CoreArrayMetadata",
    ) -> ValueHandle:
        chunk_bytes = await value.tobytes()
        if chunk_bytes is None:
            return NoneValueHandle()
        return await f(_self, chunk_bytes, array_metadata)

    return inner


def _needs_array(
    f: Callable[
        [Any, np.ndarray, "CoreArrayMetadata"],
        Awaitable[ValueHandle],
    ]
) -> Callable[[Any, ValueHandle, "CoreArrayMetadata"], Awaitable[ValueHandle],]:
    async def inner(
        _self,
        value: ValueHandle,
        array_metadata: "CoreArrayMetadata",
    ) -> ValueHandle:
        chunk_array = await value.toarray()
        if chunk_array is None:
            return NoneValueHandle()
        chunk_array = chunk_array.view(dtype=array_metadata.dtype)
        return await f(_self, chunk_array, array_metadata)

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

    supports_partial_decode = False
    supports_partial_encode = False

    @_needs_bytes
    async def decode(
        self,
        chunk_bytes: bytes,
        _array_metadata: "CoreArrayMetadata",
    ) -> ValueHandle:
        return BufferValueHandle(
            Blosc.from_config(asdict(self.configuration)).decode(chunk_bytes)
        )

    @_needs_array
    async def encode(
        self,
        chunk_array: np.ndarray,
        _array_metadata: "CoreArrayMetadata",
    ) -> ValueHandle:
        if not chunk_array.flags.c_contiguous and not chunk_array.flags.f_contiguous:
            chunk_array = chunk_array.copy(order="K")
        return BufferValueHandle(
            Blosc.from_config(asdict(self.configuration)).encode(chunk_array)
        )


@frozen
class EndianCodecConfigurationMetadata:
    endian: Literal["big", "little"] = "little"


@frozen
class EndianCodecMetadata:
    configuration: EndianCodecConfigurationMetadata
    name: Literal["endian"] = "endian"

    supports_partial_decode = True
    supports_partial_encode = True

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
        chunk_array: np.ndarray,
        _array_metadata: "CoreArrayMetadata",
    ) -> ValueHandle:
        byteorder = self._get_byteorder(chunk_array)
        if self.configuration.endian != byteorder:
            chunk_array = chunk_array.view(
                dtype=chunk_array.dtype.newbyteorder(byteorder)
            )
        return ArrayValueHandle(chunk_array)

    @_needs_array
    async def encode(
        self,
        chunk_array: np.ndarray,
        _array_metadata: "CoreArrayMetadata",
    ) -> ValueHandle:
        byteorder = self._get_byteorder(chunk_array)
        if self.configuration.endian != byteorder:
            chunk_array = chunk_array.view(
                dtype=chunk_array.dtype.newbyteorder(byteorder)
            )
        return ArrayValueHandle(chunk_array)


@frozen
class TransposeCodecConfigurationMetadata:
    order: Union[Literal["C", "F"], Tuple[int, ...]] = "C"


@frozen
class TransposeCodecMetadata:
    configuration: TransposeCodecConfigurationMetadata
    name: Literal["transpose"] = "transpose"

    supports_partial_decode = False
    supports_partial_encode = False

    @_needs_array
    async def decode(
        self,
        chunk_array: np.ndarray,
        array_metadata: "CoreArrayMetadata",
    ) -> ValueHandle:
        new_order = self.configuration.order
        chunk_array = chunk_array.view(np.dtype(array_metadata.data_type.value))
        if isinstance(new_order, tuple):
            chunk_array = chunk_array.transpose(new_order)
        elif new_order == "F":
            chunk_array = chunk_array.reshape(
                array_metadata.chunk_shape,
                order="F",
            )
        else:
            chunk_array = chunk_array.reshape(
                array_metadata.chunk_shape,
                order="C",
            )
        # if array_metadata.order == "F":
        #     chunk_array = np.asfortranarray(chunk_array)
        # else:
        #     chunk_array = np.ascontiguousarray(chunk_array)
        return ArrayValueHandle(chunk_array)

    @_needs_array
    async def encode(
        self,
        chunk_array: np.ndarray,
        _array_metadata: "CoreArrayMetadata",
    ) -> ValueHandle:
        new_order = self.configuration.order
        if isinstance(new_order, tuple):
            chunk_array = chunk_array.transpose(new_order).reshape(-1, order="C")
        else:
            chunk_array = chunk_array.ravel(order=new_order)
        return ArrayValueHandle(chunk_array)


@frozen
class GzipCodecConfigurationMetadata:
    level: int = 5


@frozen
class GzipCodecMetadata:
    configuration: GzipCodecConfigurationMetadata
    name: Literal["gzip"] = "gzip"

    supports_partial_decode = False
    supports_partial_encode = False

    @_needs_bytes
    async def decode(
        self,
        chunk_bytes: bytes,
        _array_metadata: "CoreArrayMetadata",
    ) -> ValueHandle:
        return BufferValueHandle(GZip(self.configuration.level).decode(chunk_bytes))

    @_needs_bytes
    async def encode(
        self,
        chunk_bytes: bytes,
        _array_metadata: "CoreArrayMetadata",
    ) -> ValueHandle:
        return BufferValueHandle(GZip(self.configuration.level).encode(chunk_bytes))


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
