from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Awaitable, Callable, List, Literal, Tuple, Union

import numpy as np
from attr import asdict, frozen
from numcodecs.blosc import Blosc
from numcodecs.gzip import GZip

from zarrita.common import BytesLike, to_thread
from zarrita.metadata import (
    BloscCodecConfigurationMetadata,
    BloscCodecMetadata,
    CodecMetadata,
    EndianCodecConfigurationMetadata,
    EndianCodecMetadata,
    GzipCodecConfigurationMetadata,
    GzipCodecMetadata,
    ShardingCodecConfigurationMetadata,
    ShardingCodecMetadata,
    TransposeCodecConfigurationMetadata,
    TransposeCodecMetadata,
    data_type_to_numpy,
)
from zarrita.value_handle import (
    ArrayValueHandle,
    BufferValueHandle,
    NoneValueHandle,
    ValueHandle,
)

if TYPE_CHECKING:
    from zarrita.array import CoreArrayMetadata


async def _needs_array(
    chunk_value_handle: ValueHandle,
    array_metadata: "CoreArrayMetadata",
    f: Callable[
        [np.ndarray, "CoreArrayMetadata"], Awaitable[Union[None, np.ndarray, BytesLike]]
    ],
):
    chunk_array = await chunk_value_handle.toarray()
    if chunk_array is None:
        return NoneValueHandle()
    if chunk_array.dtype.name != array_metadata.dtype.name:
        chunk_array = chunk_array.view(dtype=array_metadata.dtype)
    result = await f(chunk_array, array_metadata)
    if result is None:
        return NoneValueHandle()
    elif (
        isinstance(result, bytes)
        or isinstance(result, bytearray)
        or isinstance(result, memoryview)
    ):
        return BufferValueHandle(result)
    elif isinstance(result, np.ndarray):
        return ArrayValueHandle(result)


async def _needs_bytes(
    chunk_value_handle: ValueHandle,
    array_metadata: "CoreArrayMetadata",
    f: Callable[
        [BytesLike, "CoreArrayMetadata"], Awaitable[Union[None, np.ndarray, BytesLike]]
    ],
):
    chunk_bytes = await chunk_value_handle.tobytes()
    if chunk_bytes is None:
        return NoneValueHandle()
    result = await f(chunk_bytes, array_metadata)
    if result is None:
        return NoneValueHandle()
    elif (
        isinstance(result, bytes)
        or isinstance(result, bytearray)
        or isinstance(result, memoryview)
    ):
        return BufferValueHandle(result)
    elif isinstance(result, np.ndarray):
        return ArrayValueHandle(result)


class Codec(ABC):
    supports_partial_decode: bool
    supports_partial_encode: bool
    input_type: Literal["bytes", "array"]
    output_type: Literal["bytes", "array"]

    @abstractmethod
    async def decode(
        self,
        chunk_value_handle: ValueHandle,
        array_metadata: "CoreArrayMetadata",
    ) -> ValueHandle:
        pass

    @abstractmethod
    async def encode(
        self,
        chunk_value_handle: ValueHandle,
        array_metadata: "CoreArrayMetadata",
    ) -> ValueHandle:
        pass

    @staticmethod
    def codecs_from_metadata(codecs_metadata: List["CodecMetadata"]) -> List["Codec"]:
        out: List[Codec] = []
        for codec_metadata in codecs_metadata:
            if codec_metadata.name == "blosc":
                out.append(BloscCodec.from_metadata(codec_metadata))
            elif codec_metadata.name == "gzip":
                out.append(GzipCodec.from_metadata(codec_metadata))
            elif codec_metadata.name == "transpose":
                out.append(TransposeCodec.from_metadata(codec_metadata))
            elif codec_metadata.name == "endian":
                out.append(EndianCodec.from_metadata(codec_metadata))
            elif codec_metadata.name == "sharding_indexed":
                from zarrita.sharding import ShardingCodec

                out.append(ShardingCodec.from_metadata(codec_metadata))
            else:
                raise RuntimeError(f"Unsupported codec: {codec_metadata}")
        return out


class ArrayArrayCodec(Codec):
    input_type = "array"
    output_type = "array"

    async def decode(
        self,
        chunk_value_handle: ValueHandle,
        array_metadata: "CoreArrayMetadata",
    ) -> ValueHandle:
        return await _needs_array(chunk_value_handle, array_metadata, self.inner_decode)

    @abstractmethod
    async def inner_decode(
        self,
        chunk_array: np.ndarray,
        array_metadata: "CoreArrayMetadata",
    ) -> np.ndarray:
        pass

    async def encode(
        self,
        chunk_value_handle: ValueHandle,
        array_metadata: "CoreArrayMetadata",
    ) -> ValueHandle:
        return await _needs_array(chunk_value_handle, array_metadata, self.inner_encode)

    @abstractmethod
    async def inner_encode(
        self,
        chunk_array: np.ndarray,
        array_metadata: "CoreArrayMetadata",
    ) -> np.ndarray:
        pass


class ArrayBytesCodec(Codec):
    input_type = "array"
    output_type = "bytes"

    async def decode(
        self,
        chunk_value_handle: ValueHandle,
        array_metadata: "CoreArrayMetadata",
    ) -> ValueHandle:
        return await _needs_bytes(chunk_value_handle, array_metadata, self.inner_decode)

    @abstractmethod
    async def inner_decode(
        self,
        chunk_array: BytesLike,
        array_metadata: "CoreArrayMetadata",
    ) -> np.ndarray:
        pass

    async def encode(
        self,
        chunk_value_handle: ValueHandle,
        array_metadata: "CoreArrayMetadata",
    ) -> ValueHandle:
        return await _needs_array(chunk_value_handle, array_metadata, self.inner_encode)

    @abstractmethod
    async def inner_encode(
        self,
        chunk_array: np.ndarray,
        array_metadata: "CoreArrayMetadata",
    ) -> BytesLike:
        pass


class BytesBytesCodec(Codec):
    input_type = "bytes"
    output_type = "bytes"

    async def decode(
        self,
        chunk_value_handle: ValueHandle,
        array_metadata: "CoreArrayMetadata",
    ) -> ValueHandle:
        return await _needs_bytes(chunk_value_handle, array_metadata, self.inner_decode)

    @abstractmethod
    async def inner_decode(
        self,
        chunk_array: BytesLike,
        array_metadata: "CoreArrayMetadata",
    ) -> BytesLike:
        pass

    async def encode(
        self,
        chunk_value_handle: ValueHandle,
        array_metadata: "CoreArrayMetadata",
    ) -> ValueHandle:
        return await _needs_bytes(chunk_value_handle, array_metadata, self.inner_encode)

    @abstractmethod
    async def inner_encode(
        self,
        chunk_array: BytesLike,
        array_metadata: "CoreArrayMetadata",
    ) -> BytesLike:
        pass


@frozen
class BloscCodec(BytesBytesCodec):
    configuration: BloscCodecConfigurationMetadata

    @classmethod
    def from_metadata(cls, codec_metadata: BloscCodecMetadata) -> "BloscCodec":
        return cls(
            configuration=codec_metadata.configuration,
        )

    def _get_blosc_codec(self):
        config_dict = asdict(self.configuration)
        map_shuffle_str_to_int = {"noshuffle": 0, "shuffle": 1, "bitshuffle": 2}
        config_dict["shuffle"] = map_shuffle_str_to_int[config_dict["shuffle"]]
        return Blosc.from_config(config_dict)

    async def inner_decode(
        self,
        chunk_bytes: bytes,
        _array_metadata: "CoreArrayMetadata",
    ) -> BytesLike:
        return await to_thread(self._get_blosc_codec().decode, chunk_bytes)

    async def inner_encode(
        self,
        chunk_bytes: bytes,
        array_metadata: "CoreArrayMetadata",
    ) -> BytesLike:
        chunk_array = np.frombuffer(chunk_bytes, dtype=array_metadata.dtype)
        return await to_thread(self._get_blosc_codec().encode, chunk_array)


@frozen
class EndianCodec(ArrayBytesCodec):
    configuration: EndianCodecConfigurationMetadata

    @classmethod
    def from_metadata(cls, codec_metadata: EndianCodecMetadata) -> "EndianCodec":
        return cls(
            configuration=codec_metadata.configuration,
        )

    def _get_byteorder(self, array: np.ndarray) -> Literal["big", "little"]:
        if array.dtype.byteorder == "<":
            return "little"
        elif array.dtype.byteorder == ">":
            return "big"
        else:
            import sys

            return sys.byteorder

    async def inner_decode(
        self,
        chunk_bytes: BytesLike,
        array_metadata: "CoreArrayMetadata",
    ) -> np.ndarray:
        if self.configuration.endian == "little":
            prefix = "<"
        else:
            prefix = ">"
        dtype = np.dtype(f"{prefix}{data_type_to_numpy[array_metadata.data_type]}")
        chunk_array = np.frombuffer(chunk_bytes, dtype)
        return chunk_array

    async def inner_encode(
        self,
        chunk_array: np.ndarray,
        _array_metadata: "CoreArrayMetadata",
    ) -> BytesLike:
        byteorder = self._get_byteorder(chunk_array)
        if self.configuration.endian != byteorder:
            new_dtype = chunk_array.dtype.newbyteorder(self.configuration.endian)
            chunk_array = chunk_array.astype(new_dtype)
        return chunk_array.tobytes()


@frozen
class TransposeCodec(ArrayArrayCodec):
    configuration: TransposeCodecConfigurationMetadata

    @classmethod
    def from_metadata(cls, codec_metadata: TransposeCodecMetadata) -> "TransposeCodec":
        return cls(
            configuration=codec_metadata.configuration,
        )

    async def inner_decode(
        self,
        chunk_array: np.ndarray,
        array_metadata: "CoreArrayMetadata",
    ) -> np.ndarray:
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
        return chunk_array

    async def inner_encode(
        self,
        chunk_array: np.ndarray,
        _array_metadata: "CoreArrayMetadata",
    ) -> np.ndarray:
        new_order = self.configuration.order
        if isinstance(new_order, tuple):
            chunk_array = chunk_array.transpose(new_order).reshape(-1, order="C")
        else:
            chunk_array = chunk_array.ravel(order=new_order)
        return chunk_array


@frozen
class GzipCodec(BytesBytesCodec):
    configuration: GzipCodecConfigurationMetadata

    @classmethod
    def from_metadata(cls, codec_metadata: GzipCodecMetadata) -> "GzipCodec":
        return cls(
            configuration=codec_metadata.configuration,
        )

    async def inner_decode(
        self,
        chunk_bytes: bytes,
        _array_metadata: "CoreArrayMetadata",
    ) -> BytesLike:
        return await to_thread(GZip(self.configuration.level).decode, chunk_bytes)

    async def inner_encode(
        self,
        chunk_bytes: bytes,
        _array_metadata: "CoreArrayMetadata",
    ) -> BytesLike:
        return await to_thread(GZip(self.configuration.level).encode, chunk_bytes)


def blosc_codec(
    cname: Literal["lz4", "lz4hc", "blosclz", "zstd", "snappy", "zlib"] = "zstd",
    clevel: int = 5,
    shuffle: Literal["noshuffle", "shuffle", "bitshuffle"] = "noshuffle",
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
