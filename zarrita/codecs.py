from typing import TYPE_CHECKING, Final, List, Literal, Tuple, Union
from attr import define
from attrs import asdict
from numcodecs.blosc import Blosc
from numcodecs.gzip import GZip
import numpy as np
from zarrita.sharding import ShardingCodecConfigurationMetadata, ShardingCodecMetadata

if TYPE_CHECKING:
    from zarrita.array import CoreArrayMetadata

from zarrita.store import ArrayHandle, BufferHandle, NoneHandle, ValueHandle


@define
class BloscCodecConfigurationMetadata:
    cname: Literal["lz4", "lz4hc", "blosclz", "zstd", "snappy", "zlib"] = "zstd"
    clevel: int = 5
    shuffle: Literal[0, 1, 2, -1] = 0
    blocksize: int = 0


@define
class BloscCodecMetadata:
    configuration: BloscCodecConfigurationMetadata
    name: Final = "blosc"

    def decode(
        self,
        value: ValueHandle,
        _selection: Tuple[slice, ...],
        _array_metadata: "CoreArrayMetadata",
    ) -> ValueHandle:
        buf = value.tobytes()
        if buf is None:
            return NoneHandle()
        return BufferHandle(Blosc.from_config(asdict(self.configuration)).decode(buf))

    def encode(
        self,
        value: ValueHandle,
        _selection: Tuple[slice, ...],
        _array_metadata: "CoreArrayMetadata",
    ) -> ValueHandle:
        chunk = value.toarray()
        if chunk is None:
            return NoneHandle()
        if not chunk.flags.c_contiguous and not chunk.flags.f_contiguous:
            chunk = chunk.copy(order="K")
        return BufferHandle(Blosc.from_config(asdict(self.configuration)).encode(chunk))


@define
class EndianCodecConfigurationMetadata:
    endian: Literal["big", "little"] = "little"


@define
class EndianCodecMetadata:
    configuration: EndianCodecConfigurationMetadata
    name: Final = "endian"

    def _get_byteorder(self, array: np.ndarray) -> Literal["big", "little"]:
        if array.dtype.byteorder == "<":
            return "little"
        elif array.dtype.byteorder == ">":
            return "big"
        else:
            import sys

            return sys.byteorder

    def decode(
        self,
        value: ValueHandle,
        _selection: Tuple[slice, ...],
        _array_metadata: "CoreArrayMetadata",
    ) -> ValueHandle:
        chunk = value.toarray()
        if chunk is None:
            return NoneHandle()
        byteorder = self._get_byteorder(chunk)
        if self.configuration.endian != byteorder:
            chunk = chunk.view(dtype=chunk.dtype.newbyteorder(byteorder))
        return ArrayHandle(chunk)

    def encode(
        self,
        value: ValueHandle,
        _selection: Tuple[slice, ...],
        _array_metadata: "CoreArrayMetadata",
    ) -> ValueHandle:
        chunk = value.toarray()
        if chunk is None:
            return NoneHandle()
        byteorder = self._get_byteorder(chunk)
        if self.configuration.endian != byteorder:
            chunk = chunk.view(dtype=chunk.dtype.newbyteorder(byteorder))
        return ArrayHandle(chunk)


@define
class TransposeCodecConfigurationMetadata:
    order: Union[Literal["C", "F"], Tuple[int, ...]] = "C"


@define
class TransposeCodecMetadata:
    configuration: TransposeCodecConfigurationMetadata
    name: Final = "transpose"

    def decode(
        self,
        value: ValueHandle,
        _selection: Tuple[slice, ...],
        array_metadata: "CoreArrayMetadata",
    ) -> ValueHandle:
        chunk = value.toarray()
        if chunk is None:
            return NoneHandle()
        new_order = self.configuration.order
        order = "C" if chunk.flags["C_CONTIGUOUS"] else "F"
        if new_order != order:
            chunk = chunk.view(np.dtype(array_metadata.data_type.value))
            if isinstance(new_order, tuple):
                chunk = chunk.transpose(new_order)
            else:
                chunk = chunk.reshape(
                    array_metadata.chunk_shape,
                    order=new_order,
                )
        return ArrayHandle(chunk)

    def encode(
        self,
        value: ValueHandle,
        _selection: Tuple[slice, ...],
        _array_metadata: "CoreArrayMetadata",
    ) -> ValueHandle:
        chunk = value.toarray()
        if chunk is None:
            return NoneHandle()
        new_order = self.configuration.order
        if isinstance(new_order, tuple):
            chunk = chunk.reshape(-1, order="C")
        else:
            chunk = chunk.reshape(-1, order=new_order)
        return ArrayHandle(chunk)


@define
class GzipCodecConfigurationMetadata:
    level: int = 5


@define
class GzipCodecMetadata:
    configuration: GzipCodecConfigurationMetadata
    name: Final = "gzip"

    def decode(
        self,
        value: ValueHandle,
        _selection: Tuple[slice, ...],
        _array_metadata: "CoreArrayMetadata",
    ) -> ValueHandle:
        buf = value.tobytes()
        if buf is None:
            return NoneHandle()
        return BufferHandle(GZip(self.configuration.level).decode(buf))

    def encode(
        self,
        value: ValueHandle,
        _selection: Tuple[slice, ...],
        _array_metadata: "CoreArrayMetadata",
    ) -> ValueHandle:
        buf = value.tobytes()
        if buf is None:
            return NoneHandle()
        return BufferHandle(GZip(self.configuration.level).encode(buf))


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
