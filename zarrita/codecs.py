from typing import TYPE_CHECKING, Any, Dict, List, Literal, Tuple, Union

import numpy as np
from attr import asdict, frozen
from cattrs import register_structure_hook, structure
from numcodecs.blosc import Blosc
from numcodecs.gzip import GZip

from zarrita.sharding import ShardingCodecConfigurationMetadata, ShardingCodecMetadata

if TYPE_CHECKING:
    from zarrita.array import CoreArrayMetadata

from zarrita.store import ArrayHandle, BufferHandle, NoneHandle, ValueHandle


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


@frozen
class TransposeCodecConfigurationMetadata:
    order: Union[Literal["C", "F"], Tuple[int, ...]] = "C"


@frozen
class TransposeCodecMetadata:
    configuration: TransposeCodecConfigurationMetadata
    name: Literal["transpose"] = "transpose"

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


@frozen
class GzipCodecConfigurationMetadata:
    level: int = 5


@frozen
class GzipCodecMetadata:
    configuration: GzipCodecConfigurationMetadata
    name: Literal["gzip"] = "gzip"

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


def structure_codec_metadata(d: Dict[str, Any], _t) -> CodecMetadata:
    if d["name"] == "blosc":
        return structure(d, BloscCodecMetadata)
    if d["name"] == "endian":
        return structure(d, EndianCodecMetadata)
    if d["name"] == "transpose":
        return structure(d, TransposeCodecMetadata)
    if d["name"] == "gzip":
        return structure(d, GzipCodecMetadata)
    if d["name"] == "sharding_indexed":
        return structure(d, ShardingCodecMetadata)
    raise KeyError


register_structure_hook(CodecMetadata, structure_codec_metadata)
