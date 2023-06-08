from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from attr import field, frozen

from zarrita.common import ChunkCoords

if TYPE_CHECKING:
    from zarrita.array import ArrayRuntimeConfiguration


class DataType(Enum):
    bool = "bool"
    int8 = "int8"
    int16 = "int16"
    int32 = "int32"
    int64 = "int64"
    uint8 = "uint8"
    uint16 = "uint16"
    uint32 = "uint32"
    uint64 = "uint64"
    float32 = "float32"
    float64 = "float64"


dtype_to_data_type = {
    "bool": "bool",
    "|i1": "int8",
    "<i2": "int16",
    "<i4": "int32",
    "<i8": "int64",
    "|u1": "uint8",
    "<u2": "uint16",
    "<u4": "uint32",
    "<u8": "uint64",
    "<f4": "float32",
    "<f8": "float64",
}

data_type_to_numpy = {
    DataType.bool: "bool",
    DataType.int8: "i1",
    DataType.int16: "i2",
    DataType.int32: "i4",
    DataType.int64: "i8",
    DataType.uint8: "u1",
    DataType.uint16: "u2",
    DataType.uint32: "u4",
    DataType.uint64: "u8",
    DataType.float32: "f4",
    DataType.float64: "f8",
}


@frozen
class RegularChunkGridConfigurationMetadata:
    chunk_shape: ChunkCoords


@frozen
class RegularChunkGridMetadata:
    configuration: RegularChunkGridConfigurationMetadata
    name: Literal["regular"] = "regular"


@frozen
class DefaultChunkKeyEncodingConfigurationMetadata:
    separator: Literal[".", "/"] = "/"


@frozen
class DefaultChunkKeyEncodingMetadata:
    configuration: DefaultChunkKeyEncodingConfigurationMetadata = (
        DefaultChunkKeyEncodingConfigurationMetadata()
    )
    name: Literal["default"] = "default"

    def decode_chunk_key(self, chunk_key: str) -> ChunkCoords:
        if chunk_key == "c":
            return ()
        return tuple(map(int, chunk_key[1:].split(self.configuration.separator)))

    def encode_chunk_key(self, chunk_coords: ChunkCoords) -> str:
        return self.configuration.separator.join(map(str, ("c",) + chunk_coords))


@frozen
class V2ChunkKeyEncodingConfigurationMetadata:
    separator: Literal[".", "/"] = "."


@frozen
class V2ChunkKeyEncodingMetadata:
    configuration: V2ChunkKeyEncodingConfigurationMetadata = (
        V2ChunkKeyEncodingConfigurationMetadata()
    )
    name: Literal["v2"] = "v2"

    def decode_chunk_key(self, chunk_key: str) -> ChunkCoords:
        return tuple(map(int, chunk_key.split(self.configuration.separator)))

    def encode_chunk_key(self, chunk_coords: ChunkCoords) -> str:
        chunk_identifier = self.configuration.separator.join(map(str, chunk_coords))
        return "0" if chunk_identifier == "" else chunk_identifier


ChunkKeyEncodingMetadata = Union[
    DefaultChunkKeyEncodingMetadata, V2ChunkKeyEncodingMetadata
]


@frozen
class BloscCodecConfigurationMetadata:
    cname: Literal["lz4", "lz4hc", "blosclz", "zstd", "snappy", "zlib"] = "zstd"
    clevel: int = 5
    shuffle: Literal["noshuffle", "shuffle", "bitshuffle"] = "noshuffle"
    blocksize: int = 0


@frozen
class BloscCodecMetadata:
    configuration: BloscCodecConfigurationMetadata
    name: Literal["blosc"] = "blosc"


@frozen
class EndianCodecConfigurationMetadata:
    endian: Literal["big", "little"] = "little"


@frozen
class EndianCodecMetadata:
    configuration: EndianCodecConfigurationMetadata
    name: Literal["endian"] = "endian"


@frozen
class TransposeCodecConfigurationMetadata:
    order: Union[Literal["C", "F"], Tuple[int, ...]] = "C"


@frozen
class TransposeCodecMetadata:
    configuration: TransposeCodecConfigurationMetadata
    name: Literal["transpose"] = "transpose"


@frozen
class GzipCodecConfigurationMetadata:
    level: int = 5


@frozen
class GzipCodecMetadata:
    configuration: GzipCodecConfigurationMetadata
    name: Literal["gzip"] = "gzip"


@frozen
class ShardingCodecConfigurationMetadata:
    chunk_shape: ChunkCoords
    codecs: List["CodecMetadata"] = field(factory=list)


@frozen
class ShardingCodecMetadata:
    configuration: ShardingCodecConfigurationMetadata
    name: Literal["sharding_indexed"] = "sharding_indexed"


CodecMetadata = Union[
    BloscCodecMetadata,
    EndianCodecMetadata,
    TransposeCodecMetadata,
    GzipCodecMetadata,
    ShardingCodecMetadata,
]


@frozen
class CoreArrayMetadata:
    shape: ChunkCoords
    chunk_shape: ChunkCoords
    data_type: DataType
    fill_value: Any
    runtime_configuration: "ArrayRuntimeConfiguration"

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.data_type.value)


@frozen
class ArrayMetadata:
    shape: ChunkCoords
    data_type: DataType
    chunk_grid: RegularChunkGridMetadata
    chunk_key_encoding: ChunkKeyEncodingMetadata
    fill_value: Any
    attributes: Dict[str, Any] = field(factory=dict)
    codecs: List[CodecMetadata] = field(factory=list)
    dimension_names: Optional[Tuple[str, ...]] = None
    zarr_format: Literal[3] = 3
    node_type: Literal["array"] = "array"

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.data_type.value)


@frozen
class ArrayV2Metadata:
    shape: ChunkCoords
    chunks: ChunkCoords
    dtype: np.dtype
    fill_value: Union[None, int, float] = 0
    order: Literal["C", "F"] = "C"
    filters: Optional[List[Dict[str, Any]]] = None
    dimension_separator: Literal[".", "/"] = "."
    compressor: Optional[Dict[str, Any]] = None
    zarr_format: Literal[2] = 2
