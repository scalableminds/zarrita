from __future__ import annotations

from abc import ABC, abstractmethod
from functools import reduce
from typing import TYPE_CHECKING, Iterable, List, Literal, Optional, Tuple, Union
from warnings import warn

import attr
import numcodecs
import numpy as np
from attr import asdict, frozen
from crc32c import crc32c
from numcodecs.blosc import Blosc
from numcodecs.gzip import GZip

from zarrita.common import BytesLike, to_thread
from zarrita.metadata import (
    BloscCodecConfigurationMetadata,
    BloscCodecMetadata,
    CodecMetadata,
    Crc32cCodecMetadata,
    EndianCodecConfigurationMetadata,
    EndianCodecMetadata,
    GzipCodecConfigurationMetadata,
    GzipCodecMetadata,
    ShardingCodecConfigurationMetadata,
    ShardingCodecMetadata,
    TransposeCodecConfigurationMetadata,
    TransposeCodecMetadata,
)

if TYPE_CHECKING:
    from zarrita.metadata import CoreArrayMetadata

# See https://zarr.readthedocs.io/en/stable/tutorial.html#configuring-blosc
numcodecs.blosc.use_threads = False


class Codec(ABC):
    supports_partial_decode: bool
    supports_partial_encode: bool
    is_fixed_size: bool
    array_metadata: CoreArrayMetadata

    @abstractmethod
    def compute_encoded_size(self, input_byte_length: int) -> int:
        pass


class ArrayArrayCodec(Codec):
    @abstractmethod
    async def decode(
        self,
        chunk_array: np.ndarray,
    ) -> np.ndarray:
        pass

    @abstractmethod
    async def encode(
        self,
        chunk_array: np.ndarray,
    ) -> Optional[np.ndarray]:
        pass


class ArrayBytesCodec(Codec):
    @abstractmethod
    async def decode(
        self,
        chunk_array: BytesLike,
    ) -> np.ndarray:
        pass

    @abstractmethod
    async def encode(
        self,
        chunk_array: np.ndarray,
    ) -> Optional[BytesLike]:
        pass


class BytesBytesCodec(Codec):
    @abstractmethod
    async def decode(
        self,
        chunk_array: BytesLike,
    ) -> BytesLike:
        pass

    @abstractmethod
    async def encode(
        self,
        chunk_array: BytesLike,
    ) -> Optional[BytesLike]:
        pass


@frozen
class CodecPipeline:
    codecs: List[Codec]

    @classmethod
    def from_metadata(
        cls,
        codecs_metadata: Iterable[CodecMetadata],
        array_metadata: CoreArrayMetadata,
    ) -> CodecPipeline:
        out: List[Codec] = []
        for codec_metadata in codecs_metadata or []:
            if codec_metadata.name == "blosc":
                out.append(BloscCodec.from_metadata(codec_metadata, array_metadata))
            elif codec_metadata.name == "gzip":
                out.append(GzipCodec.from_metadata(codec_metadata, array_metadata))
            elif codec_metadata.name == "transpose":
                out.append(TransposeCodec.from_metadata(codec_metadata, array_metadata))
            elif codec_metadata.name == "endian":
                out.append(EndianCodec.from_metadata(codec_metadata, array_metadata))
            elif codec_metadata.name == "crc32c":
                out.append(Crc32cCodec.from_metadata(codec_metadata, array_metadata))
            elif codec_metadata.name == "sharding_indexed":
                from zarrita.sharding import ShardingCodec

                out.append(ShardingCodec.from_metadata(codec_metadata, array_metadata))
            else:
                raise RuntimeError(f"Unsupported codec: {codec_metadata}")
        CodecPipeline._validate_codecs(out, array_metadata)
        return cls(out)

    @staticmethod
    def _validate_codecs(
        codecs: List[Codec], array_metadata: CoreArrayMetadata
    ) -> None:
        from zarrita.sharding import ShardingCodec

        assert any(
            isinstance(codec, ArrayBytesCodec) for codec in codecs
        ), "Exactly one array-to-bytes codec is required."

        prev_codec: Optional[Codec] = None
        for codec in codecs:
            if prev_codec is not None:
                assert not isinstance(codec, ArrayBytesCodec) or not isinstance(
                    prev_codec, ArrayBytesCodec
                ), (
                    f"ArrayBytesCodec '{type(codec)}' cannot follow after "
                    + f"ArrayBytesCodec '{type(prev_codec)}' because exactly "
                    + "1 ArrayBytesCodec is allowed."
                )
                assert not isinstance(codec, ArrayBytesCodec) or not isinstance(
                    prev_codec, BytesBytesCodec
                ), (
                    f"ArrayBytesCodec '{type(codec)}' cannot follow after "
                    + f"BytesBytesCodec '{type(prev_codec)}'."
                )
                assert not isinstance(codec, ArrayArrayCodec) or not isinstance(
                    prev_codec, ArrayBytesCodec
                ), (
                    f"ArrayArrayCodec '{type(codec)}' cannot follow after "
                    + f"ArrayBytesCodec '{type(prev_codec)}'."
                )
                assert not isinstance(codec, ArrayArrayCodec) or not isinstance(
                    prev_codec, BytesBytesCodec
                ), (
                    f"ArrayArrayCodec '{type(codec)}' cannot follow after "
                    + f"BytesBytesCodec '{type(prev_codec)}'."
                )

            if isinstance(codec, ShardingCodec):
                assert len(codec.configuration.chunk_shape) == len(
                    array_metadata.shape
                ), (
                    "The shard's `chunk_shape` and array's `shape` need to have the "
                    + "same number of dimensions."
                )
                assert all(
                    s % c == 0
                    for s, c in zip(
                        array_metadata.chunk_shape,
                        codec.configuration.chunk_shape,
                    )
                ), (
                    "The array's `chunk_shape` needs to be divisible by the "
                    + "shard's inner `chunk_shape`."
                )
            prev_codec = codec

        if (
            any(isinstance(codec, ShardingCodec) for codec in codecs)
            and len(codecs) > 1
        ):
            warn(
                "Combining a `sharding_indexed` codec disables partial reads and "
                + "writes, which may lead to inefficient performance."
            )

    def _array_array_codecs(self) -> List[ArrayArrayCodec]:
        return [codec for codec in self.codecs if isinstance(codec, ArrayArrayCodec)]

    def _array_bytes_codec(self) -> ArrayBytesCodec:
        return next(
            codec for codec in self.codecs if isinstance(codec, ArrayBytesCodec)
        )

    def _bytes_bytes_codecs(self) -> List[BytesBytesCodec]:
        return [codec for codec in self.codecs if isinstance(codec, BytesBytesCodec)]

    async def decode(self, chunk_bytes: BytesLike) -> np.ndarray:
        for bb_codec in self._bytes_bytes_codecs()[::-1]:
            chunk_bytes = await bb_codec.decode(chunk_bytes)

        chunk_array = await self._array_bytes_codec().decode(chunk_bytes)

        for aa_codec in self._array_array_codecs()[::-1]:
            chunk_array = await aa_codec.decode(chunk_array)

        return chunk_array

    async def encode(self, chunk_array: np.ndarray) -> Optional[BytesLike]:
        for aa_codec in self._array_array_codecs():
            chunk_array_maybe = await aa_codec.encode(chunk_array)
            if chunk_array_maybe is None:
                return None
            chunk_array = chunk_array_maybe

        chunk_bytes_maybe = await self._array_bytes_codec().encode(chunk_array)
        if chunk_bytes_maybe is None:
            return None
        chunk_bytes = chunk_bytes_maybe

        for bb_codec in self._bytes_bytes_codecs():
            chunk_bytes_maybe = await bb_codec.encode(chunk_bytes)
            if chunk_bytes_maybe is None:
                return None
            chunk_bytes = chunk_bytes_maybe

        return chunk_bytes

    def compute_encoded_size(self, byte_length: int) -> int:
        return reduce(
            lambda acc, codec: codec.compute_encoded_size(acc), self.codecs, byte_length
        )


@frozen
class BloscCodec(BytesBytesCodec):
    array_metadata: CoreArrayMetadata
    configuration: BloscCodecConfigurationMetadata
    blosc_codec: Blosc
    is_fixed_size = False

    @classmethod
    def from_metadata(
        cls, codec_metadata: BloscCodecMetadata, array_metadata: CoreArrayMetadata
    ) -> BloscCodec:
        configuration = codec_metadata.configuration
        if configuration.typesize == 0:
            configuration = attr.evolve(
                configuration, typesize=array_metadata.data_type.byte_count
            )
        config_dict = asdict(codec_metadata.configuration)
        config_dict.pop("typesize", None)
        map_shuffle_str_to_int = {"noshuffle": 0, "shuffle": 1, "bitshuffle": 2}
        config_dict["shuffle"] = map_shuffle_str_to_int[config_dict["shuffle"]]
        return cls(
            array_metadata=array_metadata,
            configuration=configuration,
            blosc_codec=Blosc.from_config(config_dict),
        )

    async def decode(
        self,
        chunk_bytes: bytes,
    ) -> BytesLike:
        return await to_thread(self.blosc_codec.decode, chunk_bytes)

    async def encode(
        self,
        chunk_bytes: bytes,
    ) -> Optional[BytesLike]:
        chunk_array = np.frombuffer(chunk_bytes, dtype=self.array_metadata.dtype)
        return await to_thread(self.blosc_codec.encode, chunk_array)

    def compute_encoded_size(self, _input_byte_length: int) -> int:
        raise NotImplementedError


@frozen
class EndianCodec(ArrayBytesCodec):
    array_metadata: CoreArrayMetadata
    configuration: EndianCodecConfigurationMetadata
    is_fixed_size = True

    @classmethod
    def from_metadata(
        cls, codec_metadata: EndianCodecMetadata, array_metadata: CoreArrayMetadata
    ) -> EndianCodec:
        return cls(
            array_metadata=array_metadata,
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

    async def decode(
        self,
        chunk_bytes: BytesLike,
    ) -> np.ndarray:
        if self.configuration.endian == "little":
            prefix = "<"
        else:
            prefix = ">"
        dtype = np.dtype(
            f"{prefix}{self.array_metadata.data_type.to_numpy_shortname()}"
        )
        chunk_array = np.frombuffer(chunk_bytes, dtype)

        # ensure correct chunk shape
        if chunk_array.shape != self.array_metadata.chunk_shape:
            chunk_array = chunk_array.reshape(
                self.array_metadata.chunk_shape,
            )
        return chunk_array

    async def encode(
        self,
        chunk_array: np.ndarray,
    ) -> Optional[BytesLike]:
        byteorder = self._get_byteorder(chunk_array)
        if self.configuration.endian != byteorder:
            new_dtype = chunk_array.dtype.newbyteorder(self.configuration.endian)
            chunk_array = chunk_array.astype(new_dtype)
        return chunk_array.tobytes()

    def compute_encoded_size(self, input_byte_length: int) -> int:
        return input_byte_length


@frozen
class TransposeCodec(ArrayArrayCodec):
    array_metadata: CoreArrayMetadata
    configuration: TransposeCodecConfigurationMetadata
    is_fixed_size = True

    @classmethod
    def from_metadata(
        cls, codec_metadata: TransposeCodecMetadata, array_metadata: CoreArrayMetadata
    ) -> TransposeCodec:
        return cls(
            array_metadata=array_metadata,
            configuration=codec_metadata.configuration,
        )

    async def decode(
        self,
        chunk_array: np.ndarray,
    ) -> np.ndarray:
        new_order = self.configuration.order
        chunk_array = chunk_array.view(np.dtype(self.array_metadata.data_type.value))
        if isinstance(new_order, tuple):
            chunk_array = chunk_array.transpose(new_order)
        elif new_order == "F":
            chunk_array = chunk_array.T
        return chunk_array

    async def encode(
        self,
        chunk_array: np.ndarray,
    ) -> Optional[np.ndarray]:
        new_order = self.configuration.order
        if isinstance(new_order, tuple):
            chunk_array = chunk_array.transpose(new_order)
        elif new_order == "F":
            chunk_array = chunk_array.T
        return chunk_array.reshape(-1, order="C")

    def compute_encoded_size(self, input_byte_length: int) -> int:
        return input_byte_length


@frozen
class GzipCodec(BytesBytesCodec):
    array_metadata: CoreArrayMetadata
    configuration: GzipCodecConfigurationMetadata
    is_fixed_size = True

    @classmethod
    def from_metadata(
        cls, codec_metadata: GzipCodecMetadata, array_metadata: CoreArrayMetadata
    ) -> GzipCodec:
        return cls(
            array_metadata=array_metadata,
            configuration=codec_metadata.configuration,
        )

    async def decode(
        self,
        chunk_bytes: bytes,
    ) -> BytesLike:
        return await to_thread(GZip(self.configuration.level).decode, chunk_bytes)

    async def encode(
        self,
        chunk_bytes: bytes,
    ) -> Optional[BytesLike]:
        return await to_thread(GZip(self.configuration.level).encode, chunk_bytes)

    def compute_encoded_size(self, _input_byte_length: int) -> int:
        raise NotImplementedError


@frozen
class Crc32cCodec(BytesBytesCodec):
    array_metadata: CoreArrayMetadata
    is_fixed_size = True

    @classmethod
    def from_metadata(
        cls, codec_metadata: Crc32cCodecMetadata, array_metadata: CoreArrayMetadata
    ) -> Crc32cCodec:
        return cls(array_metadata=array_metadata)

    async def decode(
        self,
        chunk_bytes: bytes,
    ) -> BytesLike:
        crc32_bytes = chunk_bytes[-4:]
        inner_bytes = chunk_bytes[:-4]

        assert np.uint32(crc32c(inner_bytes)).tobytes() == bytes(crc32_bytes)
        return inner_bytes

    async def encode(
        self,
        chunk_bytes: bytes,
    ) -> Optional[BytesLike]:
        return chunk_bytes + np.uint32(crc32c(chunk_bytes)).tobytes()

    def compute_encoded_size(self, input_byte_length: int) -> int:
        return input_byte_length + 4


def blosc_codec(
    typesize: int,
    cname: Literal["lz4", "lz4hc", "blosclz", "zstd", "snappy", "zlib"] = "zstd",
    clevel: int = 5,
    shuffle: Literal["noshuffle", "shuffle", "bitshuffle"] = "noshuffle",
    blocksize: int = 0,
) -> BloscCodecMetadata:
    return BloscCodecMetadata(
        configuration=BloscCodecConfigurationMetadata(
            cname=cname,
            clevel=clevel,
            shuffle=shuffle,
            blocksize=blocksize,
            typesize=typesize,
        )
    )


def endian_codec(endian: Literal["big", "little"] = "little") -> EndianCodecMetadata:
    return EndianCodecMetadata(configuration=EndianCodecConfigurationMetadata(endian))


def transpose_codec(
    order: Union[Tuple[int, ...], Literal["C", "F"]]
) -> TransposeCodecMetadata:
    return TransposeCodecMetadata(
        configuration=TransposeCodecConfigurationMetadata(order)
    )


def gzip_codec(level: int = 5) -> GzipCodecMetadata:
    return GzipCodecMetadata(configuration=GzipCodecConfigurationMetadata(level))


def crc32c_codec() -> Crc32cCodecMetadata:
    return Crc32cCodecMetadata()


def sharding_codec(
    chunk_shape: Tuple[int, ...],
    codecs: Optional[List[CodecMetadata]] = None,
    index_codecs: Optional[List[CodecMetadata]] = None,
) -> ShardingCodecMetadata:
    codecs = codecs or [endian_codec()]
    index_codecs = index_codecs or [endian_codec(), crc32c_codec()]
    return ShardingCodecMetadata(
        configuration=ShardingCodecConfigurationMetadata(
            chunk_shape, codecs, index_codecs
        )
    )
