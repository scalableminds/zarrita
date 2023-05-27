import functools
import itertools
import math
from crc32c import crc32c
from typing import (
    TYPE_CHECKING,
    Dict,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Union,
)

import numpy as np
from attrs import field, frozen

from zarrita.common import get_order, is_total_slice
from zarrita.indexing import BasicIndexer
from zarrita.value_handle import (
    ArrayValueHandle,
    BufferValueHandle,
    NoneValueHandle,
    ValueHandle,
)

if TYPE_CHECKING:
    from zarrita.array import CoreArrayMetadata
    from zarrita.codecs import CodecMetadata


MAX_UINT_64 = 2**64 - 1


@frozen
class ShardingCodecConfigurationMetadata:
    chunk_shape: Tuple[int, ...]
    codecs: List["CodecMetadata"] = field(factory=list)


def product(tup: Tuple[int, ...]) -> int:
    return functools.reduce(lambda x, y: x * y, tup, 1)


def c_order_iter(chunk_shape: Tuple[int, ...]) -> Iterator[Tuple[int, ...]]:
    return itertools.product(*(range(x) for x in chunk_shape))


def morton_order_iter(chunk_shape: Tuple[int, ...]) -> Iterator[Tuple[int, ...]]:
    def decode_morton(z: int, chunk_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        # Inspired by compressed morton code as implemented in Neuroglancer
        # https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/volume.md#compressed-morton-code
        bits = tuple(math.ceil(math.log2(c)) for c in chunk_shape)
        max_coords_bits = max(*bits)
        input_bit = 0
        input_value = z
        out = [0 for _ in range(len(chunk_shape))]

        for coord_bit in range(max_coords_bits):
            for dim in range(len(chunk_shape)):
                if coord_bit < bits[dim]:
                    bit = (input_value >> input_bit) & 1
                    out[dim] |= bit << coord_bit
                    input_bit += 1
        return tuple(out)

    for i in range(product(chunk_shape)):
        yield decode_morton(i, chunk_shape)


class _ShardIndex(NamedTuple):
    # dtype uint64, shape (chunks_per_shard_0, chunks_per_shard_1, ..., 2)
    offsets_and_lengths: np.ndarray

    def _localize_chunk(self, chunk: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple(
            chunk_i % shard_i
            for chunk_i, shard_i in zip(chunk, self.offsets_and_lengths.shape)
        )

    def is_all_empty(self) -> bool:
        return np.array_equiv(self.offsets_and_lengths, MAX_UINT_64)

    def get_chunk_slice(self, chunk: Tuple[int, ...]) -> Optional[slice]:
        localized_chunk = self._localize_chunk(chunk)
        chunk_start, chunk_len = self.offsets_and_lengths[localized_chunk]
        if (chunk_start, chunk_len) == (MAX_UINT_64, MAX_UINT_64):
            return None
        else:
            return slice(int(chunk_start), int(chunk_start + chunk_len))

    def set_chunk_slice(
        self, chunk: Tuple[int, ...], chunk_slice: Optional[slice]
    ) -> None:
        localized_chunk = self._localize_chunk(chunk)
        if chunk_slice is None:
            self.offsets_and_lengths[localized_chunk] = (MAX_UINT_64, MAX_UINT_64)
        else:
            self.offsets_and_lengths[localized_chunk] = (
                chunk_slice.start,
                chunk_slice.stop - chunk_slice.start,
            )

    def to_bytes(self) -> bytes:
        index_bytes = self.offsets_and_lengths.tobytes(order="C")
        return index_bytes + np.uint32(crc32c(index_bytes)).tobytes()

    @property
    def byte_length(self) -> int:
        return self.index_byte_length(self.offsets_and_lengths.shape)

    @classmethod
    def from_bytes(
        cls,
        buffer: Union[bytes, bytearray],
        chunks_per_shard: Tuple[int, ...],
    ) -> "_ShardIndex":
        try:
            crc32_bytes = buffer[-4:]
            index_bytes = buffer[:-4]

            assert np.uint32(crc32c(index_bytes)).tobytes() == crc32_bytes

            return cls(
                offsets_and_lengths=np.frombuffer(
                    bytearray(index_bytes), dtype="<u8"
                ).reshape(*chunks_per_shard, 2, order="C"),
            )
        except ValueError as e:  # pragma: no cover
            raise RuntimeError from e

    @classmethod
    def create_empty(cls, chunks_per_shard: Tuple[int, ...]) -> "_ShardIndex":
        # reserving 2*64bit per chunk for offset and length + 32bit checksum:
        index_bytes = MAX_UINT_64.to_bytes(8, byteorder="little") * (
            2 * product(chunks_per_shard)
        )
        return cls.from_bytes(
            index_bytes + np.uint32(crc32c(index_bytes)).tobytes(),
            chunks_per_shard=chunks_per_shard,
        )

    @classmethod
    def index_byte_length(cls, chunks_per_shard: Tuple[int, ...]) -> int:
        return 16 * product(chunks_per_shard) + 4


@frozen
class ShardingCodecMetadata:
    configuration: ShardingCodecConfigurationMetadata
    name: Literal["sharding_indexed"] = "sharding_indexed"

    supports_partial_decode = True
    supports_partial_encode = True

    async def decode(
        self,
        value_handle: ValueHandle,
        selection: Tuple[slice, ...],
        array_metadata: "CoreArrayMetadata",
    ) -> ValueHandle:
        if isinstance(value_handle, NoneValueHandle):
            return NoneValueHandle()

        shard_shape = array_metadata.chunk_shape
        chunk_shape = self.configuration.chunk_shape
        chunks_per_shard = tuple(s // c for s, c in zip(shard_shape, chunk_shape))

        indexer = BasicIndexer(
            selection,
            shape=shard_shape,
            chunk_shape=chunk_shape,
        )

        # setup output array
        out = np.zeros(
            shard_shape,
            dtype=array_metadata.dtype,
            order=get_order(self.configuration.codecs),
        )

        indexed_chunks = list(indexer)
        all_chunk_coords = set(chunk_coords for chunk_coords, _, _ in indexed_chunks)

        # reading bytes of all requested chunks
        shard_dict: Dict[Tuple[int, ...], ValueHandle] = {}
        if self._is_total_shard(all_chunk_coords, chunks_per_shard):
            # read entire shard
            shard_dict = await self._load_full_shard(value_handle, chunks_per_shard)
        else:
            # read some chunks within the shard
            shard_index = await self._load_shard_index(value_handle, chunks_per_shard)
            shard_dict = {}
            for chunk_coords in all_chunk_coords:
                chunk_byte_slice = shard_index.get_chunk_slice(chunk_coords)
                if chunk_byte_slice:
                    chunk_bytes = await value_handle[chunk_byte_slice].tobytes()
                    if chunk_bytes:
                        shard_dict[chunk_coords] = BufferValueHandle(chunk_bytes)

        # decoding chunks and writing them into the output buffer
        for chunk_coords, chunk_selection, out_selection in indexed_chunks:
            chunk_value = shard_dict.get(chunk_coords, NoneValueHandle())
            chunk = await self._decode_chunk(
                chunk_value, chunk_selection, array_metadata
            )
            if chunk is not None:
                tmp = chunk[chunk_selection]
                out[out_selection] = tmp
            elif array_metadata.fill_value is not None:
                out[out_selection] = array_metadata.fill_value

        return ArrayValueHandle(out)

    async def _decode_chunk(
        self,
        value_handle: ValueHandle,
        selection: Tuple[slice, ...],
        array_metadata: "CoreArrayMetadata",
    ) -> Optional[np.ndarray]:
        if isinstance(value_handle, NoneValueHandle):
            return None

        from zarrita.array import CoreArrayMetadata

        # rewriting the metadata to scope it to the shard
        core_metadata = CoreArrayMetadata(
            shape=array_metadata.chunk_shape,
            chunk_shape=self.configuration.chunk_shape,
            data_type=array_metadata.data_type,
            fill_value=array_metadata.fill_value,
        )

        # applying codecs in revers e order
        for codec_metadata in self.configuration.codecs[::-1]:
            value_handle = await codec_metadata.decode(
                value_handle, selection, core_metadata
            )

        chunk = await value_handle.toarray()
        if chunk is None:
            return None

        # ensure correct dtype
        if str(chunk.dtype) != array_metadata.data_type.name:
            chunk = chunk.view(np.dtype(array_metadata.data_type.name))

        # ensure correct chunk shape
        if chunk.shape != self.configuration.chunk_shape:
            chunk = chunk.reshape(self.configuration.chunk_shape)

        return chunk

    async def encode(
        self,
        value: ValueHandle,
        array_metadata: "CoreArrayMetadata",
    ) -> ValueHandle:
        shard_array = await value.toarray()
        if shard_array is None:
            return NoneValueHandle()

        shard_shape = array_metadata.chunk_shape
        chunk_shape = self.configuration.chunk_shape
        chunks_per_shard = tuple(s // c for s, c in zip(shard_shape, chunk_shape))

        indexer = list(
            BasicIndexer(
                tuple(slice(0, s) for s in shard_shape),
                shape=shard_shape,
                chunk_shape=chunk_shape,
            )
        )

        # assembling and encoding chunks within the shard
        shard_dict: Dict[Tuple[int, ...], ValueHandle] = {}
        for chunk_coords, chunk_selection, out_selection in indexer:
            if is_total_slice(chunk_selection, chunk_shape):
                chunk = shard_array[out_selection]
            else:
                # handling writing partial chunks
                chunk = np.empty(
                    chunk_shape,
                    dtype=array_metadata.dtype,
                    order="C",
                )
                if array_metadata.fill_value:
                    chunk.fill(array_metadata.fill_value)
                chunk[chunk_selection] = shard_array[out_selection]
            if array_metadata.fill_value and np.all(chunk == array_metadata.fill_value):
                shard_dict[chunk_coords] = NoneValueHandle()  # TODO
            else:
                shard_dict[chunk_coords] = await self._encode_chunk(
                    chunk, array_metadata
                )

        if all(
            isinstance(chunk_value, NoneValueHandle)  # TODO
            for chunk_value in shard_dict.values()
        ):
            return NoneValueHandle()
        return BufferValueHandle(await self._build_shard(shard_dict, chunks_per_shard))

    async def encode_partial(
        self,
        old_value_handle: ValueHandle,
        shard_array: np.ndarray,
        selection: Tuple[slice, ...],
        array_metadata: "CoreArrayMetadata",
    ) -> ValueHandle:
        shard_shape = array_metadata.chunk_shape
        chunk_shape = self.configuration.chunk_shape
        chunks_per_shard = tuple(s // c for s, c in zip(shard_shape, chunk_shape))

        shard_dict = await self._load_full_shard(old_value_handle, chunks_per_shard)

        indexer = list(
            BasicIndexer(
                selection,
                shape=shard_shape,
                chunk_shape=chunk_shape,
            )
        )

        for chunk_coords, chunk_selection, out_selection in indexer:
            if is_total_slice(chunk_selection, chunk_shape):
                chunk_array = shard_array[out_selection]
            else:
                # handling writing partial chunks
                # read chunk first
                tmp = await self._decode_chunk(
                    shard_dict[chunk_coords],
                    tuple(slice(0, c) for c in chunk_shape),
                    array_metadata,
                )
                # merge new value
                if tmp is None:
                    chunk_array = np.empty(
                        chunk_shape,
                        dtype=array_metadata.dtype,
                        order="C",
                    )
                    if array_metadata.fill_value:
                        chunk_array.fill(array_metadata.fill_value)
                else:
                    chunk_array = tmp.copy(order="K")  # make a writable copy
                chunk_array[chunk_selection] = shard_array[out_selection]

            if array_metadata.fill_value and np.all(
                chunk_array == array_metadata.fill_value
            ):
                shard_dict[chunk_coords] = NoneValueHandle()  # TODO
            else:
                shard_dict[chunk_coords] = await self._encode_chunk(
                    chunk_array, array_metadata
                )

        if all(
            isinstance(chunk_value, NoneValueHandle)  # TODO
            for chunk_value in shard_dict.values()
        ):
            return NoneValueHandle()
        return BufferValueHandle(await self._build_shard(shard_dict, chunks_per_shard))

    async def _encode_chunk(
        self,
        chunk: np.ndarray,
        array_metadata: "CoreArrayMetadata",
    ):
        from zarrita.array import CoreArrayMetadata

        # rewriting the metadata to scope it to the shard
        core_metadata = CoreArrayMetadata(
            shape=array_metadata.chunk_shape,
            chunk_shape=self.configuration.chunk_shape,
            data_type=array_metadata.data_type,
            fill_value=array_metadata.fill_value,
        )

        encoded_chunk_value: ValueHandle = ArrayValueHandle(chunk)
        for codec in self.configuration.codecs:
            encoded_chunk_value = await codec.encode(
                encoded_chunk_value,
                array_metadata,
            )

        return encoded_chunk_value

    def _is_total_shard(
        self, all_chunk_coords: Set[Tuple[int, ...]], chunks_per_shard: Tuple[int, ...]
    ) -> bool:
        return len(all_chunk_coords) == product(chunks_per_shard) and all(
            chunk_coords in all_chunk_coords
            for chunk_coords in c_order_iter(chunks_per_shard)
        )

    async def _load_shard_index(
        self, value_handle: ValueHandle, chunks_per_shard: Tuple[int, ...]
    ) -> _ShardIndex:
        index_bytes = await value_handle[
            -_ShardIndex.index_byte_length(chunks_per_shard) :
        ].tobytes()
        assert isinstance(index_bytes, bytes)
        if index_bytes is not None:
            return _ShardIndex.from_bytes(index_bytes, chunks_per_shard)
        else:
            return _ShardIndex.create_empty(chunks_per_shard)

    async def _build_shard(
        self,
        shard_dict: Dict[Tuple[int, ...], ValueHandle],
        chunks_per_shard: Tuple[int, ...],
    ) -> bytes:
        shard_index = _ShardIndex.create_empty(chunks_per_shard)
        byte_shard_dict = {
            chunk_coords: await chunk_value.tobytes()
            for chunk_coords, chunk_value in shard_dict.items()
        }

        # output buffer
        shard_bytes = bytearray(
            sum(
                len(chunk_bytes)
                for chunk_bytes in byte_shard_dict.values()
                if chunk_bytes is not None
            )
            + shard_index.byte_length
        )

        # write chunks within shard in morton order
        byte_offset = 0
        for chunk_coords in morton_order_iter(chunks_per_shard):
            chunk_bytes = byte_shard_dict.get(chunk_coords, None)
            if chunk_bytes is not None:
                byte_offset_end = byte_offset + len(chunk_bytes)
                shard_bytes[byte_offset : byte_offset + len(chunk_bytes)] = chunk_bytes
                shard_index.set_chunk_slice(
                    chunk_coords, slice(byte_offset, byte_offset_end)
                )
                byte_offset = byte_offset_end
        shard_bytes[-shard_index.byte_length :] = shard_index.to_bytes()
        return bytes(shard_bytes)

    async def _load_full_shard(
        self, value_handle: ValueHandle, chunks_per_shard: Tuple[int, ...]
    ) -> Dict[Tuple[int, ...], ValueHandle]:
        shard_bytes = await value_handle.tobytes()
        if shard_bytes:
            assert isinstance(shard_bytes, bytes)
            shard_bytes_view = memoryview(shard_bytes)
            index_bytes = shard_bytes_view[
                -_ShardIndex.index_byte_length(chunks_per_shard) :
            ]
            shard_index = _ShardIndex.from_bytes(index_bytes, chunks_per_shard)

            shard_dict: Dict[Tuple[int, ...], ValueHandle] = {}
            for chunk_coords in c_order_iter(chunks_per_shard):
                chunk_byte_slice = shard_index.get_chunk_slice(chunk_coords)
                if chunk_byte_slice is not None:
                    shard_dict[chunk_coords] = BufferValueHandle(
                        shard_bytes_view[chunk_byte_slice]
                    )
                else:  # TODO
                    shard_dict[chunk_coords] = NoneValueHandle()

            return shard_dict
        else:
            return {  # TODO
                chunk_coords: NoneValueHandle()
                for chunk_coords in c_order_iter(chunks_per_shard)
            }
