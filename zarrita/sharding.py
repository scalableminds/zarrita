from typing import Iterator, List, Mapping, NamedTuple, Optional, Set, Tuple

import numpy as np
from attrs import frozen
from crc32c import crc32c

from zarrita.codecs import ArrayBytesCodec, Codec
from zarrita.common import (
    BytesLike,
    ChunkCoords,
    SliceSelection,
    concurrent_map,
    product,
)
from zarrita.indexing import (
    BasicIndexer,
    c_order_iter,
    is_total_slice,
    morton_order_iter,
)
from zarrita.metadata import (
    CoreArrayMetadata,
    ShardingCodecConfigurationMetadata,
    ShardingCodecMetadata,
)
from zarrita.value_handle import (
    ArrayValueHandle,
    BufferValueHandle,
    NoneValueHandle,
    ValueHandle,
)

MAX_UINT_64 = 2**64 - 1


class _ShardIndex(NamedTuple):
    # dtype uint64, shape (chunks_per_shard_0, chunks_per_shard_1, ..., 2)
    offsets_and_lengths: np.ndarray

    def _localize_chunk(self, chunk_coords: ChunkCoords) -> ChunkCoords:
        return tuple(
            chunk_i % shard_i
            for chunk_i, shard_i in zip(chunk_coords, self.offsets_and_lengths.shape)
        )

    def is_all_empty(self) -> bool:
        return bool(np.array_equiv(self.offsets_and_lengths, MAX_UINT_64))

    def get_chunk_slice(self, chunk_coords: ChunkCoords) -> Optional[slice]:
        localized_chunk = self._localize_chunk(chunk_coords)
        chunk_start, chunk_len = self.offsets_and_lengths[localized_chunk]
        if (chunk_start, chunk_len) == (MAX_UINT_64, MAX_UINT_64):
            return None
        else:
            return slice(int(chunk_start), int(chunk_start + chunk_len))

    def set_chunk_slice(
        self, chunk_coords: ChunkCoords, chunk_slice: Optional[slice]
    ) -> None:
        localized_chunk = self._localize_chunk(chunk_coords)
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
        buffer: BytesLike,
        chunks_per_shard: ChunkCoords,
    ) -> "_ShardIndex":
        try:
            crc32_bytes = buffer[-4:]
            index_bytes = buffer[:-4]

            assert np.uint32(crc32c(index_bytes)).tobytes() == bytes(crc32_bytes)

            return cls(
                offsets_and_lengths=np.frombuffer(
                    bytearray(index_bytes), dtype="<u8"
                ).reshape(*chunks_per_shard, 2, order="C"),
            )
        except ValueError as e:  # pragma: no cover
            raise RuntimeError from e

    @classmethod
    def create_empty(cls, chunks_per_shard: ChunkCoords) -> "_ShardIndex":
        offsets_and_lengths = np.zeros(chunks_per_shard + (2,), dtype="<u8", order="C")
        offsets_and_lengths.fill(MAX_UINT_64)
        return cls(offsets_and_lengths)

    @classmethod
    def index_byte_length(cls, chunks_per_shard: ChunkCoords) -> int:
        return 16 * product(chunks_per_shard) + 4


class _ShardProxy(Mapping):
    index: _ShardIndex
    buf: BytesLike

    @classmethod
    def from_bytes(cls, buf: BytesLike, chunks_per_shard: ChunkCoords) -> "_ShardProxy":
        index = _ShardIndex.from_bytes(
            buf[-_ShardIndex.index_byte_length(chunks_per_shard) :],
            chunks_per_shard,
        )
        obj = cls()
        obj.buf = memoryview(buf)
        obj.index = index
        return obj

    @classmethod
    def create_empty(cls, chunks_per_shard: ChunkCoords) -> "_ShardProxy":
        index = _ShardIndex.create_empty(chunks_per_shard)
        obj = cls()
        obj.buf = memoryview(b"")
        obj.index = index
        return obj

    def __getitem__(self, chunk_coords: ChunkCoords) -> Optional[BytesLike]:
        chunk_byte_slice = self.index.get_chunk_slice(chunk_coords)
        if chunk_byte_slice:
            return self.buf[chunk_byte_slice]
        return None

    def __len__(self) -> int:
        return int(self.index.offsets_and_lengths.size / 2)

    def __iter__(self) -> Iterator[ChunkCoords]:
        return c_order_iter(self.index.offsets_and_lengths.shape[:-1])


class _ShardBuilder(_ShardProxy):
    buf: bytearray
    index: _ShardIndex

    @classmethod
    def merge_with_morton_order(
        cls,
        chunks_per_shard: ChunkCoords,
        tombstones: Set[ChunkCoords],
        *shard_dicts: Mapping[ChunkCoords, BytesLike]
    ) -> "_ShardBuilder":
        obj = cls.create_empty(chunks_per_shard)
        for chunk_coords in morton_order_iter(chunks_per_shard):
            if tombstones is not None and chunk_coords in tombstones:
                continue
            for shard_dict in shard_dicts:
                maybe_value = shard_dict.get(chunk_coords, None)
                if maybe_value is not None:
                    obj.append(chunk_coords, maybe_value)
        return obj

    @classmethod
    def create_empty(cls, chunks_per_shard: ChunkCoords) -> "_ShardBuilder":
        obj = cls()
        obj.buf = bytearray()
        obj.index = _ShardIndex.create_empty(chunks_per_shard)
        return obj

    def append(self, chunk_coords: ChunkCoords, value: BytesLike):
        chunk_start = len(self.buf)
        chunk_length = len(value)
        self.buf.extend(value)
        self.index.set_chunk_slice(
            chunk_coords, slice(chunk_start, chunk_start + chunk_length)
        )

    def finalize(self) -> BytesLike:
        self.buf.extend(self.index.to_bytes())
        return self.buf


@frozen
class ShardingCodec(ArrayBytesCodec):
    configuration: ShardingCodecConfigurationMetadata
    codecs: List[Codec]

    @classmethod
    def from_metadata(cls, codec_metadata: ShardingCodecMetadata) -> "ShardingCodec":
        return cls(
            configuration=codec_metadata.configuration,
            codecs=Codec.codecs_from_metadata(codec_metadata.configuration.codecs),
        )

    async def inner_decode(
        self, chunk_array: BytesLike, array_metadata: "CoreArrayMetadata"
    ) -> np.ndarray:
        # Not implemented because `decode` is overridden
        raise NotImplementedError

    async def inner_encode(
        self, chunk_array: np.ndarray, array_metadata: CoreArrayMetadata
    ) -> BytesLike:
        # Not implemented because `encode` is overridden
        raise NotImplementedError

    async def decode(
        self,
        value_handle: ValueHandle,
        array_metadata: "CoreArrayMetadata",
    ) -> ValueHandle:
        # print("decode")
        if isinstance(value_handle, NoneValueHandle):
            return NoneValueHandle()

        shard_shape = array_metadata.chunk_shape
        chunk_shape = self.configuration.chunk_shape
        chunks_per_shard = tuple(s // c for s, c in zip(shard_shape, chunk_shape))

        indexer = BasicIndexer(
            tuple(slice(0, s) for s in shard_shape),
            shape=shard_shape,
            chunk_shape=chunk_shape,
        )

        # setup output array
        out = np.zeros(
            shard_shape,
            dtype=array_metadata.dtype,
            order=array_metadata.runtime_configuration.order,
        )
        shard_dict = await self._load_full_shard(value_handle, chunks_per_shard)

        if shard_dict.index.is_all_empty():
            return NoneValueHandle()

        # decoding chunks and writing them into the output buffer
        await concurrent_map(
            [
                (
                    shard_dict,
                    chunk_coords,
                    chunk_selection,
                    out_selection,
                    array_metadata,
                    out,
                )
                for chunk_coords, chunk_selection, out_selection in indexer
            ],
            self._read_chunk,
            array_metadata.runtime_configuration.concurrency,
        )

        return ArrayValueHandle(out)

    async def decode_partial(
        self,
        value_handle: ValueHandle,
        selection: SliceSelection,
        array_metadata: "CoreArrayMetadata",
    ) -> ValueHandle:
        # print("decode_partial")
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
            indexer.shape,
            dtype=array_metadata.dtype,
            order=array_metadata.runtime_configuration.order,
        )

        indexed_chunks = list(indexer)
        all_chunk_coords = set(chunk_coords for chunk_coords, _, _ in indexed_chunks)

        # reading bytes of all requested chunks
        shard_dict: Mapping[ChunkCoords, BytesLike] = {}
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
                        shard_dict[chunk_coords] = chunk_bytes

        # decoding chunks and writing them into the output buffer
        await concurrent_map(
            [
                (
                    shard_dict,
                    chunk_coords,
                    chunk_selection,
                    out_selection,
                    array_metadata,
                    out,
                )
                for chunk_coords, chunk_selection, out_selection in indexed_chunks
            ],
            self._read_chunk,
            array_metadata.runtime_configuration.concurrency,
        )

        return ArrayValueHandle(out)

    async def _read_chunk(
        self,
        shard_dict: Mapping[ChunkCoords, Optional[BytesLike]],
        chunk_coords: ChunkCoords,
        chunk_selection: SliceSelection,
        out_selection: SliceSelection,
        array_metadata: "CoreArrayMetadata",
        out: np.ndarray,
    ):
        chunk_value = shard_dict.get(chunk_coords, None)
        chunk_array = await self._decode_chunk(chunk_value, array_metadata)
        if chunk_array is not None:
            tmp = chunk_array[chunk_selection]
            out[out_selection] = tmp
        else:
            out[out_selection] = array_metadata.fill_value

    async def _decode_chunk(
        self,
        value: Optional[BytesLike],
        array_metadata: "CoreArrayMetadata",
    ) -> Optional[np.ndarray]:
        if value is None:
            return None

        value_handle: ValueHandle = BufferValueHandle(value)

        from zarrita.array import CoreArrayMetadata

        # rewriting the metadata to scope it to the shard
        core_metadata = CoreArrayMetadata(
            shape=array_metadata.chunk_shape,
            chunk_shape=self.configuration.chunk_shape,
            data_type=array_metadata.data_type,
            fill_value=array_metadata.fill_value,
            runtime_configuration=array_metadata.runtime_configuration,
        )

        # applying codecs in revers e order
        for codec in self.codecs[::-1]:
            value_handle = await codec.decode(value_handle, core_metadata)

        chunk_array = await value_handle.toarray()
        if chunk_array is None:
            return None

        # ensure correct dtype
        if chunk_array.dtype.name != array_metadata.data_type.name:
            chunk_array = chunk_array.view(np.dtype(array_metadata.data_type.name))

        # ensure correct chunk shape
        if chunk_array.shape != self.configuration.chunk_shape:
            chunk_array = chunk_array.reshape(self.configuration.chunk_shape)

        return chunk_array

    async def encode(
        self,
        value: ValueHandle,
        array_metadata: "CoreArrayMetadata",
    ) -> ValueHandle:
        # print("encode")
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

        async def _write_chunk(
            shard_array: np.ndarray,
            chunk_coords: ChunkCoords,
            chunk_selection: SliceSelection,
            out_selection: SliceSelection,
        ) -> Tuple[ChunkCoords, Optional[BytesLike]]:
            if is_total_slice(chunk_selection, chunk_shape):
                chunk_array = shard_array[out_selection]
            else:
                # handling writing partial chunks
                chunk_array = np.empty(
                    chunk_shape,
                    dtype=array_metadata.dtype,
                    order=array_metadata.runtime_configuration.order,
                )
                chunk_array.fill(array_metadata.fill_value)
                chunk_array[chunk_selection] = shard_array[out_selection]
            if not np.array_equiv(chunk_array, array_metadata.fill_value):
                return (
                    chunk_coords,
                    await self._encode_chunk(chunk_array, array_metadata),
                )
            return (chunk_coords, None)

        # assembling and encoding chunks within the shard
        encoded_chunks: List[
            Tuple[ChunkCoords, Optional[BytesLike]]
        ] = await concurrent_map(
            [
                (shard_array, chunk_coords, chunk_selection, out_selection)
                for chunk_coords, chunk_selection, out_selection in indexer
            ],
            _write_chunk,
            array_metadata.runtime_configuration.concurrency,
        )
        shard_builder = _ShardBuilder.create_empty(chunks_per_shard)
        for chunk_coords, chunk_bytes in encoded_chunks:
            if chunk_bytes is not None:
                shard_builder.append(chunk_coords, chunk_bytes)

        if shard_builder.index.is_all_empty():
            return NoneValueHandle()
        return BufferValueHandle(shard_builder.finalize())

    async def encode_partial(
        self,
        old_value_handle: ValueHandle,
        shard_array: np.ndarray,
        selection: SliceSelection,
        array_metadata: "CoreArrayMetadata",
    ) -> ValueHandle:
        # print("encode_partial")
        shard_shape = array_metadata.chunk_shape
        chunk_shape = self.configuration.chunk_shape
        chunks_per_shard = tuple(s // c for s, c in zip(shard_shape, chunk_shape))

        old_shard_dict = await self._load_full_shard(old_value_handle, chunks_per_shard)
        new_shard_builder = _ShardBuilder.create_empty(chunks_per_shard)
        tombstones: Set[ChunkCoords] = set()

        indexer = list(
            BasicIndexer(
                selection,
                shape=shard_shape,
                chunk_shape=chunk_shape,
            )
        )

        async def _write_chunk(
            chunk_coords: ChunkCoords,
            chunk_selection: SliceSelection,
            out_selection: SliceSelection,
        ) -> Tuple[ChunkCoords, Optional[BytesLike]]:
            chunk_array = None
            if is_total_slice(chunk_selection, chunk_shape):
                chunk_array = shard_array[out_selection]
            else:
                # handling writing partial chunks
                # read chunk first
                tmp = await self._decode_chunk(
                    old_shard_dict.get(chunk_coords, None),
                    array_metadata,
                )
                # merge new value
                if tmp is None:
                    chunk_array = np.empty(
                        chunk_shape,
                        dtype=array_metadata.dtype,
                        order=array_metadata.runtime_configuration.order,
                    )
                    chunk_array.fill(array_metadata.fill_value)
                else:
                    chunk_array = tmp.copy(
                        order=array_metadata.runtime_configuration.order
                    )  # make a writable copy
                chunk_array[chunk_selection] = shard_array[out_selection]

            if not np.array_equiv(chunk_array, array_metadata.fill_value):
                return (
                    chunk_coords,
                    await self._encode_chunk(chunk_array, array_metadata),
                )
            else:
                return (chunk_coords, None)

        encoded_chunks: List[
            Tuple[ChunkCoords, Optional[BytesLike]]
        ] = await concurrent_map(
            [
                (chunk_coords, chunk_selection, out_selection)
                for chunk_coords, chunk_selection, out_selection in indexer
            ],
            _write_chunk,
            array_metadata.runtime_configuration.concurrency,
        )

        for chunk_coords, chunk_bytes in encoded_chunks:
            if chunk_bytes is not None:
                new_shard_builder.append(chunk_coords, chunk_bytes)
            else:
                tombstones.add(chunk_coords)

        shard_builder = _ShardBuilder.merge_with_morton_order(
            chunks_per_shard, tombstones, new_shard_builder, old_shard_dict
        )

        if shard_builder.index.is_all_empty():
            return NoneValueHandle()
        return BufferValueHandle(shard_builder.finalize())

    async def _encode_chunk(
        self,
        chunk: np.ndarray,
        array_metadata: "CoreArrayMetadata",
    ) -> BytesLike:
        from zarrita.array import CoreArrayMetadata

        # rewriting the metadata to scope it to the shard
        CoreArrayMetadata(
            shape=array_metadata.chunk_shape,
            chunk_shape=self.configuration.chunk_shape,
            data_type=array_metadata.data_type,
            fill_value=array_metadata.fill_value,
            runtime_configuration=array_metadata.runtime_configuration,
        )

        encoded_chunk_value: ValueHandle = ArrayValueHandle(chunk)
        for codec in self.codecs:
            encoded_chunk_value = await codec.encode(
                encoded_chunk_value,
                array_metadata,
            )

        encoded_chunk_bytes = await encoded_chunk_value.tobytes()
        assert encoded_chunk_bytes is not None
        return encoded_chunk_bytes

    def _is_total_shard(
        self, all_chunk_coords: Set[ChunkCoords], chunks_per_shard: ChunkCoords
    ) -> bool:
        return len(all_chunk_coords) == product(chunks_per_shard) and all(
            chunk_coords in all_chunk_coords
            for chunk_coords in c_order_iter(chunks_per_shard)
        )

    async def _load_shard_index(
        self, value_handle: ValueHandle, chunks_per_shard: ChunkCoords
    ) -> _ShardIndex:
        # print("load_shard_index")
        index_bytes = await value_handle[
            -_ShardIndex.index_byte_length(chunks_per_shard) :
        ].tobytes()
        assert isinstance(index_bytes, bytes), index_bytes
        if index_bytes is not None:
            return _ShardIndex.from_bytes(index_bytes, chunks_per_shard)
        else:
            return _ShardIndex.create_empty(chunks_per_shard)

    async def _load_full_shard(
        self, value_handle: ValueHandle, chunks_per_shard: ChunkCoords
    ) -> _ShardProxy:
        # print("load_full_shard")
        shard_bytes = await value_handle.tobytes()
        if shard_bytes:
            assert isinstance(shard_bytes, bytes)
            return _ShardProxy.from_bytes(shard_bytes, chunks_per_shard)
        return _ShardProxy.create_empty(chunks_per_shard)
