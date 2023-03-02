from collections import defaultdict
from dataclasses import dataclass
import functools
import itertools
from typing import (
    Dict,
    Final,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Union,
    TYPE_CHECKING,
)

import numpy as np

from zarrita.store import Store

if TYPE_CHECKING:
    from . import ChunkKeyEncodingMetadata

MAX_UINT_64 = 2**64 - 1


@dataclass
class ShardingStorageTransformerConfigurationMetadata:
    chunks_per_shard: Tuple[int, ...]

    @property
    def num_per_chunks(self) -> int:
        return functools.reduce(lambda x, y: x * y, self.chunks_per_shard, 1)


@dataclass
class ShardingStorageTransformerMetadata:
    configuration: ShardingStorageTransformerConfigurationMetadata
    name: Final = "sharding"


def c_order_iter(chunk_shape: Tuple[int, ...]) -> Iterator[Tuple[int, ...]]:
    return itertools.product(*(range(x) for x in chunk_shape))


class _ShardIndex(NamedTuple):
    configuration: ShardingStorageTransformerConfigurationMetadata
    # dtype uint64, shape (chunks_per_shard_0, chunks_per_shard_1, ..., 2)
    offsets_and_lengths: np.ndarray

    def __localize_chunk__(self, chunk: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple(
            chunk_i % shard_i
            for chunk_i, shard_i in zip(chunk, self.configuration.chunks_per_shard)
        )

    def is_all_empty(self) -> bool:
        return np.array_equiv(self.offsets_and_lengths, MAX_UINT_64)

    def get_chunk_slice(self, chunk: Tuple[int, ...]) -> Optional[slice]:
        localized_chunk = self.__localize_chunk__(chunk)
        chunk_start, chunk_len = self.offsets_and_lengths[localized_chunk]
        if (chunk_start, chunk_len) == (MAX_UINT_64, MAX_UINT_64):
            return None
        else:
            return slice(int(chunk_start), int(chunk_start + chunk_len))

    def set_chunk_slice(
        self, chunk: Tuple[int, ...], chunk_slice: Optional[slice]
    ) -> None:
        localized_chunk = self.__localize_chunk__(chunk)
        if chunk_slice is None:
            self.offsets_and_lengths[localized_chunk] = (MAX_UINT_64, MAX_UINT_64)
        else:
            self.offsets_and_lengths[localized_chunk] = (
                chunk_slice.start,
                chunk_slice.stop - chunk_slice.start,
            )

    def to_bytes(self) -> bytes:
        return self.offsets_and_lengths.tobytes(order="C")

    @property
    def byte_length(self) -> int:
        return self.index_byte_length(self.configuration)

    @classmethod
    def from_bytes(
        cls,
        buffer: Union[bytes, bytearray],
        configuration: ShardingStorageTransformerConfigurationMetadata,
    ) -> "_ShardIndex":
        try:
            return cls(
                configuration=configuration,
                offsets_and_lengths=np.frombuffer(
                    bytearray(buffer), dtype="<u8"
                ).reshape(*configuration.chunks_per_shard, 2, order="C"),
            )
        except ValueError as e:  # pragma: no cover
            raise RuntimeError from e

    @classmethod
    def create_empty(
        cls, configuration: ShardingStorageTransformerConfigurationMetadata
    ) -> "_ShardIndex":
        # reserving 2*64bit per chunk for offset and length:
        return cls.from_bytes(
            MAX_UINT_64.to_bytes(8, byteorder="little")
            * (2 * configuration.num_per_chunks),
            configuration=configuration,
        )

    @classmethod
    def index_byte_length(
        cls, configuration: ShardingStorageTransformerConfigurationMetadata
    ) -> int:
        return 16 * configuration.num_per_chunks


class ShardingStorageTransformer(Store):
    configuration: ShardingStorageTransformerConfigurationMetadata
    chunk_key_encoding: "ChunkKeyEncodingMetadata"
    inner_store: Store
    path: str

    def __init__(
        self,
        configuration: ShardingStorageTransformerConfigurationMetadata,
        chunk_key_encoding: "ChunkKeyEncodingMetadata",
        inner_store: Store,
        path: str,
    ):
        self.configuration = configuration
        self.chunk_key_encoding = chunk_key_encoding
        self.inner_store = inner_store
        self.path = path

    def multi_get(
        self, keys: List[Tuple[str, Optional[Tuple[int, int]]]]
    ) -> List[Optional[bytes]]:
        class _ChunkInfo(NamedTuple):
            i: int
            chunk_coords: Tuple[int, ...]
            key: str

        output = [None for _ in keys]

        shard_to_chunk_mapping: defaultdict[
            Tuple[int, ...], List[_ChunkInfo]
        ] = defaultdict(list)
        for i, (key, _) in enumerate(keys):
            global_chunk_coords = self.chunk_key_encoding.decode_chunk_key(
                key[len(self.path) + 1 :]
            )
            chunk_coords = tuple(
                c % s
                for c, s in zip(
                    global_chunk_coords, self.configuration.chunks_per_shard
                )
            )
            shard_coords = tuple(
                c // s
                for c, s in zip(
                    global_chunk_coords, self.configuration.chunks_per_shard
                )
            )
            shard_to_chunk_mapping[shard_coords].append(
                _ChunkInfo(i=i, chunk_coords=chunk_coords, key=key)
            )

        for shard_coords, chunk_infos in shard_to_chunk_mapping.items():
            shard_key = (
                f"{self.path}/{self.chunk_key_encoding.encode_chunk_key(shard_coords)}"
            )

            shard_index = self._load_shard_index(shard_key)
            chunks_to_fetch = [
                (chunk_info, shard_index.get_chunk_slice(chunk_info.chunk_coords))
                for chunk_info in chunk_infos
            ]
            chunks_to_fetch = [
                chunk_info
                for chunk_info in chunks_to_fetch
                if chunk_info[1] is not None
            ]

            for (chunk_info, _), chunk_value in zip(
                chunks_to_fetch,
                self.inner_store.multi_get(
                    [
                        (shard_key, (chunk_slice.start, chunk_slice.stop))
                        for _, chunk_slice in chunks_to_fetch
                    ]
                ),
            ):
                output[chunk_info.i] = chunk_value
        return output

    def multi_set(
        self, key_values: List[Tuple[str, bytes, Optional[Tuple[int, int]]]]
    ) -> None:
        class _ChunkInfo(NamedTuple):
            i: int
            chunk_coords: Tuple[int, ...]
            key: str
            chunk_value: bytes

        shard_to_chunk_mapping: defaultdict[
            Tuple[int, ...], List[_ChunkInfo]
        ] = defaultdict(list)
        for i, (key, chunk_value, _) in enumerate(key_values):
            global_chunk_coords = self.chunk_key_encoding.decode_chunk_key(
                key[len(self.path) + 1 :]
            )
            chunk_coords = tuple(
                c % s
                for c, s in zip(
                    global_chunk_coords, self.configuration.chunks_per_shard
                )
            )
            shard_coords = tuple(
                c // s
                for c, s in zip(
                    global_chunk_coords, self.configuration.chunks_per_shard
                )
            )
            shard_to_chunk_mapping[shard_coords].append(
                _ChunkInfo(
                    i=i, chunk_coords=chunk_coords, key=key, chunk_value=chunk_value
                )
            )

        for shard_coords, chunk_infos in shard_to_chunk_mapping.items():
            shard_key = (
                f"{self.path}/{self.chunk_key_encoding.encode_chunk_key(shard_coords)}"
            )
            if self._is_total_shard(
                set(chunk_info.chunk_coords for chunk_info in chunk_infos)
            ):
                shard_bytes = self._build_shard(
                    {
                        chunk_info.chunk_coords: chunk_info.chunk_value
                        for chunk_info in chunk_infos
                    }
                )
                self.inner_store.multi_set([(shard_key, shard_bytes, None)])
            else:
                shard_dict = self._load_full_shard(shard_key)
                for chunk_info in chunk_infos:
                    shard_dict[chunk_info.chunk_coords] = chunk_info.chunk_value
                shard_bytes = self._build_shard(shard_dict)
                self.inner_store.multi_set([(shard_key, shard_bytes, None)])

    def _is_total_shard(self, all_chunk_coords: Set[Tuple[int, ...]]) -> bool:
        return len(all_chunk_coords) == self.configuration.num_per_chunks and all(
            chunk_coords in all_chunk_coords
            for chunk_coords in c_order_iter(self.configuration.chunks_per_shard)
        )

    def _merge_ranges(self, key_ranges):
        merged_key_ranges = []
        reverse_lookup = []
        mergable_ranges_per_key = {}
        for i, (key, _range) in enumerate(key_ranges):
            if self._is_data_key(key):
                ranges = mergable_ranges_per_key.setdefault(key, [])
                ranges.append((i, _range))
                reverse_lookup.append(None)  # placeholder
            else:
                reverse_lookup.append((len(merged_key_ranges), slice(0, None)))
                merged_key_ranges.append((key, _range))
        for key, indexed_ranges in mergable_ranges_per_key.items():
            current_start, current_length = (None, None)
            for i, (range_start, range_length) in sorted(
                indexed_ranges, key=lambda x: x[1]
            ):
                # range_start and range_length are positive integers
                if current_start is None:
                    current_start = range_start
                    current_length = range_length
                if range_start > current_start + current_length:
                    # merging not possible, write out previous range and reset:
                    merged_key_ranges.append((key, (current_start, current_length)))
                    current_start = range_start
                    current_length = range_length
                else:
                    # merge with previous ranges
                    current_length = max(
                        current_length, range_start + range_length - current_start
                    )
                relative_start = range_start - current_start
                relative_end = relative_start + range_length
                reverse_lookup[i] = (
                    len(merged_key_ranges),
                    slice(relative_start, relative_end),
                )
            # write out last range
            merged_key_ranges.append((key, (current_start, current_length)))
        return merged_key_ranges, reverse_lookup

    def _load_shard_index(self, shard_key: str) -> _ShardIndex:
        index_bytes = self.inner_store.multi_get(
            [
                (
                    shard_key,
                    (-_ShardIndex.index_byte_length(self.configuration), None),
                )
            ]
        )[0]
        if index_bytes is not None:
            return _ShardIndex.from_bytes(index_bytes, self.configuration)
        else:
            return _ShardIndex.create_empty(self.configuration)

    def _build_shard(self, shard_dict: Dict[Tuple[int, ...], Optional[bytes]]) -> bytes:
        shard_index = _ShardIndex.create_empty(self.configuration)
        shard_bytes = bytearray(
            sum(
                len(chunk_value)
                for chunk_value in shard_dict.values()
                if chunk_value is not None
            )
            + shard_index.byte_length
        )
        byte_offset = 0
        for chunk_coords in c_order_iter(self.configuration.chunks_per_shard):
            chunk_value = shard_dict.get(chunk_coords, None)
            if chunk_value is not None:
                byte_offset_end = byte_offset + len(chunk_value)
                shard_bytes[byte_offset : byte_offset + len(chunk_value)] = chunk_value
                shard_index.set_chunk_slice(
                    chunk_coords, slice(byte_offset, byte_offset_end)
                )
                byte_offset = byte_offset_end
        shard_bytes[-shard_index.byte_length :] = shard_index.to_bytes()
        return bytes(shard_bytes)

    def _load_full_shard(
        self, shard_key: str
    ) -> Dict[Tuple[int, ...], Optional[bytes]]:
        shard_bytes = self.inner_store.multi_get([(shard_key, None)])[0]
        if shard_bytes:
            index_bytes = shard_bytes[
                -_ShardIndex.index_byte_length(self.configuration) :
            ]
            shard_index = _ShardIndex.from_bytes(index_bytes, self.configuration)

            shard_dict = {}
            for chunk_coords in c_order_iter(self.configuration.chunks_per_shard):
                chunk_byte_slice = shard_index.get_chunk_slice(chunk_coords)
                if chunk_byte_slice is not None:
                    shard_dict[chunk_coords] = shard_bytes[chunk_byte_slice]
                else:
                    shard_dict[chunk_coords] = None

            return shard_dict
        else:
            return {
                chunk_coords: None
                for chunk_coords in c_order_iter(self.configuration.chunks_per_shard)
            }
