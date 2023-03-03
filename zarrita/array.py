from enum import Enum
import json
from typing import Any, Dict, Final, List, Literal, Optional, Tuple, Union
from attr import asdict, define, field
import numpy as np

from zarrita.codecs import CodecMetadata
from zarrita.common import ZARR_JSON, _is_total_slice, get_order
from zarrita.indexing import BasicIndexer
from zarrita.store import ArrayHandle, FileHandle, Store, ValueHandle


class DataType(Enum):
    bool = "bool"
    int8 = "|i1"
    int16 = "<i2"
    int32 = "<i4"
    int64 = "<i8"
    uint8 = "|u1"
    uint16 = "<u2"
    uint32 = "<u4"
    uint64 = "<u8"
    float32 = "<f4"
    float64 = "<f8"


@define
class RegularChunkGridConfigurationMetadata:
    chunk_shape: Tuple[int, ...]


@define
class RegularChunkGridMetadata:
    configuration: RegularChunkGridConfigurationMetadata
    name: Final = "regular"


@define
class DefaultChunkKeyEncodingConfigurationMetadata:
    separator: Literal[".", "/"] = "/"


@define
class DefaultChunkKeyEncodingMetadata:
    configuration: DefaultChunkKeyEncodingConfigurationMetadata = (
        DefaultChunkKeyEncodingConfigurationMetadata()
    )
    name: Final = "default"

    def decode_chunk_key(self, chunk_key: str) -> Tuple[int, ...]:
        if chunk_key == "c":
            return ()
        return tuple(map(int, chunk_key[1:].split(self.configuration.separator)))

    def encode_chunk_key(self, chunk_coords: Tuple[int, ...]) -> str:
        chunk_identifier = self.configuration.separator.join(map(str, chunk_coords))
        return f"c{'0' if chunk_identifier == '' else chunk_identifier}"


@define
class V2ChunkKeyEncodingConfigurationMetadata:
    separator: Literal[".", "/"] = "."


@define
class V2ChunkKeyEncodingMetadata:
    configuration: V2ChunkKeyEncodingConfigurationMetadata = (
        V2ChunkKeyEncodingConfigurationMetadata()
    )
    name: Final = "v2"

    def decode_chunk_key(self, chunk_key: str) -> Tuple[int, ...]:
        return tuple(map(int, chunk_key.split(self.configuration.separator)))

    def encode_chunk_key(self, chunk_coords: Tuple[int, ...]) -> str:
        chunk_identifier = self.configuration.separator.join(map(str, chunk_coords))
        return "0" if chunk_identifier == "" else chunk_identifier


ChunkKeyEncodingMetadata = Union[
    DefaultChunkKeyEncodingMetadata, V2ChunkKeyEncodingMetadata
]


@define
class CoreArrayMetadata:
    shape: Tuple[int, ...]
    chunk_shape: Tuple[int, ...]
    data_type: DataType
    fill_value: Any


@define
class ArrayMetadata:
    shape: Tuple[int, ...]
    data_type: DataType
    chunk_grid: RegularChunkGridMetadata
    chunk_key_encoding: ChunkKeyEncodingMetadata
    fill_value: Any
    attributes: Dict[str, Any] = field(factory=dict)
    codecs: List[CodecMetadata] = field(factory=list)
    dimension_names: Optional[List[str]] = None
    zarr_format: Final = 3
    node_type: Final = "array"

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.data_type.value)


class Array:
    metadata: ArrayMetadata
    store: "Store"
    path: str

    @classmethod
    def create(
        cls,
        store: "Store",
        path: str,
        *,
        shape: Tuple[int, ...],
        dtype: np.dtype,
        chunk_shape: Tuple[int, ...],
        fill_value: Optional[Any] = None,
        chunk_key_encoding: Union[
            Tuple[Literal["default"], Literal[".", "/"]],
            Tuple[Literal["v2"], Literal[".", "/"]],
        ] = ("default", "/"),
        codecs: Optional[List[CodecMetadata]] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> "Array":
        metadata = ArrayMetadata(
            shape=shape,
            data_type=DataType(dtype.str),
            chunk_grid=RegularChunkGridMetadata(
                configuration=RegularChunkGridConfigurationMetadata(
                    chunk_shape=chunk_shape
                )
            ),
            chunk_key_encoding=(
                V2ChunkKeyEncodingMetadata(
                    configuration=V2ChunkKeyEncodingConfigurationMetadata(
                        separator=chunk_key_encoding[1]
                    )
                )
                if chunk_key_encoding[0] == "v2"
                else DefaultChunkKeyEncodingMetadata(
                    configuration=DefaultChunkKeyEncodingConfigurationMetadata(
                        separator=chunk_key_encoding[1]
                    )
                )
            ),
            fill_value=fill_value,
            codecs=codecs or [],
            attributes=attributes or {},
        )
        array = cls()
        array.metadata = metadata
        array.store = store
        array.path = path
        array._save_metadata()
        return array

    @classmethod
    def open(cls, store: "Store", path: str) -> "Array":
        zarr_json_bytes = store.get(f"{path}/{ZARR_JSON}")
        assert zarr_json_bytes is not None
        return cls.from_json(store, path, json.loads(zarr_json_bytes))

    @classmethod
    def from_json(cls, store: Store, path: str, zarr_json: Any) -> "Array":
        metadata = ArrayMetadata(**zarr_json)
        array = cls()
        array.metadata = metadata
        array.store = store
        array.path = path
        return array

    def _save_metadata(self) -> None:
        def convert(o):
            if isinstance(o, DataType):
                return o.name
            raise TypeError

        self.store.set(
            f"{self.path}/{ZARR_JSON}",
            json.dumps(asdict(self.metadata), default=convert).encode(),
        )

    @property
    def ndim(self) -> int:
        return len(self.metadata.shape)

    def __getitem__(self, selection):
        indexer = BasicIndexer(
            selection,
            shape=self.metadata.shape,
            chunk_shape=self.metadata.chunk_grid.configuration.chunk_shape,
        )

        # setup output array
        out = np.zeros(
            indexer.shape,
            dtype=self.metadata.dtype,
            order=get_order(self.metadata.codecs),
        )

        # iterate over chunks
        for chunk_coords, chunk_selection, out_selection in indexer:
            chunk_key = f"{self.path}/{self.metadata.chunk_key_encoding.encode_chunk_key(chunk_coords)}"
            value_handle = FileHandle(self.store, chunk_key)
            chunk = self._decode_chunk(value_handle, chunk_selection)
            if chunk is not None:
                tmp = chunk[chunk_selection]
                out[out_selection] = tmp
            elif self.metadata.fill_value is not None:
                out[out_selection] = self.metadata.fill_value

        if out.shape:
            return out
        else:
            return out[()]

    def _decode_chunk(
        self, value_handle: ValueHandle, selection: Tuple[slice, ...]
    ) -> Optional[np.ndarray]:
        core_metadata = CoreArrayMetadata(
            shape=self.metadata.shape,
            chunk_shape=self.metadata.chunk_grid.configuration.chunk_shape,
            data_type=self.metadata.data_type,
            fill_value=self.metadata.fill_value,
        )

        for codec_metadata in self.metadata.codecs[::-1]:
            value_handle = codec_metadata.decode(value_handle, selection, core_metadata)

        # view as numpy array with correct dtype
        chunk = value_handle.toarray()
        if chunk is None:
            return None

        if str(chunk.dtype) != self.metadata.data_type.name:
            chunk = chunk.view(self.metadata.dtype)

        # ensure correct chunk shape
        if chunk.shape != self.metadata.chunk_grid.configuration.chunk_shape:
            chunk = chunk.reshape(self.metadata.chunk_grid.configuration.chunk_shape)

        return chunk

    def __setitem__(self, selection, value):
        indexer = BasicIndexer(
            selection,
            shape=self.metadata.shape,
            chunk_shape=self.metadata.chunk_grid.configuration.chunk_shape,
        )

        sel_shape = indexer.shape

        # check value shape
        if sel_shape == ():
            # setting a single item
            assert np.isscalar(value)
        elif np.isscalar(value):
            # setting a scalar value
            pass
        else:
            if not hasattr(value, "shape"):
                value = np.asarray(value, self.dtype)
            assert value.shape == sel_shape

        for chunk_coords, chunk_selection, out_selection in indexer:
            assert _is_total_slice(
                chunk_selection, self.metadata.chunk_grid.configuration.chunk_shape
            )
            chunk_key = f"{self.path}/{self.metadata.chunk_key_encoding.encode_chunk_key(chunk_coords)}"
            value_handle = FileHandle(self.store, chunk_key)

            # extract data to store
            if sel_shape == ():
                chunk_value = value
            elif np.isscalar(value):
                chunk_value = np.empty(
                    self.metadata.chunk_grid.configuration.chunk_shape,
                    dtype=self.metadata.dtypes,
                    order="C",
                )
                chunk_value.fill(value)
            else:
                chunk_value = value[out_selection]
            chunk_value = self._encode_chunk(chunk_value, chunk_selection)
            value_handle[:] = chunk_value

    def _encode_chunk(self, chunk_value: np.ndarray, selection: Tuple[slice, ...]):
        core_metadata = CoreArrayMetadata(
            shape=self.metadata.shape,
            chunk_shape=self.metadata.chunk_grid.configuration.chunk_shape,
            data_type=self.metadata.data_type,
            fill_value=self.metadata.fill_value,
        )
        encoded_chunk_value: ValueHandle = ArrayHandle(chunk_value)
        for codec in self.metadata.codecs:
            encoded_chunk_value = codec.encode(
                encoded_chunk_value,
                selection,
                core_metadata,
            )

        return encoded_chunk_value

    def __repr__(self):
        path = self.path
        return f"<Array {path}>"
