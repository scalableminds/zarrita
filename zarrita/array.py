import json
from enum import Enum
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

import numpy as np
from attr import asdict, field, frozen

from zarrita.codecs import CodecMetadata
from zarrita.common import ZARR_JSON, get_order, is_total_slice, make_cattr
from zarrita.indexing import BasicIndexer
from zarrita.store import Store
from zarrita.value_handle import ArrayHandle, FileHandle, NoneHandle, ValueHandle


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


@frozen
class RegularChunkGridConfigurationMetadata:
    chunk_shape: Tuple[int, ...]


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

    def decode_chunk_key(self, chunk_key: str) -> Tuple[int, ...]:
        if chunk_key == "c":
            return ()
        return tuple(map(int, chunk_key[1:].split(self.configuration.separator)))

    def encode_chunk_key(self, chunk_coords: Tuple[int, ...]) -> str:
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

    def decode_chunk_key(self, chunk_key: str) -> Tuple[int, ...]:
        return tuple(map(int, chunk_key.split(self.configuration.separator)))

    def encode_chunk_key(self, chunk_coords: Tuple[int, ...]) -> str:
        chunk_identifier = self.configuration.separator.join(map(str, chunk_coords))
        return "0" if chunk_identifier == "" else chunk_identifier


ChunkKeyEncodingMetadata = Union[
    DefaultChunkKeyEncodingMetadata, V2ChunkKeyEncodingMetadata
]


@frozen
class CoreArrayMetadata:
    shape: Tuple[int, ...]
    chunk_shape: Tuple[int, ...]
    data_type: DataType
    fill_value: Any

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.data_type.value)


@frozen
class ArrayMetadata:
    shape: Tuple[int, ...]
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
        dtype: Union[str, np.dtype],
        chunk_shape: Tuple[int, ...],
        fill_value: Optional[Any] = None,
        chunk_key_encoding: Union[
            Tuple[Literal["default"], Literal[".", "/"]],
            Tuple[Literal["v2"], Literal[".", "/"]],
        ] = ("default", "/"),
        codecs: Optional[Iterable[CodecMetadata]] = None,
        dimension_names: Optional[Iterable[str]] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> "Array":
        data_type = (
            DataType[dtype]
            if isinstance(dtype, str)
            else DataType[dtype_to_data_type[dtype.str]]
        )

        metadata = ArrayMetadata(
            shape=shape,
            data_type=data_type,
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
            codecs=list(codecs) if codecs else [],
            dimension_names=tuple(dimension_names) if dimension_names else None,
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
        array = cls()
        array.metadata = make_cattr().structure(zarr_json, ArrayMetadata)
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

    def __getitem__(self, selection: Union[slice, Tuple[slice, ...]]):
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

        # reading chunks and decoding them
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
        if isinstance(value_handle, NoneHandle):
            return None

        core_metadata = CoreArrayMetadata(
            shape=self.metadata.shape,
            chunk_shape=self.metadata.chunk_grid.configuration.chunk_shape,
            data_type=self.metadata.data_type,
            fill_value=self.metadata.fill_value,
        )

        # apply codecs in reverse order
        for codec_metadata in self.metadata.codecs[::-1]:
            value_handle = codec_metadata.decode(value_handle, selection, core_metadata)

        chunk = value_handle.toarray()
        if chunk is None:
            return None

        # ensure correct dtype
        if str(chunk.dtype) != self.metadata.data_type.name:
            chunk = chunk.view(self.metadata.dtype)

        # ensure correct chunk shape
        if chunk.shape != self.metadata.chunk_grid.configuration.chunk_shape:
            chunk = chunk.reshape(self.metadata.chunk_grid.configuration.chunk_shape)

        return chunk

    def __setitem__(
        self, selection: Union[slice, Tuple[slice, ...]], value: np.ndarray
    ) -> None:
        chunk_shape = self.metadata.chunk_grid.configuration.chunk_shape
        indexer = BasicIndexer(
            selection,
            shape=self.metadata.shape,
            chunk_shape=chunk_shape,
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
                value = np.asarray(value, self.metadata.dtype)
            assert value.shape == sel_shape

        # merging with existing data and encoding chunks
        for chunk_coords, chunk_selection, out_selection in indexer:
            chunk_key = f"{self.path}/{self.metadata.chunk_key_encoding.encode_chunk_key(chunk_coords)}"
            value_handle = FileHandle(self.store, chunk_key)

            if is_total_slice(chunk_selection, chunk_shape):
                # write entire chunks
                if sel_shape == ():
                    chunk = value
                elif np.isscalar(value):
                    chunk = np.empty(
                        chunk_shape,
                        dtype=self.metadata.dtype,
                        order="C",
                    )
                    chunk.fill(value)
                else:
                    chunk = value[out_selection].astype(
                        self.metadata.dtype, order="C", copy=False
                    )
            else:
                # writing partial chunks
                # read chunk first
                tmp = self._decode_chunk(
                    value_handle,
                    tuple(slice(0, c) for c in chunk_shape),
                )

                # merge new value
                if tmp is None:
                    chunk = np.empty(
                        chunk_shape,
                        dtype=self.metadata.dtype,
                        order="C",
                    )
                    if self.metadata.fill_value:
                        chunk.fill(self.metadata.fill_value)
                else:
                    chunk = tmp.copy(order="K")  # make a writable copy
                chunk[chunk_selection] = value[out_selection]

            chunk_value: ValueHandle
            if self.metadata.fill_value and np.all(chunk == self.metadata.fill_value):
                # chunks that only contain fill_value will be removed
                chunk_value = NoneHandle()
            else:
                chunk_value = self._encode_chunk(
                    chunk,
                    tuple(slice(0, c) for c in chunk_shape),
                )

            # write out chunk
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
