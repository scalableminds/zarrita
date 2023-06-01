import asyncio
import json
from enum import Enum
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

import numpy as np
from attr import asdict, field, frozen

from zarrita.codecs import CodecMetadata
from zarrita.common import ZARR_JSON, is_total_slice, make_cattr
from zarrita.indexing import BasicIndexer
from zarrita.sharding import ChunkCoords
from zarrita.store import Store
from zarrita.value_handle import (
    ArrayValueHandle,
    FileValueHandle,
    NoneValueHandle,
    ValueHandle,
)


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
class CoreArrayMetadata:
    shape: Tuple[int, ...]
    chunk_shape: Tuple[int, ...]
    data_type: DataType
    fill_value: Any
    order: Literal["C", "F"]

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


@frozen
class ArrayRuntimeConfiguration:
    order: Literal["C", "F"] = "C"


@frozen
class Array:
    metadata: ArrayMetadata
    store: "Store"
    path: str
    runtime_configuration: ArrayRuntimeConfiguration

    @classmethod
    async def create_async(
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
        runtime_configuration: Optional[ArrayRuntimeConfiguration] = None,
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
            fill_value=0 if fill_value is None else fill_value,
            codecs=list(codecs) if codecs else [],
            dimension_names=tuple(dimension_names) if dimension_names else None,
            attributes=attributes or {},
        )
        array = cls(
            metadata=metadata,
            store=store,
            path=path,
            runtime_configuration=runtime_configuration
            if runtime_configuration
            else ArrayRuntimeConfiguration(),
        )
        await array._save_metadata()
        return array

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
        runtime_configuration: Optional[ArrayRuntimeConfiguration] = None,
    ) -> "Array":
        return asyncio.get_event_loop().run_until_complete(
            cls.create_async(
                store,
                path,
                shape=shape,
                dtype=dtype,
                chunk_shape=chunk_shape,
                fill_value=fill_value,
                chunk_key_encoding=chunk_key_encoding,
                codecs=codecs,
                dimension_names=dimension_names,
                attributes=attributes,
                runtime_configuration=runtime_configuration,
            )
        )

    @classmethod
    async def open_async(
        cls,
        store: "Store",
        path: str,
        runtime_configuration: Optional[ArrayRuntimeConfiguration] = None,
    ) -> "Array":
        zarr_json_bytes = await store.get_async(f"{path}/{ZARR_JSON}")
        assert zarr_json_bytes is not None
        return cls.from_json(
            store,
            path,
            json.loads(zarr_json_bytes),
            runtime_configuration=runtime_configuration
            if runtime_configuration
            else ArrayRuntimeConfiguration(),
        )

    @classmethod
    def open(cls, store: "Store", path: str) -> "Array":
        return asyncio.get_event_loop().run_until_complete(cls.open_async(store, path))

    @classmethod
    def from_json(
        cls,
        store: Store,
        path: str,
        zarr_json: Any,
        runtime_configuration: ArrayRuntimeConfiguration,
    ) -> "Array":
        return cls(
            metadata=make_cattr().structure(zarr_json, ArrayMetadata),
            store=store,
            path=path,
            runtime_configuration=runtime_configuration,
        )

    async def _save_metadata(self) -> None:
        def convert(o):
            if isinstance(o, DataType):
                return o.name
            raise TypeError

        await self.store.set_async(
            f"{self.path}/{ZARR_JSON}",
            json.dumps(asdict(self.metadata), default=convert).encode(),
        )

    @property
    def ndim(self) -> int:
        return len(self.metadata.shape)

    @property
    def _core_metadata(self) -> CoreArrayMetadata:
        return CoreArrayMetadata(
            shape=self.metadata.shape,
            chunk_shape=self.metadata.chunk_grid.configuration.chunk_shape,
            data_type=self.metadata.data_type,
            fill_value=self.metadata.fill_value,
            order=self.runtime_configuration.order,
        )

    def __getitem__(self, selection: Union[slice, Tuple[slice, ...]]):
        return asyncio.get_event_loop().run_until_complete(self.get_async(selection))

    async def _fetch_chunk(self, chunk_coords, chunk_selection, out_selection, out):
        chunk_key_encoding = self.metadata.chunk_key_encoding
        chunk_key = f"{self.path}/{chunk_key_encoding.encode_chunk_key(chunk_coords)}"
        value_handle = FileValueHandle(self.store, chunk_key)
        if (
            len(self.metadata.codecs) == 1
            and self.metadata.codecs[0].name == "sharding_indexed"
        ):
            value_handle = await self.metadata.codecs[0].decode_partial(
                value_handle, chunk_selection, self._core_metadata
            )
            chunk_array = await value_handle.toarray()
            if chunk_array is not None:
                out[out_selection] = chunk_array
            else:
                out[out_selection] = self.metadata.fill_value
        else:
            chunk_array = await self._decode_chunk(value_handle, chunk_selection)

            if chunk_array is not None:
                tmp = chunk_array[chunk_selection]
                out[out_selection] = tmp
            else:
                out[out_selection] = self.metadata.fill_value

    async def get_async(self, selection: Union[slice, Tuple[slice, ...]]):
        indexer = BasicIndexer(
            selection,
            shape=self.metadata.shape,
            chunk_shape=self.metadata.chunk_grid.configuration.chunk_shape,
        )

        # setup output array
        out = np.zeros(
            indexer.shape,
            dtype=self.metadata.dtype,
            order=self.runtime_configuration.order,
        )

        # reading chunks and decoding them
        await asyncio.gather(
            *[
                self._fetch_chunk(chunk_coords, chunk_selection, out_selection, out)
                for chunk_coords, chunk_selection, out_selection in indexer
            ]
        )

        if out.shape:
            return out
        else:
            return out[()]

    async def _decode_chunk(
        self, value_handle: ValueHandle, selection: Tuple[slice, ...]
    ) -> Optional[np.ndarray]:
        if isinstance(value_handle, NoneValueHandle):
            return None

        # apply codecs in reverse order
        for codec_metadata in self.metadata.codecs[::-1]:
            value_handle = await codec_metadata.decode(
                value_handle, self._core_metadata
            )

        chunk_array = await value_handle.toarray()
        if chunk_array is None:
            return None

        # ensure correct dtype
        if str(chunk_array.dtype) != self.metadata.data_type.name:
            chunk_array = chunk_array.view(self.metadata.dtype)

        # ensure correct chunk shape
        if chunk_array.shape != self.metadata.chunk_grid.configuration.chunk_shape:
            chunk_array = chunk_array.reshape(
                self.metadata.chunk_grid.configuration.chunk_shape,
                order=self.runtime_configuration.order,
            )

        return chunk_array

    def __setitem__(
        self, selection: Union[slice, Tuple[slice, ...]], value: np.ndarray
    ) -> None:
        asyncio.get_event_loop().run_until_complete(self.set_async(selection, value))

    async def set_async(
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
            if value.dtype != self.metadata.dtype:
                value = value.astype(self.metadata.dtype, order="K")

        # merging with existing data and encoding chunks
        for chunk_coords, chunk_selection, out_selection in indexer:
            chunk_key_encoding = self.metadata.chunk_key_encoding
            chunk_key = (
                f"{self.path}/{chunk_key_encoding.encode_chunk_key(chunk_coords)}"
            )
            value_handle = FileValueHandle(self.store, chunk_key)

            if is_total_slice(chunk_selection, chunk_shape):
                # write entire chunks
                if sel_shape == ():
                    chunk_array = value
                elif np.isscalar(value):
                    chunk_array = np.empty(
                        chunk_shape,
                        dtype=self.metadata.dtype,
                        order=self.runtime_configuration.order,
                    )
                    chunk_array.fill(value)
                else:
                    chunk_array = value[out_selection]
                await self._write_chunk(value_handle, chunk_array)

            elif (
                len(self.metadata.codecs) == 1
                and self.metadata.codecs[0].name == "sharding_indexed"
            ):
                sharding_codec = self.metadata.codecs[0]
                chunk_value = await sharding_codec.encode_partial(
                    value_handle,
                    value[out_selection],
                    chunk_selection,
                    self._core_metadata,
                )
                await value_handle.set_async(chunk_value)
            else:
                # writing partial chunks
                # read chunk first
                tmp = await self._decode_chunk(
                    value_handle,
                    tuple(slice(0, c) for c in chunk_shape),
                )

                # merge new value
                if tmp is None:
                    chunk_array = np.empty(
                        chunk_shape,
                        dtype=self.metadata.dtype,
                        order=self.runtime_configuration.order,
                    )
                    chunk_array.fill(self.metadata.fill_value)
                else:
                    chunk_array = tmp.copy(
                        order=self.runtime_configuration.order,
                    )  # make a writable copy
                chunk_array[chunk_selection] = value[out_selection]

                await self._write_chunk(value_handle, chunk_array)

    async def _write_chunk(self, value_handle: ValueHandle, chunk_array: np.ndarray):
        chunk_value: ValueHandle
        if np.all(chunk_array == self.metadata.fill_value):
            # chunks that only contain fill_value will be removed
            chunk_value = NoneValueHandle()
        else:
            chunk_value = await self._encode_chunk(chunk_array)

        # write out chunk
        await value_handle.set_async(chunk_value)

    async def _encode_chunk(self, chunk_array: np.ndarray):
        encoded_chunk_value: ValueHandle = ArrayValueHandle(chunk_array)
        for codec in self.metadata.codecs:
            encoded_chunk_value = await codec.encode(
                encoded_chunk_value,
                self._core_metadata,
            )

        return encoded_chunk_value

    def __repr__(self):
        path = self.path
        return f"<Array {path}>"
