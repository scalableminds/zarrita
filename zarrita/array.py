import json
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union
from warnings import warn

import attr
import numpy as np
from attr import asdict, frozen

from zarrita.array_v2 import ArrayV2
from zarrita.codecs import Codec, CodecMetadata
from zarrita.common import (
    ZARR_JSON,
    ChunkCoords,
    Selection,
    SliceSelection,
    concurrent_map,
    make_cattr,
)
from zarrita.indexing import BasicIndexer, all_chunk_coords, is_total_slice
from zarrita.metadata import (
    ArrayMetadata,
    CoreArrayMetadata,
    DataType,
    DefaultChunkKeyEncodingConfigurationMetadata,
    DefaultChunkKeyEncodingMetadata,
    RegularChunkGridConfigurationMetadata,
    RegularChunkGridMetadata,
    V2ChunkKeyEncodingConfigurationMetadata,
    V2ChunkKeyEncodingMetadata,
    dtype_to_data_type,
)
from zarrita.sharding import ShardingCodec
from zarrita.store import Store
from zarrita.sync import sync
from zarrita.value_handle import (
    ArrayValueHandle,
    FileValueHandle,
    NoneValueHandle,
    ValueHandle,
)


@frozen
class ArrayRuntimeConfiguration:
    order: Literal["C", "F"] = "C"
    concurrency: Optional[int] = None


def runtime_configuration(
    order: Literal["C", "F"], concurrency: Optional[int] = None
) -> ArrayRuntimeConfiguration:
    return ArrayRuntimeConfiguration(order=order, concurrency=concurrency)


@frozen
class _AsyncArrayProxy:
    array: "Array"

    def __getitem__(self, selection: Selection) -> "_AsyncArraySelectionProxy":
        return _AsyncArraySelectionProxy(self.array, selection)


@frozen
class _AsyncArraySelectionProxy:
    array: "Array"
    selection: Selection

    async def get(self) -> np.ndarray:
        return await self.array.get_async(self.selection)

    async def set(self, value: np.ndarray):
        return await self.array.set_async(self.selection, value)


def _json_convert(o):
    if isinstance(o, DataType):
        return o.name
    raise TypeError


@frozen
class Array:
    metadata: ArrayMetadata
    store: "Store"
    path: str
    runtime_configuration: ArrayRuntimeConfiguration
    codecs: List[Codec]

    @classmethod
    async def create_async(
        cls,
        store: "Store",
        path: str,
        *,
        shape: ChunkCoords,
        dtype: Union[str, np.dtype],
        chunk_shape: ChunkCoords,
        fill_value: Optional[Any] = None,
        chunk_key_encoding: Union[
            Tuple[Literal["default"], Literal[".", "/"]],
            Tuple[Literal["v2"], Literal[".", "/"]],
        ] = ("default", "/"),
        codecs: Optional[Iterable[CodecMetadata]] = None,
        dimension_names: Optional[Iterable[str]] = None,
        attributes: Optional[Dict[str, Any]] = None,
        runtime_configuration: Optional[ArrayRuntimeConfiguration] = None,
        exists_ok: bool = False,
    ) -> "Array":
        if not exists_ok:
            assert not await store.exists_async(f"{path}/{ZARR_JSON}")

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
            runtime_configuration=runtime_configuration or ArrayRuntimeConfiguration(),
            codecs=Codec.codecs_from_metadata(metadata.codecs),
        )

        await array._save_metadata()
        return array

    @classmethod
    def create(
        cls,
        store: "Store",
        path: str,
        *,
        shape: ChunkCoords,
        dtype: Union[str, np.dtype],
        chunk_shape: ChunkCoords,
        fill_value: Optional[Any] = None,
        chunk_key_encoding: Union[
            Tuple[Literal["default"], Literal[".", "/"]],
            Tuple[Literal["v2"], Literal[".", "/"]],
        ] = ("default", "/"),
        codecs: Optional[Iterable[CodecMetadata]] = None,
        dimension_names: Optional[Iterable[str]] = None,
        attributes: Optional[Dict[str, Any]] = None,
        runtime_configuration: Optional[ArrayRuntimeConfiguration] = None,
        exists_ok: bool = False,
    ) -> "Array":
        return sync(
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
                exists_ok=exists_ok,
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
            runtime_configuration=runtime_configuration or ArrayRuntimeConfiguration(),
        )

    @classmethod
    def open(
        cls,
        store: "Store",
        path: str,
        runtime_configuration: Optional[ArrayRuntimeConfiguration] = None,
    ) -> "Array":
        return sync(
            cls.open_async(store, path, runtime_configuration=runtime_configuration)
        )

    @classmethod
    def from_json(
        cls,
        store: Store,
        path: str,
        zarr_json: Any,
        runtime_configuration: ArrayRuntimeConfiguration,
    ) -> "Array":
        metadata = make_cattr().structure(zarr_json, ArrayMetadata)
        out = cls(
            metadata=metadata,
            store=store,
            path=path,
            runtime_configuration=runtime_configuration,
            codecs=Codec.codecs_from_metadata(metadata.codecs),
        )
        out._validate_metadata()
        return out

    @classmethod
    async def open_auto_async(
        cls,
        store: "Store",
        path: str,
        runtime_configuration: Optional[ArrayRuntimeConfiguration] = None,
    ) -> Union["Array", ArrayV2]:
        v3_metadata_bytes = await store.get_async(f"{path}/{ZARR_JSON}")
        if v3_metadata_bytes is not None:
            return cls.from_json(
                store,
                path,
                json.loads(v3_metadata_bytes),
                runtime_configuration=runtime_configuration
                or ArrayRuntimeConfiguration(),
            )
        return await ArrayV2.open_async(store, path)

    @classmethod
    async def open_auto(
        cls,
        store: "Store",
        path: str,
        runtime_configuration: Optional[ArrayRuntimeConfiguration] = None,
    ) -> Union["Array", ArrayV2]:
        return sync(cls.open_auto_async(store, path, runtime_configuration))

    async def _save_metadata(self) -> None:
        self._validate_metadata()

        await self.store.set_async(
            f"{self.path}/{ZARR_JSON}",
            json.dumps(asdict(self.metadata), default=_json_convert).encode(),
        )

    def _validate_metadata(self) -> None:
        assert len(self.metadata.shape) == len(
            self.metadata.chunk_grid.configuration.chunk_shape
        ), "`chunk_shape` and `shape` need to have the same number of dimensions."
        assert self.metadata.dimension_names is None or len(self.metadata.shape) == len(
            self.metadata.dimension_names
        ), "`dimension_names` and `shape` need to have the same number of dimensions."
        assert self.metadata.fill_value is not None, "`fill_value` is required."

        prev_codec: Optional[Codec] = None
        for codec in self.codecs:
            assert (
                prev_codec is None
                or codec.input_type == "bytes"
                or (codec.input_type == "array" and prev_codec.output_type == "array")
            ), (
                "Ordering of codecs is invalid. Codecs that take arrays as input need "
                + "to be placed before codecs that output bytes."
            )
            if isinstance(codec, ShardingCodec):
                assert len(codec.configuration.chunk_shape) == len(
                    self.metadata.shape
                ), (
                    "The shard's `chunk_shape` and array's `shape` need to have the "
                    + "same number of dimensions."
                )
                assert all(
                    s % c == 0
                    for s, c in zip(
                        self.metadata.chunk_grid.configuration.chunk_shape,
                        codec.configuration.chunk_shape,
                    )
                ), (
                    "The array's `chunk_shape` needs to be divisible by the "
                    + "shard's inner `chunk_shape`."
                )
            prev_codec = codec

        if (
            any(
                codec_metadata.name == "sharding_indexed"
                for codec_metadata in self.metadata.codecs
            )
            and len(self.metadata.codecs) > 1
        ):
            warn(
                "Combining a `sharding_indexed` codec disables partial reads and "
                + "writes, which may lead to inefficient performance."
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
            runtime_configuration=self.runtime_configuration,
        )

    @property
    def async_(self) -> _AsyncArrayProxy:
        return _AsyncArrayProxy(self)

    def __getitem__(self, selection: Selection):
        return sync(self.get_async(selection))

    async def get_async(self, selection: Selection):
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
        await concurrent_map(
            [
                (chunk_coords, chunk_selection, out_selection, out)
                for chunk_coords, chunk_selection, out_selection in indexer
            ],
            self._read_chunk,
            self.runtime_configuration.concurrency,
        )

        if out.shape:
            return out
        else:
            return out[()]

    async def _read_chunk(self, chunk_coords, chunk_selection, out_selection, out):
        chunk_key_encoding = self.metadata.chunk_key_encoding
        chunk_key = f"{self.path}/{chunk_key_encoding.encode_chunk_key(chunk_coords)}"
        value_handle = FileValueHandle(self.store, chunk_key)

        if len(self.codecs) == 1 and isinstance(self.codecs[0], ShardingCodec):
            value_handle = await self.codecs[0].decode_partial(
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

    async def _decode_chunk(
        self, value_handle: ValueHandle, selection: SliceSelection
    ) -> Optional[np.ndarray]:
        if isinstance(value_handle, NoneValueHandle):
            return None

        # apply codecs in reverse order
        for codec in self.codecs[::-1]:
            value_handle = await codec.decode(value_handle, self._core_metadata)

        chunk_array = await value_handle.toarray()
        if chunk_array is None:
            return None

        # ensure correct dtype
        if chunk_array.dtype.name != self.metadata.data_type.name:
            chunk_array = chunk_array.view(self.metadata.dtype)

        # ensure correct chunk shape
        if chunk_array.shape != self.metadata.chunk_grid.configuration.chunk_shape:
            chunk_array = chunk_array.reshape(
                self.metadata.chunk_grid.configuration.chunk_shape,
                order=self.runtime_configuration.order,
            )

        return chunk_array

    def __setitem__(self, selection: Selection, value: np.ndarray) -> None:
        sync(self.set_async(selection, value))

    async def set_async(self, selection: Selection, value: np.ndarray) -> None:
        chunk_shape = self.metadata.chunk_grid.configuration.chunk_shape
        indexer = BasicIndexer(
            selection,
            shape=self.metadata.shape,
            chunk_shape=chunk_shape,
        )

        sel_shape = indexer.shape

        # check value shape
        if np.isscalar(value):
            # setting a scalar value
            pass
        else:
            if not hasattr(value, "shape"):
                value = np.asarray(value, self.metadata.dtype)
            assert value.shape == sel_shape
            if value.dtype.name != self.metadata.dtype.name:
                value = value.astype(self.metadata.dtype, order="A")

        # merging with existing data and encoding chunks
        await concurrent_map(
            [
                (
                    value,
                    chunk_shape,
                    chunk_coords,
                    chunk_selection,
                    out_selection,
                )
                for chunk_coords, chunk_selection, out_selection in indexer
            ],
            self._write_chunk,
            self.runtime_configuration.concurrency,
        )

    async def _write_chunk(
        self,
        value: np.ndarray,
        chunk_shape: ChunkCoords,
        chunk_coords: ChunkCoords,
        chunk_selection: SliceSelection,
        out_selection: SliceSelection,
    ):
        chunk_key_encoding = self.metadata.chunk_key_encoding
        chunk_key = f"{self.path}/{chunk_key_encoding.encode_chunk_key(chunk_coords)}"
        value_handle = FileValueHandle(self.store, chunk_key)

        if is_total_slice(chunk_selection, chunk_shape):
            # write entire chunks
            if np.isscalar(value):
                chunk_array = np.empty(
                    chunk_shape,
                    dtype=self.metadata.dtype,
                    order=self.runtime_configuration.order,
                )
                chunk_array.fill(value)
            else:
                chunk_array = value[out_selection]
            await self._write_chunk_to_store(value_handle, chunk_array)

        elif len(self.codecs) == 1 and isinstance(self.codecs[0], ShardingCodec):
            sharding_codec = self.codecs[0]
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

            await self._write_chunk_to_store(value_handle, chunk_array)

    async def _write_chunk_to_store(
        self, value_handle: ValueHandle, chunk_array: np.ndarray
    ):
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
        for codec in self.codecs:
            encoded_chunk_value = await codec.encode(
                encoded_chunk_value,
                self._core_metadata,
            )

        return encoded_chunk_value

    async def reshape_async(self, new_shape: ChunkCoords) -> "Array":
        assert len(new_shape) == len(self.metadata.shape)
        new_metadata = attr.evolve(self.metadata, shape=new_shape)

        # Remove all chunks outside of the new shape
        chunk_shape = self.metadata.chunk_grid.configuration.chunk_shape
        chunk_key_encoding = self.metadata.chunk_key_encoding
        old_chunk_coords = set(all_chunk_coords(self.metadata.shape, chunk_shape))
        new_chunk_coords = set(all_chunk_coords(new_shape, chunk_shape))

        await concurrent_map(
            [
                (f"{self.path}/{chunk_key_encoding.encode_chunk_key(chunk_coords)}",)
                for chunk_coords in old_chunk_coords.difference(new_chunk_coords)
            ],
            self.store.delete_async,
            self.runtime_configuration.concurrency,
        )

        # Write new metadata
        await self.store.set_async(
            f"{self.path}/{ZARR_JSON}",
            json.dumps(asdict(new_metadata), default=_json_convert).encode(),
        )
        return attr.evolve(self, metadata=new_metadata)

    def reshape(self, new_shape: ChunkCoords) -> "Array":
        return sync(self.reshape_async(new_shape))

    def __repr__(self):
        path = self.path
        return f"<Array {path}>"
