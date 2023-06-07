import asyncio
import json
from typing import Any, Dict, List, Literal, Optional, Union

import numcodecs
import numpy as np
from attr import asdict, frozen
from numcodecs.compat import ensure_bytes, ensure_ndarray

from zarrita.common import (
    ZARRAY_JSON,
    ZATTRS_JSON,
    BytesLike,
    ChunkCoords,
    Selection,
    SliceSelection,
    make_cattr,
    to_thread,
)
from zarrita.indexing import BasicIndexer, is_total_slice
from zarrita.metadata import ArrayV2Metadata
from zarrita.store import Store
from zarrita.sync import sync
from zarrita.value_handle import (
    BufferValueHandle,
    FileValueHandle,
    NoneValueHandle,
    ValueHandle,
)


@frozen
class _AsyncArrayProxy:
    array: "ArrayV2"

    def __getitem__(self, selection: Selection) -> "_AsyncArraySelectionProxy":
        return _AsyncArraySelectionProxy(self.array, selection)


@frozen
class _AsyncArraySelectionProxy:
    array: "ArrayV2"
    selection: Selection

    async def get(self) -> np.ndarray:
        return await self.array.get_async(self.selection)

    async def set(self, value: np.ndarray):
        return await self.array.set_async(self.selection, value)


@frozen
class ArrayV2:
    metadata: ArrayV2Metadata
    attributes: Optional[Dict[str, Any]]
    store: "Store"
    path: str

    @classmethod
    async def create_async(
        cls,
        store: "Store",
        path: str,
        *,
        shape: ChunkCoords,
        dtype: np.dtype,
        chunks: ChunkCoords,
        dimension_separator: Literal[".", "/"] = ".",
        fill_value: Optional[Union[None, int, float]] = None,
        order: Literal["C", "F"] = "C",
        filters: Optional[List[Dict[str, Any]]] = None,
        compressor: Optional[Dict[str, Any]] = None,
        attributes: Optional[Dict[str, Any]] = None,
        exists_ok: bool = False,
    ) -> "ArrayV2":
        if not exists_ok:
            assert not await store.exists_async(f"{path}/{ZARRAY_JSON}")

        metadata = ArrayV2Metadata(
            shape=shape,
            dtype=dtype,
            chunks=chunks,
            order=order,
            dimension_separator=dimension_separator,
            fill_value=0 if fill_value is None else fill_value,
            compressor=numcodecs.get_codec(compressor).get_config()
            if compressor is not None
            else None,
            filters=[numcodecs.get_codec(filter).get_config() for filter in filters]
            if filters is not None
            else None,
        )
        array = cls(
            metadata=metadata,
            store=store,
            path=path,
            attributes=attributes,
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
        dtype: np.dtype,
        chunks: ChunkCoords,
        dimension_separator: Literal[".", "/"] = ".",
        fill_value: Optional[Union[None, int, float]] = None,
        order: Literal["C", "F"] = "C",
        filters: Optional[List[Dict[str, Any]]] = None,
        compressor: Optional[Dict[str, Any]] = None,
        attributes: Optional[Dict[str, Any]] = None,
        exists_ok: bool = False,
    ) -> "ArrayV2":
        return sync(
            cls.create_async(
                store,
                path,
                shape=shape,
                dtype=dtype,
                chunks=chunks,
                order=order,
                dimension_separator=dimension_separator,
                fill_value=0 if fill_value is None else fill_value,
                compressor=compressor,
                filters=filters,
                attributes=attributes,
                exists_ok=exists_ok,
            )
        )

    @classmethod
    async def open_async(
        cls,
        store: "Store",
        path: str,
    ) -> "ArrayV2":
        zarray_bytes, zattrs_bytes = await asyncio.gather(
            store.get_async(f"{path}/{ZARRAY_JSON}"),
            store.get_async(f"{path}/{ZATTRS_JSON}"),
        )
        assert zarray_bytes is not None
        return cls.from_json(
            store,
            path,
            zarray_json=json.loads(zarray_bytes),
            zattrs_json=json.loads(zattrs_bytes) if zattrs_bytes is not None else None,
        )

    @classmethod
    def open(
        cls,
        store: "Store",
        path: str,
    ) -> "ArrayV2":
        return sync(
            cls.open_async(
                store,
                path,
            )
        )

    @classmethod
    def from_json(
        cls,
        store: Store,
        path: str,
        zarray_json: Any,
        zattrs_json: Optional[Any],
    ) -> "ArrayV2":
        metadata = make_cattr().structure(zarray_json, ArrayV2Metadata)
        out = cls(
            store=store,
            path=path,
            metadata=metadata,
            attributes=zattrs_json,
        )
        out._validate_metadata()
        return out

    async def _save_metadata(self) -> None:
        def convert(o):
            if isinstance(o, np.dtype):
                if o.fields is None:
                    return o.str
                else:
                    return o.descr
            raise TypeError

        self._validate_metadata()

        await self.store.set_async(
            f"{self.path}/{ZARRAY_JSON}",
            json.dumps(asdict(self.metadata), default=convert).encode(),
        )
        if self.attributes is not None and len(self.attributes) > 0:
            await self.store.set_async(
                f"{self.path}/{ZATTRS_JSON}",
                json.dumps(self.attributes).encode(),
            )
        else:
            await self.store.delete_async(
                f"{self.path}/{ZATTRS_JSON}",
            )

    def _validate_metadata(self) -> None:
        assert len(self.metadata.shape) == len(
            self.metadata.chunks
        ), "`chunks` and `shape` need to have the same number of dimensions."

    @property
    def ndim(self) -> int:
        return len(self.metadata.shape)

    @property
    def async_(self) -> _AsyncArrayProxy:
        return _AsyncArrayProxy(self)

    def __getitem__(self, selection: Selection):
        return sync(self.get_async(selection))

    async def get_async(self, selection: Selection):
        indexer = BasicIndexer(
            selection,
            shape=self.metadata.shape,
            chunk_shape=self.metadata.chunks,
        )

        # setup output array
        out = np.zeros(
            indexer.shape,
            dtype=self.metadata.dtype,
            order=self.metadata.order,
        )

        # reading chunks and decoding them
        await asyncio.gather(
            *[
                self._read_chunk(chunk_coords, chunk_selection, out_selection, out)
                for chunk_coords, chunk_selection, out_selection in indexer
            ]
        )

        if out.shape:
            return out
        else:
            return out[()]

    async def _read_chunk(self, chunk_coords, chunk_selection, out_selection, out):
        chunk_key = f"{self.path}/{self._encode_chunk_key(chunk_coords)}"
        value_handle = FileValueHandle(self.store, chunk_key)

        chunk_array = await self._decode_chunk(await value_handle.tobytes())
        if chunk_array is not None:
            tmp = chunk_array[chunk_selection]
            out[out_selection] = tmp
        else:
            out[out_selection] = self.metadata.fill_value

    async def _decode_chunk(
        self, chunk_bytes: Optional[BytesLike]
    ) -> Optional[np.ndarray]:
        if chunk_bytes is None:
            return None

        if self.metadata.compressor is not None:
            compressor = numcodecs.get_codec(self.metadata.compressor)
            chunk_array = ensure_ndarray(
                await to_thread(compressor.decode, chunk_bytes)
            )
        else:
            chunk_array = ensure_ndarray(chunk_bytes)

        # ensure correct dtype
        if str(chunk_array.dtype) != self.metadata.dtype:
            chunk_array = chunk_array.view(self.metadata.dtype)

        # apply filters in reverse order
        if self.metadata.filters is not None:
            for filter_metadata in self.metadata.filters[::-1]:
                filter = numcodecs.get_codec(filter_metadata)
                chunk_array = await to_thread(filter.decode, chunk_array)

        # ensure correct chunk shape
        if chunk_array.shape != self.metadata.chunks:
            chunk_array = chunk_array.reshape(
                self.metadata.chunks,
                order=self.metadata.order,
            )

        return chunk_array

    def __setitem__(self, selection: Selection, value: np.ndarray) -> None:
        sync(self.set_async(selection, value))

    async def set_async(self, selection: Selection, value: np.ndarray) -> None:
        chunk_shape = self.metadata.chunks
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
            if value.dtype != self.metadata.dtype:
                value = value.astype(self.metadata.dtype, order="A")

        # merging with existing data and encoding chunks
        await asyncio.gather(
            *[
                self._write_chunk(
                    value,
                    chunk_shape,
                    chunk_coords,
                    chunk_selection,
                    out_selection,
                )
                for chunk_coords, chunk_selection, out_selection in indexer
            ]
        )

    async def _write_chunk(
        self,
        value: np.ndarray,
        chunk_shape: ChunkCoords,
        chunk_coords: ChunkCoords,
        chunk_selection: SliceSelection,
        out_selection: SliceSelection,
    ):
        chunk_key = f"{self.path}/{self._encode_chunk_key(chunk_coords)}"
        value_handle = FileValueHandle(self.store, chunk_key)

        if is_total_slice(chunk_selection, chunk_shape):
            # write entire chunks
            if np.isscalar(value):
                chunk_array = np.empty(
                    chunk_shape,
                    dtype=self.metadata.dtype,
                    order=self.metadata.order,
                )
                chunk_array.fill(value)
            else:
                chunk_array = value[out_selection]
            await self._write_chunk_to_store(value_handle, chunk_array)

        else:
            # writing partial chunks
            # read chunk first
            tmp = await self._decode_chunk(await value_handle.tobytes())

            # merge new value
            if tmp is None:
                chunk_array = np.empty(
                    chunk_shape,
                    dtype=self.metadata.dtype,
                    order=self.metadata.order,
                )
                chunk_array.fill(self.metadata.fill_value)
            else:
                chunk_array = tmp.copy(
                    order=self.metadata.order,
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
        chunk_array = chunk_array.ravel(order=self.metadata.order)

        if self.metadata.filters is not None:
            for filter_metadata in self.metadata.filters:
                filter = numcodecs.get_codec(filter_metadata)
                chunk_array = await to_thread(filter.encode, chunk_array)

        if self.metadata.compressor is not None:
            compressor = numcodecs.get_codec(self.metadata.compressor)
            if (
                not chunk_array.flags.c_contiguous
                and not chunk_array.flags.f_contiguous
            ):
                chunk_array = chunk_array.copy(order="A")
            encoded_chunk_bytes = ensure_bytes(
                await to_thread(compressor.encode, chunk_array)
            )
        else:
            encoded_chunk_bytes = ensure_bytes(chunk_array)

        return BufferValueHandle(encoded_chunk_bytes)

    def _encode_chunk_key(self, chunk_coords: ChunkCoords) -> str:
        chunk_identifier = self.metadata.dimension_separator.join(
            map(str, chunk_coords)
        )
        return "0" if chunk_identifier == "" else chunk_identifier

    def __repr__(self):
        path = self.path
        return f"<Array_v2 {path}>"
