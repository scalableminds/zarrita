from enum import Enum
import json
import sys
from typing import (
    Final,
    Literal,
    Union,
    Optional,
    Tuple,
    Any,
    List,
    Dict,
)
from attrs import define, field, asdict

import numpy as np
from numcodecs.blosc import Blosc
from numcodecs.gzip import GZip


from zarrita.indexing import _BasicIndexer
from zarrita.sharding import (
    ShardingCodecMetadata,
    ShardingCodecConfigurationMetadata,
)
from zarrita.store import (
    BufferHandle,
    FileHandle,
    Store,
    FileSystemStore,
    ValueHandle,
    ArrayHandle,
)


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
    chunk_shape: List[int]


@define
class RegularChunkGridMetadata:
    configuration: RegularChunkGridConfigurationMetadata
    name: Final = "regular"


@define
class DefaultChunkKeyEncodingConfigurationMetadata:
    separator: Optional[Union[Literal["."], Literal["/"]]] = "/"


@define
class DefaultChunkKeyEncodingMetadata:
    configuration: Optional[DefaultChunkKeyEncodingConfigurationMetadata]
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
    separator: Optional[Union[Literal["."], Literal["/"]]] = "."


@define
class V2ChunkKeyEncodingMetadata:
    configuration: Optional[V2ChunkKeyEncodingConfigurationMetadata]
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


class Codec:
    def decode(
        self,
        file: ValueHandle,
        selection: Tuple[slice, ...],
        array_metadata: CoreArrayMetadata,
    ) -> ValueHandle:
        pass

    def encode(
        self,
        value: ValueHandle,
        selection: Tuple[slice, ...],
        array_metadata: CoreArrayMetadata,
    ) -> ValueHandle:
        pass


@define
class BloscCodecConfigurationMetadata:
    cname: Union[
        Literal["lz4"],
        Literal["lz4hc"],
        Literal["blosclz"],
        Literal["zstd"],
        Literal["snappy"],
        Literal["zlib"],
    ] = "zstd"
    clevel: int = 5
    shuffle: Union[Literal[0], Literal[1], Literal[2], Literal[-1]] = 0
    blocksize: int = 0


@define
class BloscCodecMetadata:
    configuration: BloscCodecConfigurationMetadata
    name: Final = "blosc"

    def decode(
        self,
        value: ValueHandle,
        _selection: Tuple[slice, ...],
        _array_metadata: CoreArrayMetadata,
    ) -> ValueHandle:
        return BufferHandle(
            Blosc.from_config(asdict(self.configuration)).decode(value.tobytes())
        )

    def encode(
        self,
        value: ValueHandle,
        _selection: Tuple[int, ...],
        _array_metadata: CoreArrayMetadata,
    ) -> ValueHandle:
        chunk = value.toarray()
        if not chunk.flags.c_contiguous and not chunk.flags.f_contiguous:
            chunk = chunk.copy(order="K")
        return BufferHandle(Blosc.from_config(asdict(self.configuration)).encode(chunk))


@define
class EndianCodecConfigurationMetadata:
    endian: Union[Literal["big"], Literal["little"]] = "little"


@define
class EndianCodecMetadata:
    configuration: EndianCodecConfigurationMetadata
    name: Final = "endian"

    def _get_byteorder(
        self, array: np.ndarray
    ) -> Union[Literal["big"], Literal["little"]]:
        if array.dtype.byteorder == "<":
            return "little"
        elif array.dtype.byteorder == ">":
            return "big"
        else:
            return sys.byteorder

    def decode(
        self,
        value: ValueHandle,
        _selection: Tuple[slice, ...],
        _array_metadata: CoreArrayMetadata,
    ) -> ValueHandle:
        encoded_chunk = value.toarray()
        byteorder = self._get_byteorder(encoded_chunk)
        if self.configuration.endian != byteorder:
            encoded_chunk = encoded_chunk.view(
                dtype=encoded_chunk.dtype.newbyteorder(byteorder)
            )
        return ArrayHandle(encoded_chunk)

    def encode(
        self,
        value: ValueHandle,
        _selection: Tuple[int, ...],
        _array_metadata: CoreArrayMetadata,
    ) -> ValueHandle:
        encoded_chunk = value.toarray()
        byteorder = self._get_byteorder(encoded_chunk)
        if self.configuration.endian != byteorder:
            encoded_chunk = encoded_chunk.view(
                dtype=encoded_chunk.dtype.newbyteorder(byteorder)
            )
        return ArrayHandle(encoded_chunk)


@define
class TransposeCodecConfigurationMetadata:
    order: Union[Literal["C"], Literal["F"], List[int]] = "C"


@define
class TransposeCodecMetadata:
    configuration: TransposeCodecConfigurationMetadata
    name: Final = "transpose"

    def decode(
        self,
        value: ValueHandle,
        _selection: Tuple[slice, ...],
        array_metadata: CoreArrayMetadata,
    ) -> ValueHandle:
        encoded_chunk = value.toarray()
        assert self.configuration.order in ("C", "F")
        order = "C" if encoded_chunk.flags["C_CONTIGUOUS"] else "F"
        if self.configuration.order != order:
            encoded_chunk = encoded_chunk.view(
                np.dtype(array_metadata.data_type.value)
            ).reshape(
                array_metadata.chunk_shape,
                order=self.configuration.order,
            )
        return ArrayHandle(encoded_chunk)

    def encode(
        self,
        value: ValueHandle,
        _selection: Tuple[int, ...],
        _array_metadata: CoreArrayMetadata,
    ) -> ValueHandle:
        encoded_chunk = value.toarray()
        assert self.configuration.order in ("C", "F")
        encoded_chunk = encoded_chunk.reshape(-1, order=self.configuration.order)
        return ArrayHandle(encoded_chunk)


@define
class GzipCodecConfigurationMetadata:
    level: int = 5


@define
class GzipCodecMetadata:
    configuration: GzipCodecConfigurationMetadata
    name: Final = "gzip"

    def decode(
        self,
        value: ValueHandle,
        _selection: Tuple[slice, ...],
        _array_metadata: CoreArrayMetadata,
    ) -> ValueHandle:
        return BufferHandle(
            GZip.from_config(self.configuration).decode(value.tobytes())
        )

    def encode(
        self,
        value: ValueHandle,
        _selection: Tuple[int, ...],
        _array_metadata: CoreArrayMetadata,
    ) -> ValueHandle:
        return BufferHandle(
            GZip.from_config(self.configuration).encode(value.tobytes())
        )


CodecMetadata = Union[
    BloscCodecMetadata,
    EndianCodecMetadata,
    TransposeCodecMetadata,
    GzipCodecMetadata,
    ShardingCodecMetadata,
]


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


@define
class GroupMetadata:
    attributes: Dict[str, Any] = field(factory=dict)
    zarr_format: Final = 3
    node_type: Final = "group"


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
            Tuple[Literal["default"], Union[Literal["."], Literal["/"]]],
            Tuple[Literal["v2"], Union[Literal["."], Literal["/"]]],
        ] = ("default", "/"),
        codecs: Optional[List[CodecMetadata]] = None,
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
        )
        array = cls()
        array.metadata = metadata
        array.store = store
        array.path = path
        array._save_metadata()
        return array

    @classmethod
    def open(cls, store: "Store", path: str) -> "Array":
        metadata = ArrayMetadata(**json.loads(store.get(f"{path}/zarr.json")))
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
            f"{self.path}/zarr.json",
            json.dumps(asdict(self.metadata), default=convert).encode(),
        )

    @property
    def ndim(self) -> int:
        return len(self.metadata.shape)

    def __getitem__(self, selection):
        return self.get_basic_selection(selection)

    def get_basic_selection(self, selection):
        indexer = _BasicIndexer(
            selection,
            shape=self.metadata.shape,
            chunk_shape=self.metadata.chunk_grid.configuration.chunk_shape,
        )
        return self._get_selection(indexer=indexer)

    def _get_order(self) -> Union[Literal["C"], Literal["F"]]:
        for codec in self.metadata.codecs:
            if codec.name == "transpose":
                return codec.configuration.order
        return "C"

    def _get_selection(self, indexer):
        # setup output array
        out = np.zeros(
            indexer.shape, dtype=self.metadata.dtype, order=self._get_order()
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
        self.set_basic_selection(selection, value)

    def set_basic_selection(self, selection, value):
        indexer = _BasicIndexer(
            selection,
            shape=self.metadata.shape,
            chunk_shape=self.metadata.chunk_grid.configuration.chunk_shape,
        )
        self._set_selection(indexer, value)

    def _set_selection(self, indexer, value):
        # We iterate over all chunks which overlap the selection and thus contain data
        # that needs to be replaced. Each chunk is processed in turn, extracting the
        # necessary data from the value array and storing into the chunk array.

        # N.B., it is an important optimisation that we only visit chunks which overlap
        # the selection. This minimises the number of iterations in the main for loop.

        # determine indices of chunks overlapping the selection
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
        encoded_chunk_value = ArrayHandle(chunk_value)
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


# def _check_store(store: Union[str, Store], **storage_options) -> Store:
#     # if store arg is a string, assume it"s an fsspec-style URL
#     if isinstance(store, str):
#         store = FileSystemStore(store, **storage_options)

#     assert isinstance(store, Store)

#     return store


# def create_group(store: Union[str, Store], **storage_options) -> Hierarchy:
#     # sanity checks
#     store = _check_store(store, **storage_options)

#     # create entry point metadata document
#     meta: Dict[str, Any] = dict(
#         zarr_format=3,
#         node_type="group",
#     )

#     # serialise and store metadata document
#     entry_meta_doc = _json_encode_object(meta)
#     entry_meta_key = "zarr.json"
#     store[entry_meta_key] = entry_meta_doc

#     # instantiate a hierarchy
#     hierarchy = Hierarchy(store=store)

#     return hierarchy


# class Group(Node, Mapping):
#     def __init__(self, store: Store, path: str):
#         super().__init__(store=store, path=path)

#     def __len__(self) -> int:
#         return sum(1 for _ in self)

#     def __iter__(self) -> Iterator[str]:
#         yield from self.get_children().keys()

#     def get_children(self) -> Dict[str, str]:
#         return self.owner.get_children(path=self.path)

#     def _dereference_path(self, path: str) -> str:
#         assert isinstance(path, str)
#         if path[0] != "/":
#             # treat as relative path
#             if self.path == "/":
#                 # special case root group
#                 path = f"/{path}"
#             else:
#                 path = f"{self.path}/{path}"
#         if len(path) > 1:
#             assert path[-1] != "/"
#         return path

#     def __getitem__(self, path: str) -> Node:
#         path = self._dereference_path(path)
#         return self.owner[path]

#     def create_group(self, path: str, **kwargs) -> Group:
#         path = self._dereference_path(path)
#         return self.owner.create_group(path=path, **kwargs)

#     def create_array(self, path: str, **kwargs) -> Array:
#         path = self._dereference_path(path)
#         return self.owner.create_array(path=path, **kwargs)

#     def get_array(self, path: str) -> Array:
#         path = self._dereference_path(path)
#         return self.owner.get_array(path=path)


def _is_total_slice(item, shape):
    """Determine whether `item` specifies a complete slice of array with the
    given `shape`. Used to optimize __setitem__ operations on the Chunk
    class."""

    # N.B., assume shape is normalized

    if item == Ellipsis:
        return True
    if item == slice(None):
        return True
    if isinstance(item, slice):
        item = (item,)
    if isinstance(item, tuple):
        return all(
            (
                isinstance(s, slice)
                and (
                    (s == slice(None))
                    or ((s.stop - s.start == l) and (s.step in [1, None]))
                )
            )
            for s, l in zip(item, shape)
        )
    else:
        raise TypeError("expected slice or tuple of slices, found %r" % item)
