# standard library dependencies
from __future__ import annotations
import json
import numbers
import itertools
import functools
import math
import re
from collections.abc import Mapping, MutableMapping
from typing import Iterator, Union, Optional, Tuple, Any, List, Dict, NamedTuple, Iterable, Type

# third-party dependencies

import fsspec
import numpy as np
import numcodecs
from numcodecs.abc import Codec
from numcodecs.compat import ensure_ndarray


def _json_encode_object(o: Mapping) -> bytes:
    assert isinstance(o, Mapping)
    s = json.dumps(o, ensure_ascii=False, allow_nan=False, indent=4,
                   sort_keys=False)
    b = s.encode("utf8")
    return b


def _json_decode_object(b: bytes) -> Mapping:
    assert isinstance(b, bytes)
    o = json.loads(b)
    assert isinstance(o, Mapping)
    return o


def _check_store(store: Union[str, Store],
                 **storage_options) -> Store:

    # if store arg is a string, assume it"s an fsspec-style URL
    if isinstance(store, str):
        store = FileSystemStore(store, **storage_options)

    assert isinstance(store, Store)

    return store


def create_hierarchy(store: Union[str, Store], **storage_options) -> Hierarchy:

    # sanity checks
    store = _check_store(store, **storage_options)

    # create entry point metadata document
    meta_key_suffix = ".json"
    meta: Dict[str, Any] = dict(
        zarr_format="https://purl.org/zarr/spec/protocol/core/3.0",
        metadata_encoding="https://purl.org/zarr/spec/protocol/core/3.0",
        metadata_key_suffix=meta_key_suffix,
        extensions=[],
    )

    # serialise and store metadata document
    entry_meta_doc = _json_encode_object(meta)
    entry_meta_key = "zarr.json"
    store[entry_meta_key] = entry_meta_doc

    # instantiate a hierarchy
    hierarchy = Hierarchy(store=store, meta_key_suffix=meta_key_suffix)

    return hierarchy


def get_hierarchy(store: Union[str, Store], **storage_options) -> Hierarchy:

    # sanity checks
    store = _check_store(store, **storage_options)

    # retrieve and parse entry point metadata document
    meta_key = "zarr.json"
    meta_doc = store[meta_key]
    meta = _json_decode_object(meta_doc)

    # check protocol version
    zarr_format = meta["zarr_format"]
    protocol_uri, protocol_version = zarr_format.rsplit("/", 1)
    if protocol_uri != "https://purl.org/zarr/spec/protocol/core":
        raise NotImplementedError
    protocol_major_version = int(protocol_version.split(".")[0])
    if protocol_major_version != 3:
        raise NotImplementedError

    # check metadata encoding
    metadata_encoding = meta["metadata_encoding"]
    if metadata_encoding != "https://purl.org/zarr/spec/protocol/core/3.0":
        raise NotImplementedError
    meta_key_suffix = meta["metadata_key_suffix"]

    # check extensions
    extensions = meta["extensions"]
    for spec in extensions:
        if spec["must_understand"]:
            raise NotImplementedError

    # instantiate hierarchy
    hierarchy = Hierarchy(store=store, meta_key_suffix=meta_key_suffix)

    return hierarchy


ALLOWED_NODE_NAME_REGEX = re.compile(r"^\.*[\w\-][\w\-\.]*$", flags=re.ASCII)


def _check_path(path: str) -> str:
    assert isinstance(path, str)
    if len(path) == 0:
        raise ValueError
    if path[0] != "/":
        # handle relative paths, treat as relative to the root, for user convenience
        path = "/" + path
    if len(path) > 1:
        segments = path[1:].split("/")
        for segment in segments:
            if len(segment) == 0 or len(segment) > 255:
                raise ValueError
            if not ALLOWED_NODE_NAME_REGEX.match(segment):
                raise ValueError
    return path


def _check_attrs(attrs: Optional[Mapping]) -> Mapping:
    assert attrs is None or isinstance(attrs, Mapping)
    if attrs is None:
        attrs = dict()
    return attrs


def _check_shape(shape: Union[int, Tuple[int, ...]]) -> Tuple[int, ...]:
    assert isinstance(shape, (int, tuple))
    if isinstance(shape, int):
        shape = shape,
    assert all([isinstance(s, int) for s in shape])
    return shape


def _check_dtype(dtype: Any) -> np.dtype:
    if not isinstance(dtype, np.dtype):
        dtype = np.dtype(dtype)
    assert dtype.str in {
        "|b1", "|i1", "|u1",
        "<i2", "<i4", "<i8",
        ">i2", ">i4", ">i8",
        "<u2", "<u4", "<u8",
        ">u2", ">u4", ">u8",
        "<f2", "<f4", "<f8",
        ">f2", ">f4", ">f8",
    }
    return dtype


def _check_chunk_shape(chunk_shape: Union[int, Tuple[int, ...]],
                       shape: Tuple[int, ...]) -> Tuple[int, ...]:
    assert isinstance(chunk_shape, (int, tuple))
    if isinstance(chunk_shape, int):
        chunk_shape = chunk_shape,
    assert all([isinstance(c, int) for c in chunk_shape])
    assert len(chunk_shape) == len(shape)
    return chunk_shape


def _check_compressor(compressor: Optional[Codec]) -> None:
    assert compressor is None or isinstance(compressor, Codec)


def _check_shard_format(shard_format: str) -> None:
    assert shard_format in SHARDED_STORES, (
        f"Shard format {shard_format} is not supported, "
        + f"use one of {list(SHARDED_STORES)}"
    )


def _check_shards(shards: Union[int, Tuple[int, ...], None]) -> Optional[Tuple[int, ...]]:
    if shards is None:
        return None
    assert isinstance(shards, (int, tuple))
    if isinstance(shards, int):
        shards = shards,
    assert all([isinstance(s, int) for s in shards])
    return shards


def _encode_codec_metadata(codec: Codec) -> Optional[Mapping]:
    if codec is None:
        return None

    # only support gzip for now
    assert codec.codec_id in {"gzip"}
    config = codec.get_config()
    del config["id"]
    meta = {
        "codec": "https://purl.org/zarr/spec/codec/gzip/1.0",
        "configuration": config,
    }
    return meta


def _decode_codec_metadata(meta: Mapping) -> Optional[Codec]:
    if meta is None:
        return None

    # only support gzip for now
    if meta["codec"] != "https://purl.org/zarr/spec/codec/gzip/1.0":
        raise NotImplementedError
    codec = numcodecs.GZip(level=meta["configuration"]["level"])
    return codec


def _array_meta_key(path: str, suffix: str):
    if path == "/":
        # special case root path
        key = "meta/root.array" + suffix
    else:
        key = f"meta/root{path}.array" + suffix
    return key


def _group_meta_key(path: str, suffix: str):
    if path == "/":
        # special case root path
        key = "meta/root.group" + suffix
    else:
        key = f"meta/root{path}.group" + suffix
    return key


class Hierarchy(Mapping):

    def __init__(self, store: Store, meta_key_suffix: str):
        self.store = store
        self.meta_key_suffix = meta_key_suffix

    @property
    def root(self) -> Node:
        return self["/"]

    @property
    def array_suffix(self) -> str:
        return ".array" + self.meta_key_suffix

    @property
    def group_suffix(self) -> str:
        return ".group" + self.meta_key_suffix

    def create_group(self,
                     path: str,
                     attrs: Optional[Mapping] = None) -> ExplicitGroup:

        # sanity checks
        path = _check_path(path)
        _check_attrs(attrs)

        # create group metadata
        meta: Dict[str, Any] = dict(
            extensions=[],
            attributes=attrs,
        )

        # serialise and store metadata document
        meta_doc = _json_encode_object(meta)
        meta_key = _group_meta_key(path, self.meta_key_suffix)
        self.store[meta_key] = meta_doc

        # instantiate group
        group = ExplicitGroup(store=self.store, path=path, owner=self,
                              attrs=attrs)

        return group

    def create_array(self,
                     path: str,
                     shape: Union[int, Tuple[int, ...]],
                     dtype: Any,
                     chunk_shape: Union[int, Tuple[int, ...]],
                     chunk_separator: str = "/",
                     compressor: Optional[Codec] = None,
                     fill_value: Any = None,
                     attrs: Optional[Mapping] = None,
                     shard_format: str = "indexed",
                     shards: Union[int, Tuple[int, ...], None] = None) -> Array:

        # sanity checks
        path = _check_path(path)
        shape = _check_shape(shape)
        dtype = _check_dtype(dtype)
        chunk_shape = _check_chunk_shape(chunk_shape, shape)
        _check_compressor(compressor)
        attrs = _check_attrs(attrs)
        _check_shard_format(shard_format)
        shards = _check_shards(shards)

        # encode data type
        if dtype == np.bool_:
            data_type = "bool"
        else:
            data_type = dtype.str.strip('|')

        # create array metadata
        meta: Dict[str, Any] = dict(
            shape=shape,
            data_type=data_type,
            chunk_grid=dict(
                type="regular",
                chunk_shape=chunk_shape,
                separator=chunk_separator,
            ),
            chunk_memory_layout="C",
            fill_value=fill_value,
            extensions=[],
            attributes=attrs,
        )
        if compressor is not None:
            meta["compressor"] = _encode_codec_metadata(compressor)
        if shards is not None:
            meta["shards"] = shards
            meta["shard_format"] = shard_format

        # serialise and store metadata document
        meta_doc = _json_encode_object(meta)
        meta_key = _array_meta_key(path, self.meta_key_suffix)
        self.store[meta_key] = meta_doc

        # instantiate array
        array = Array(store=self.store, path=path, owner=self,
                      shape=shape, dtype=dtype, chunk_shape=chunk_shape,
                      chunk_separator=chunk_separator, compressor=compressor,
                      fill_value=fill_value, attrs=attrs,
                      shard_format=shard_format, shards=shards)

        return array

    def get_array(self, path: str) -> Array:
        path = _check_path(path)

        # retrieve and parse array metadata document
        meta_key = _array_meta_key(path, self.meta_key_suffix)
        try:
            meta_doc = self.store[meta_key]
        except KeyError:
            raise NodeNotFoundError(path=path)
        meta = _json_decode_object(meta_doc)

        # decode and check metadata
        shape = tuple(meta["shape"])
        _check_shape(shape)
        dtype = _check_dtype(meta["data_type"])
        chunk_grid = meta["chunk_grid"]
        if chunk_grid["type"] != "regular":
            raise NotImplementedError
        chunk_shape = tuple(chunk_grid["chunk_shape"])
        _check_chunk_shape(chunk_shape, shape)
        chunk_separator = chunk_grid["separator"]
        chunk_memory_layout = meta["chunk_memory_layout"]
        if chunk_memory_layout != "C":
            raise NotImplementedError
        compressor = _decode_codec_metadata(meta.get("compressor", None))
        fill_value = meta["fill_value"]
        for spec in meta["extensions"]:
            if spec["must_understand"]:
                raise NotImplementedError(spec)
        attrs = meta["attributes"]
        shards = meta.get("shards", None)
        if shards is not None:
            shards = tuple(shards)
        shard_format = meta.get("shard_format", "indexed")

        # instantiate array
        a = Array(store=self.store, path=path, owner=self, shape=shape,
                  dtype=dtype, chunk_shape=chunk_shape,
                  chunk_separator=chunk_separator, compressor=compressor,
                  fill_value=fill_value, attrs=attrs,
                  shard_format=shard_format, shards=shards)

        return a

    def get_explicit_group(self, path: str) -> ExplicitGroup:
        path = _check_path(path)

        # retrieve and parse group metadata document
        meta_key = _group_meta_key(path, self.meta_key_suffix)
        try:
            meta_doc = self.store[meta_key]
        except KeyError:
            raise NodeNotFoundError(path=path)
        meta = _json_decode_object(meta_doc)

        # check metadata
        attrs = meta["attributes"]

        # instantiate explicit group
        g = ExplicitGroup(store=self.store, path=path, owner=self, attrs=attrs)

        return g

    def get_implicit_group(self, path: str) -> ImplicitGroup:
        path = _check_path(path)

        # attempt to list directory
        if path == "/":
            key_prefix = "meta/root/"
        else:
            key_prefix = f"meta/root{path}/"
        result = self.store.list_dir(key_prefix)
        if not (result.contents or result.prefixes):
            raise NodeNotFoundError(path=path)

        # instantiate implicit group
        g = ImplicitGroup(store=self.store, path=path, owner=self)

        return g

    def __getitem__(self, path: str) -> Node:
        assert isinstance(path, str)

        # try array
        try:
            return self.get_array(path=path)
        except NodeNotFoundError:
            pass

        # try explicit group
        try:
            return self.get_explicit_group(path=path)
        except NodeNotFoundError:
            pass

        # try implicit group
        try:
            return self.get_implicit_group(path=path)
        except NodeNotFoundError:
            pass

        raise KeyError(path)

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __iter__(self) -> Iterator[str]:
        yield from self.get_nodes().keys()

    def __repr__(self) -> str:
        return f"<Hierarchy at {repr(self.store)}>"

    def get_nodes(self) -> Dict:
        nodes: Dict[str, str] = dict()
        result = self.store.list_prefix("meta/")
        for key in result:
            if key == "root.array" + self.meta_key_suffix:
                nodes["/"] = "array"
            elif key == "root.group" + self.meta_key_suffix:
                nodes["/"] = "explicit_group"
            elif key.startswith("root/"):
                # TODO remove code duplication below
                if key.endswith(self.array_suffix):
                    path = key[len("root"):-len(self.array_suffix)]
                    nodes[path] = "array"
                    parent = path.rsplit("/", 1)[0]
                    while parent:
                        nodes.setdefault(parent, "implicit_group")
                        parent = parent.rsplit("/", 1)[0]
                    nodes.setdefault("/", "implicit_group")
                if key.endswith(self.group_suffix):
                    path = key[len("root"):-len(self.group_suffix)]
                    nodes[path] = "explicit_group"
                    parent = path.rsplit("/", 1)[0]
                    while parent:
                        nodes.setdefault(parent, "implicit_group")
                        parent = parent.rsplit("/", 1)[0]
                    nodes.setdefault("/", "implicit_group")

        # sort by path for readability
        items = sorted(nodes.items())
        nodes = dict(items)

        return nodes

    def get_children(self, path: str = "/") -> Dict[str, str]:
        path = _check_path(path)
        children = dict()

        # attempt to list directory
        if path == "/":
            key_prefix = "meta/root/"
        else:
            key_prefix = f"meta/root{path}/"
        result = self.store.list_dir(key_prefix)

        # find explicit children
        for n in result.contents:
            if n.endswith(self.array_suffix):
                node_type = "array"
                name = n[:-len(self.array_suffix)]
                children[name] = node_type
            elif n.endswith(self.group_suffix):
                node_type = "explicit_group"
                name = n[:-len(self.group_suffix)]
                children[name] = node_type

        # find implicit children
        for name in result.prefixes:
            children.setdefault(name, "implicit_group")

        # sort by path for readability
        items = sorted(children.items())
        children = dict(items)

        return children


class NodeNotFoundError(Exception):

    def __init__(self, path: str):
        self.path = path


class Node:

    def __init__(self, store: Store, path: str, owner: Hierarchy):
        self.store = store
        self.path = path
        self.owner = owner

    @property
    def name(self) -> str:
        return self.path.split("/")[-1]


class Group(Node, Mapping):

    def __init__(self, store: Store, path: str, owner: Hierarchy):
        super().__init__(store=store, path=path, owner=owner)

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __iter__(self) -> Iterator[str]:
        yield from self.get_children().keys()

    def get_children(self) -> Dict[str, str]:
        return self.owner.get_children(path=self.path)

    def _dereference_path(self, path: str) -> str:
        assert isinstance(path, str)
        if path[0] != "/":
            # treat as relative path
            if self.path == "/":
                # special case root group
                path = f"/{path}"
            else:
                path = f"{self.path}/{path}"
        if len(path) > 1:
            assert path[-1] != "/"
        return path

    def __getitem__(self, path: str) -> Node:
        path = self._dereference_path(path)
        return self.owner[path]

    def create_group(self, path: str, **kwargs) -> ExplicitGroup:
        path = self._dereference_path(path)
        return self.owner.create_group(path=path, **kwargs)

    def create_array(self, path: str, **kwargs) -> Array:
        path = self._dereference_path(path)
        return self.owner.create_array(path=path, **kwargs)

    def get_array(self, path: str) -> Array:
        path = self._dereference_path(path)
        return self.owner.get_array(path=path)

    def get_explicit_group(self, path: str) -> ExplicitGroup:
        path = self._dereference_path(path)
        return self.owner.get_explicit_group(path=path)

    def get_implicit_group(self, path: str) -> ImplicitGroup:
        path = self._dereference_path(path)
        return self.owner.get_implicit_group(path=path)


class ExplicitGroup(Group):

    def __init__(self, store: Store, path: str, owner: Hierarchy, attrs: Optional[Mapping]):
        super().__init__(store=store, path=path, owner=owner)
        self.attrs = attrs

    def __repr__(self) -> str:
        path = self.path
        return f"<Group {path}>"


class ImplicitGroup(Group):

    def __init__(self, store: Store, path: str, owner: Hierarchy):
        super().__init__(store=store, path=path, owner=owner)

    def __repr__(self) -> str:
        path = self.path
        return f"<Group {path} (implied)>"


class Array(Node):

    def __init__(self,
                 store: Store,
                 path: str,
                 owner: Hierarchy,
                 shape: Tuple[int, ...],
                 dtype: Any,
                 chunk_shape: Tuple[int, ...],
                 chunk_separator: str,
                 compressor: Optional[Codec],
                 fill_value: Any = None,
                 attrs: Optional[Mapping] = None,
                 shard_format: str = "indexed",
                 shards: Optional[Tuple[int, ...]] = None,
                 ):
        if shards is not None:
            store = SHARDED_STORES[shard_format](  # type: ignore
                store=store,
                shards=shards,
                chunk_separator=chunk_separator,
            )
        super().__init__(store=store, path=path, owner=owner)
        self.shape = shape
        self.dtype = dtype
        self.chunk_shape = chunk_shape
        self.chunk_separator = chunk_separator
        self.compressor = compressor
        self.fill_value = fill_value
        self.attrs = attrs

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __getitem__(self, selection):
        return self.get_basic_selection(selection)

    def get_basic_selection(self, selection):
        indexer = _BasicIndexer(selection, shape=self.shape, chunk_shape=self.chunk_shape)
        return self._get_selection(indexer=indexer)

    def _get_selection(self, indexer):

        # setup output array
        out = np.zeros(indexer.shape, dtype=self.dtype, order="C")

        # iterate over chunks
        for chunk_coords, chunk_selection, out_selection in indexer:

            # load chunk selection into output array
            self._chunk_getitem(chunk_coords, chunk_selection, out, out_selection)

        if out.shape:
            return out
        else:
            return out[()]

    def _chunk_getitem(self, chunk_coords, chunk_selection, out, out_selection):

        # obtain key for chunk
        chunk_key = self._chunk_key(chunk_coords)

        try:
            # obtain encoded data for chunk
            encoded_chunk_data = self.store[chunk_key]

        except KeyError:
            # chunk not initialized, maybe fill
            if self.fill_value is not None:
                out[out_selection] = self.fill_value

        else:
            # decode chunk
            chunk = self._decode_chunk(encoded_chunk_data)

            # select data from chunk
            tmp = chunk[chunk_selection]

            # store selected data in output
            out[out_selection] = tmp

    def _chunk_key(self, chunk_coords):
        chunk_identifier = "c" + self.chunk_separator.join(map(str, chunk_coords))
        chunk_key = f"data/root{self.path}/{chunk_identifier}"
        return chunk_key

    def _decode_chunk(self, encoded_data):

        # decompress
        if self.compressor is not None:
            chunk = self.compressor.decode(encoded_data)
        else:
            chunk = encoded_data

        # view as numpy array with correct dtype
        chunk = ensure_ndarray(chunk)
        chunk = chunk.view(self.dtype)

        # ensure correct chunk shape
        chunk = chunk.reshape(-1, order="A")
        chunk = chunk.reshape(self.chunk_shape, order="C")

        return chunk

    def __setitem__(self, selection, value):
        self.set_basic_selection(selection, value)

    def set_basic_selection(self, selection, value):
        indexer = _BasicIndexer(selection, shape=self.shape, chunk_shape=self.chunk_shape)
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

        # iterate over chunks in range
        for chunk_coords, chunk_selection, out_selection in indexer:

            # extract data to store
            if sel_shape == ():
                chunk_value = value
            elif np.isscalar(value):
                chunk_value = value
            else:
                chunk_value = value[out_selection]

            # put data
            self._chunk_setitem(chunk_coords, chunk_selection, chunk_value)

    def _chunk_setitem(self, chunk_coords, chunk_selection, value):

        # obtain key for chunk storage
        chunk_key = self._chunk_key(chunk_coords)

        if _is_total_slice(chunk_selection, self.chunk_shape):
            # totally replace chunk

            # optimization: we are completely replacing the chunk, so no need
            # to access the existing chunk data

            if np.isscalar(value):

                # setup array filled with value
                chunk = np.empty(self.chunk_shape, dtype=self.dtype, order="C")
                chunk.fill(value)

            else:

                # ensure array is contiguous
                chunk = value.astype(self.dtype, order="C", copy=False)

        else:
            # partially replace the contents of this chunk

            try:

                # obtain compressed data for chunk
                encoded_chunk_data = self.store[chunk_key]

            except KeyError:

                # chunk not initialized
                if self.fill_value is not None:
                    chunk = np.empty(self.chunk_shape, dtype=self.dtype, order="C")
                    chunk.fill(self.fill_value)
                else:
                    # N.B., use zeros here so any region beyond the array has consistent
                    # and compressible data
                    chunk = np.zeros(self.chunk_shape, dtype=self.dtype, order="C")

            else:

                # decode chunk
                chunk = self._decode_chunk(encoded_chunk_data)
                if not chunk.flags.writeable:
                    chunk = chunk.copy(order="K")

            # modify
            chunk[chunk_selection] = value

        # encode chunk
        encoded_chunk_data = self._encode_chunk(chunk)

        # store
        self.store[chunk_key] = encoded_chunk_data.tobytes()

    def _encode_chunk(self, chunk):

        # compress
        if self.compressor is not None:
            encoded_chunk_data = self.compressor.encode(chunk)
        else:
            encoded_chunk_data = chunk

        return encoded_chunk_data

    def __repr__(self):
        path = self.path
        return f"<Array {path}>"


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
        item = item,
    if isinstance(item, tuple):
        return all(
            (isinstance(s, slice) and
                ((s == slice(None)) or
                 ((s.stop - s.start == l) and (s.step in [1, None]))))
            for s, l in zip(item, shape)
        )
    else:
        raise TypeError("expected slice or tuple of slices, found %r" % item)


def _ensure_tuple(v):
    if not isinstance(v, tuple):
        v = (v,)
    return v


def _err_too_many_indices(selection, shape):
    raise IndexError("too many indices for array; expected {}, got {}"
                     .format(len(shape), len(selection)))


def _err_boundscheck(dim_len):
    raise IndexError("index out of bounds for dimension with length {}"
                     .format(dim_len))


def _err_negative_step():
    raise IndexError("only slices with step >= 1 are supported")


def _check_selection_length(selection, shape):
    if len(selection) > len(shape):
        _err_too_many_indices(selection, shape)


def _replace_ellipsis(selection, shape):

    selection = _ensure_tuple(selection)

    # count number of ellipsis present
    n_ellipsis = sum(1 for i in selection if i is Ellipsis)

    if n_ellipsis > 1:
        # more than 1 is an error
        raise IndexError("an index can only have a single ellipsis ('...')")

    elif n_ellipsis == 1:
        # locate the ellipsis, count how many items to left and right
        n_items_l = selection.index(Ellipsis)  # items to left of ellipsis
        n_items_r = len(selection) - (n_items_l + 1)  # items to right of ellipsis
        n_items = len(selection) - 1  # all non-ellipsis items

        if n_items >= len(shape):
            # ellipsis does nothing, just remove it
            selection = tuple(i for i in selection if i != Ellipsis)

        else:
            # replace ellipsis with as many slices are needed for number of dims
            new_item = selection[:n_items_l] + ((slice(None),) * (len(shape) - n_items))
            if n_items_r:
                new_item += selection[-n_items_r:]
            selection = new_item

    # fill out selection if not completely specified
    if len(selection) < len(shape):
        selection += (slice(None),) * (len(shape) - len(selection))

    # check selection not too long
    _check_selection_length(selection, shape)

    return selection


class _ChunkDimProjection(NamedTuple):
    dim_chunk_ix: Any
    dim_chunk_sel: Any
    dim_out_sel: Any


def _normalize_integer_selection(dim_sel, dim_len):

    # normalize type to int
    dim_sel = int(dim_sel)

    # handle wraparound
    if dim_sel < 0:
        dim_sel = dim_len + dim_sel

    # handle out of bounds
    if dim_sel >= dim_len or dim_sel < 0:
        _err_boundscheck(dim_len)

    return dim_sel


class _IntDimIndexer(object):

    def __init__(self, dim_sel, dim_len, dim_chunk_len):

        # normalize
        dim_sel = _normalize_integer_selection(dim_sel, dim_len)

        # store attributes
        self.dim_sel = dim_sel
        self.dim_len = dim_len
        self.dim_chunk_len = dim_chunk_len
        self.nitems = 1

    def __iter__(self):
        dim_chunk_ix = self.dim_sel // self.dim_chunk_len
        dim_offset = dim_chunk_ix * self.dim_chunk_len
        dim_chunk_sel = self.dim_sel - dim_offset
        dim_out_sel = None
        yield _ChunkDimProjection(dim_chunk_ix, dim_chunk_sel, dim_out_sel)


def _ceildiv(a, b):
    return math.ceil(a / b)


class _SliceDimIndexer(object):

    def __init__(self, dim_sel, dim_len, dim_chunk_len):

        # normalize
        self.start, self.stop, self.step = dim_sel.indices(dim_len)
        if self.step < 1:
            _err_negative_step()

        # store attributes
        self.dim_len = dim_len
        self.dim_chunk_len = dim_chunk_len
        self.nitems = max(0, _ceildiv((self.stop - self.start), self.step))
        self.nchunks = _ceildiv(self.dim_len, self.dim_chunk_len)

    def __iter__(self):

        # figure out the range of chunks we need to visit
        dim_chunk_ix_from = self.start // self.dim_chunk_len
        dim_chunk_ix_to = _ceildiv(self.stop, self.dim_chunk_len)

        # iterate over chunks in range
        for dim_chunk_ix in range(dim_chunk_ix_from, dim_chunk_ix_to):

            # compute offsets for chunk within overall array
            dim_offset = dim_chunk_ix * self.dim_chunk_len
            dim_limit = min(self.dim_len, (dim_chunk_ix + 1) * self.dim_chunk_len)

            # determine chunk length, accounting for trailing chunk
            dim_chunk_len = dim_limit - dim_offset

            if self.start < dim_offset:
                # selection starts before current chunk
                dim_chunk_sel_start = 0
                remainder = (dim_offset - self.start) % self.step
                if remainder:
                    dim_chunk_sel_start += self.step - remainder
                # compute number of previous items, provides offset into output array
                dim_out_offset = _ceildiv((dim_offset - self.start), self.step)

            else:
                # selection starts within current chunk
                dim_chunk_sel_start = self.start - dim_offset
                dim_out_offset = 0

            if self.stop > dim_limit:
                # selection ends after current chunk
                dim_chunk_sel_stop = dim_chunk_len

            else:
                # selection ends within current chunk
                dim_chunk_sel_stop = self.stop - dim_offset

            dim_chunk_sel = slice(dim_chunk_sel_start, dim_chunk_sel_stop, self.step)
            dim_chunk_nitems = _ceildiv((dim_chunk_sel_stop - dim_chunk_sel_start),
                                        self.step)
            dim_out_sel = slice(dim_out_offset, dim_out_offset + dim_chunk_nitems)

            yield _ChunkDimProjection(dim_chunk_ix, dim_chunk_sel, dim_out_sel)


class _ChunkProjection(NamedTuple):
    chunk_coords: Any
    chunk_selection: Any
    out_selection: Any


class _BasicIndexer(object):

    def __init__(self, selection, shape, chunk_shape):

        # handle ellipsis
        selection = _replace_ellipsis(selection, shape)

        # setup per-dimension indexers
        dim_indexers = []
        for dim_sel, dim_len, dim_chunk_len in zip(selection, shape, chunk_shape):

            if isinstance(dim_sel, numbers.Integral):
                dim_indexer = _IntDimIndexer(dim_sel, dim_len, dim_chunk_len)

            elif isinstance(dim_sel, slice):
                dim_indexer = _SliceDimIndexer(dim_sel, dim_len, dim_chunk_len)

            else:
                raise IndexError("unsupported selection item for basic indexing; "
                                 "expected integer or slice, got {!r}"
                                 .format(type(dim_sel)))

            dim_indexers.append(dim_indexer)

        self.dim_indexers = dim_indexers
        self.shape = tuple(s.nitems for s in self.dim_indexers
                           if not isinstance(s, _IntDimIndexer))

    def __iter__(self):
        for dim_projections in itertools.product(*self.dim_indexers):

            chunk_coords = tuple(p.dim_chunk_ix for p in dim_projections)
            chunk_selection = tuple(p.dim_chunk_sel for p in dim_projections)
            out_selection = tuple(p.dim_out_sel for p in dim_projections
                                  if p.dim_out_sel is not None)

            yield _ChunkProjection(chunk_coords, chunk_selection, out_selection)


class ListDirResult(NamedTuple):
    contents: List[str]
    prefixes: List[str]


class Store(MutableMapping):

    def __getitem__(self, key: str, default: Optional[bytes] = None) -> bytes:
        raise NotImplementedError

    def __setitem__(self, key: str, value: bytes) -> None:
        raise NotImplementedError

    def __delitem__(self, key: str) -> None:
        raise NotImplementedError

    def delitems(self, keys: Iterable[str]) -> None:
        for key in keys:
            del self[key]

    def __iter__(self) -> Iterator[str]:
        raise NotImplementedError

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def list_prefix(self, prefix: str) -> List[str]:
        raise NotImplementedError

    def list_dir(self, prefix: str) -> ListDirResult:
        raise NotImplementedError


class FileSystemStore(Store):

    # TODO ultimately replace this with the fsspec FSMap class, but for now roll
    # our own implementation in order to be able to add some extra methods for
    # listing keys.

    def __init__(self, url: str, **storage_options):
        assert isinstance(url, str)

        # instantiate file system
        fs, root = fsspec.core.url_to_fs(url, **storage_options)
        self.fs = fs
        self.root = root.rstrip("/")

    def __getitem__(self, key: str, default: Optional[bytes] = None) -> bytes:
        assert isinstance(key, str)
        path = f"{self.root}/{key}"

        try:
            value = self.fs.cat(path)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            if default is not None:
                return default
            raise KeyError(key)

        return value

    def __setitem__(self, key: str, value: bytes) -> None:
        assert isinstance(key, str)
        path = f"{self.root}/{key}"

        # ensure parent folder exists
        # noinspection PyProtectedMember
        self.fs.mkdirs(self.fs._parent(path), exist_ok=True)

        # write data
        with self.fs.open(path, "wb") as f:
            f.write(value)

    def __delitem__(self, key: str) -> None:
        assert isinstance(key, str)
        # TODO
        pass

    def __iter__(self) -> Iterator[str]:
        for item in self.fs.find(self.root, withdirs=False, detail=False):
            yield item.split(self.root + "/")[1]

    def list_prefix(self, prefix: str) -> List[str]:
        assert isinstance(prefix, str)
        assert prefix[-1] == "/"
        path = f"{self.root}/{prefix}"
        try:
            items = self.fs.find(path, withdirs=False, detail=False)
        except FileNotFoundError:
            return []
        return [item.split(path)[1] for item in items]

    def list_dir(self, prefix: str = "") -> ListDirResult:
        assert isinstance(prefix, str)
        if prefix:
            assert prefix[-1] == "/"

        # setup result
        contents: List[str] = []
        prefixes: List[str] = []

        # attempt to list directory
        path = f"{self.root}/{prefix}"
        try:
            ls = self.fs.ls(path, detail=True)
        except FileNotFoundError:
            return ListDirResult(contents=contents, prefixes=prefixes)

        # build result
        for item in ls:
            name = item["name"].split(path)[1]
            if item["type"] == "file":
                contents.append(name)
            elif item["type"] == "directory":
                prefixes.append(name)

        return ListDirResult(contents=contents, prefixes=prefixes)

    def __repr__(self) -> str:
        protocol = self.fs.protocol
        if isinstance(protocol, tuple):
            protocol = protocol[-1]
        return f"{protocol}://{self.root}"


MAX_UINT_64 = 2 ** 64 - 1


def _is_data_key(key: str) -> bool:
    return key.startswith("data/root")


class _ShardIndex(NamedTuple):
    store: "IndexedShardedStore"
    offsets_and_lengths: np.ndarray  # dtype uint64, shape (shards_0, _shards_1, ..., 2)

    def __localize_chunk__(self, chunk: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple(chunk_i % shard_i for chunk_i, shard_i in zip(chunk, self.store._shards))

    def get_chunk_slice(self, chunk: Tuple[int, ...]) -> Optional[slice]:
        localized_chunk = self.__localize_chunk__(chunk)
        chunk_start, chunk_len = self.offsets_and_lengths[localized_chunk]
        if (chunk_start, chunk_len) == (MAX_UINT_64, MAX_UINT_64):
            return None
        else:
            return slice(chunk_start, chunk_start + chunk_len)

    def set_chunk_slice(self, chunk: Tuple[int, ...], chunk_slice: Optional[slice]) -> None:
        localized_chunk = self.__localize_chunk__(chunk)
        if chunk_slice is None:
            self.offsets_and_lengths[localized_chunk] = (MAX_UINT_64, MAX_UINT_64)
        else:
            self.offsets_and_lengths[localized_chunk] = (
                chunk_slice.start,
                chunk_slice.stop - chunk_slice.start
            )

    def to_bytes(self) -> bytes:
        return self.offsets_and_lengths.tobytes(order='C')

    @classmethod
    def from_bytes(
        cls, buffer: Union[bytes, bytearray], store: "IndexedShardedStore"
    ) -> "_ShardIndex":
        return cls(
            store=store,
            offsets_and_lengths=np.frombuffer(
                bytearray(buffer), dtype="<u8"
            ).reshape(*store._shards, 2, order="C")
        )

    @classmethod
    def create_empty(cls, store: "IndexedShardedStore"):
        # reserving 2*64bit per chunk for offset and length:
        return cls.from_bytes(
            MAX_UINT_64.to_bytes(8, byteorder="little") * (2 * store._num_chunks_per_shard),
            store=store
        )


class IndexedShardedStore(Store):
    """This class should not be used directly,
    but is added to an Array as a wrapper when needed automatically."""

    def __init__(
        self,
        store: Store,
        shards: Tuple[int, ...],
        chunk_separator: str,
    ) -> None:
        self._store = store
        self._shards = shards
        self._num_chunks_per_shard = functools.reduce(lambda x, y: x*y, shards, 1)
        self._chunk_separator = chunk_separator

    def __key_to_shard__(
        self, chunk_key: str
    ) -> Tuple[str, Tuple[int, ...]]:
        prefix, _, chunk_string = chunk_key.rpartition("c")
        chunk_subkeys = tuple(map(int, chunk_string.split(self._chunk_separator)))
        shard_key_tuple = (
            subkey // shard_i for subkey, shard_i in zip(chunk_subkeys, self._shards)
        )
        shard_key = prefix + "c" + self._chunk_separator.join(map(str, shard_key_tuple))
        return shard_key, chunk_subkeys

    def __get_index__(self, buffer: Union[bytes, bytearray]) -> _ShardIndex:
        # At the end of each shard 2*64bit per chunk for offset and length define the index:
        return _ShardIndex.from_bytes(buffer[-16 * self._num_chunks_per_shard:], self)

    def __get_chunks_in_shard(self, shard_key: str) -> Iterator[Tuple[int, ...]]:
        _, _, chunk_string = shard_key.rpartition("c")
        shard_key_tuple = tuple(map(int, chunk_string.split(self._chunk_separator)))
        for chunk_offset in itertools.product(*(range(i) for i in self._shards)):
            yield tuple(
                shard_key_i * shards_i + offset_i
                for shard_key_i, offset_i, shards_i
                in zip(shard_key_tuple, chunk_offset, self._shards)
            )

    def __getitem__(self, key: str, default: Optional[bytes] = None) -> bytes:
        if _is_data_key(key):
            shard_key, chunk_subkeys = self.__key_to_shard__(key)
            full_shard_value = self._store[shard_key]
            index = self.__get_index__(full_shard_value)
            chunk_slice = index.get_chunk_slice(chunk_subkeys)
            if chunk_slice is not None:
                return full_shard_value[chunk_slice]
            else:
                if default is not None:
                    return default
                raise KeyError(key)
        else:
            return self._store.__getitem__(key, default)

    def __setitem__(self, key: str, value: bytes) -> None:
        if _is_data_key(key):
            shard_key, chunk_subkeys = self.__key_to_shard__(key)
            chunks_to_read = set(self.__get_chunks_in_shard(shard_key))
            chunks_to_read.remove(chunk_subkeys)
            new_content = {chunk_subkeys: value}
            try:
                full_shard_value = self._store[shard_key]
            except KeyError:
                index = _ShardIndex.create_empty(self)
            else:
                index = self.__get_index__(full_shard_value)
                for chunk_to_read in chunks_to_read:
                    chunk_slice = index.get_chunk_slice(chunk_to_read)
                    if chunk_slice is not None:
                        new_content[chunk_to_read] = full_shard_value[chunk_slice]

            shard_content = b""
            for chunk_subkeys, chunk_content in new_content.items():
                chunk_slice = slice(len(shard_content), len(shard_content) + len(chunk_content))
                index.set_chunk_slice(chunk_subkeys, chunk_slice)
                shard_content += chunk_content
            # Appending the index at the end of the shard:
            shard_content += index.to_bytes()
            self._store[shard_key] = shard_content
        else:
            self._store[key] = value

    def delitems(self, keys: Iterable[str]) -> None:
        raise NotImplementedError

    def __shard_key_to_original_keys__(self, key: str) -> Iterator[str]:
        if not _is_data_key(key):
            # Special keys such as meta-keys are passed on as-is
            yield key
        else:
            index = self.__get_index__(self._store[key])
            prefix, _, _ = key.rpartition("c")
            for chunk_tuple in self.__get_chunks_in_shard(key):
                if index.get_chunk_slice(chunk_tuple) is not None:
                    yield prefix + "c" + self._chunk_separator.join(map(str, chunk_tuple))

    def __iter__(self) -> Iterator[str]:
        for key in self._store:
            yield from self.__shard_key_to_original_keys__(key)

    def list_prefix(self, prefix: str) -> List[str]:
        if _is_data_key(prefix):
            # Needs translation of the prefix to shard_key
            raise NotImplementedError
        return self._store.list_prefix(prefix)

    def list_dir(self, prefix: str) -> ListDirResult:
        if _is_data_key(prefix):
            # Needs translation of the prefix to shard_key
            raise NotImplementedError
        return self._store.list_dir(prefix)


SHARDED_STORES: Dict[str, Type[Store]] = {
    "indexed": IndexedShardedStore,
}
