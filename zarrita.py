# standard library dependencies
from __future__ import annotations
import json
import numbers
import itertools
import collections
import math
from collections.abc import Mapping, MutableMapping
from typing import Iterator, Union, Optional, Tuple, Any, List, Dict, NamedTuple


# third-party dependencies

import fsspec
import numpy as np
import numcodecs
from numcodecs.abc import Codec
from numcodecs.compat import ensure_ndarray


def _json_encode(o: Mapping) -> bytes:
    s = json.dumps(o, ensure_ascii=False, allow_nan=False, indent=4,
                   sort_keys=False)
    b = s.encode('utf8')
    return b


def _json_decode(b: bytes) -> Mapping:
    assert isinstance(b, bytes)
    o = json.loads(b)
    return o


def _check_store(store: Union[str, Store],
                 **storage_options) -> Store:

    # if store arg is a string, assume it's an fsspec-style URL
    if isinstance(store, str):
        store = FileSystemStore(store, **storage_options)

    assert isinstance(store, Store)

    return store


def create_hierarchy(store: Store, **storage_options) -> Hierarchy:

    # sanity checks
    store = _check_store(store, **storage_options)

    # create entry point metadata document
    meta: Dict[str, Any] = dict(
        zarr_format='https://purl.org/zarr/spec/protocol/core/3.0',
        metadata_encoding='application/json',
        extensions=[],
    )

    # serialise and store metadata document
    meta_doc = _json_encode(meta)
    meta_key = 'zarr.json'
    store[meta_key] = meta_doc

    # instantiate a hierarchy
    hierarchy = Hierarchy(store=store)

    return hierarchy


def get_hierarchy(store: Store, **storage_options) -> Hierarchy:

    # sanity checks
    store = _check_store(store, **storage_options)

    # retrieve and parse entry point metadata document
    meta_key = 'zarr.json'
    meta_doc = store[meta_key]
    meta = _json_decode(meta_doc)

    # check protocol version
    zarr_format = meta['zarr_format']
    protocol_version = zarr_format.split('/')[-1]
    protocol_major_version = int(protocol_version.split('.')[0])
    if protocol_major_version != 3:
        raise NotImplementedError

    # check metadata encoding
    metadata_encoding = meta['metadata_encoding']
    if metadata_encoding != 'application/json':
        raise NotImplementedError

    # check extensions
    extensions = meta['extensions']
    for spec in extensions:
        if spec['must_understand']:
            raise NotImplementedError

    # instantiate hierarchy
    hierarchy = Hierarchy(store=store)

    return hierarchy


ALLOWED_NODE_NAME_CHARS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ._-'


def _check_path(path: str) -> None:
    assert isinstance(path, str)
    if len(path) == 0:
        raise ValueError
    if path[0] != '/':
        raise ValueError
    if len(path) > 1:
        segments = path[1:].split('/')
        for segment in segments:
            if len(segment) == 0:
                raise ValueError
            for c in segment:
                if c not in ALLOWED_NODE_NAME_CHARS:
                    raise ValueError
            if all([c == '.' for c in segment]):
                raise ValueError


def _check_attrs(attrs: Optional[Mapping]) -> None:
    assert attrs is None or isinstance(attrs, Mapping)


def _check_shape(shape: Tuple[Any, ...]) -> None:
    assert isinstance(shape, tuple)
    assert all([isinstance(s, int) for s in shape])


def _check_dtype(dtype: Any) -> np.dtype:
    if not isinstance(dtype, np.dtype):
        dtype = np.dtype(dtype)
    assert dtype.str in {
        '|b1', 'i1', 'u1',
        '<i2', '<i4', '<i8',
        '>i2', '>i4', '>i8',
        '<u2', '<u4', '<u8',
        '>u2', '>u4', '>u8',
        '<f2', '<f4', '<f8',
        '>f2', '>f4', '>f8',
    }
    return dtype


def _check_chunk_shape(chunk_shape: Tuple[Any, ...], shape: Tuple[Any, ...]) -> None:
    assert isinstance(chunk_shape, tuple)
    assert all([isinstance(c, int) for c in chunk_shape])
    assert len(chunk_shape) == len(shape)


def _check_compressor(compressor: Optional[Codec]) -> None:
    assert compressor is None or isinstance(compressor, Codec)


def _encode_codec_metadata(codec: Codec) -> Optional[Mapping]:
    if codec is None:
        return None

    # only support gzip for now
    assert codec.codec_id in {'gzip'}
    config = codec.get_config()
    del config['id']
    meta = {
        'codec': 'https://purl.org/zarr/spec/codec/gzip/1.0',
        'configuration': config,
    }
    return meta


def _decode_codec_metadata(meta: Mapping) -> Optional[Codec]:
    if meta is None:
        return None

    # only support gzip for now
    if meta['codec'] != 'https://purl.org/zarr/spec/codec/gzip/1.0':
        raise NotImplementedError
    codec = numcodecs.GZip(level=meta['configuration']['level'])
    return codec


class Hierarchy(Mapping):

    def __init__(self, store: Store):
        self.store = store

    @property
    def root(self) -> Node:
        return self['/']

    def create_group(self,
                     path: str,
                     attrs: Mapping = None) -> ExplicitGroup:

        # sanity checks
        _check_path(path)
        _check_attrs(attrs)

        # create group metadata
        meta: Dict[str, Any] = dict(
            extensions=[],
            attributes=attrs,
        )

        # serialise and store metadata document
        meta_doc = _json_encode(meta)
        if path == '/':
            # special case root path
            meta_key = 'meta/root.group'
        else:
            meta_key = f'meta/root{path}.group'
        self.store[meta_key] = meta_doc

        # instantiate group
        group = ExplicitGroup(store=self.store, path=path, owner=self, 
                              attrs=attrs)

        return group

    def create_array(self,
                     path: str,
                     shape: Tuple[int],
                     dtype: Any,
                     chunk_shape: Tuple[int],
                     compressor: Optional[Codec] = None,
                     fill_value: Any = None,
                     attrs: Optional[Mapping] = None) -> Array:

        # sanity checks
        _check_path(path)
        _check_shape(shape)
        dtype = _check_dtype(dtype)
        _check_chunk_shape(chunk_shape, shape)
        _check_compressor(compressor)
        _check_attrs(attrs)

        # encode data type
        if dtype == np.bool:
            data_type = 'bool'
        else:
            data_type = dtype.str

        # create array metadata
        meta: Dict[str, Any] = dict(
            shape=shape,
            data_type=data_type,
            chunk_grid=dict(
                type='regular',
                chunk_shape=chunk_shape,
            ),
            chunk_memory_layout='C',
            compressor=_encode_codec_metadata(compressor),
            fill_value=fill_value,
            extensions=[],
            attributes=attrs,
        )

        # serialise and store metadata document
        meta_doc = _json_encode(meta)
        if path == '/':
            # special case root path
            meta_key = 'meta/root.array'
        else:
            meta_key = f'meta/root{path}.array'
        self.store[meta_key] = meta_doc

        # instantiate array
        array = Array(store=self.store, path=path, owner=self,
                      shape=shape, dtype=dtype, chunk_shape=chunk_shape,
                      compressor=compressor, fill_value=fill_value,
                      attrs=attrs)

        return array

    def get_array(self, path: str) -> Array:
        _check_path(path)

        # retrieve and parse array metadata document
        if path == '/':
            meta_key = 'meta/root.array'
        else:
            meta_key = f'meta/root{path}.array'
        try:
            meta_doc = self.store[meta_key]
        except KeyError:
            raise NodeNotFoundError(path=path)
        meta = _json_decode(meta_doc)

        # decode and check metadata
        shape = tuple(meta['shape'])
        _check_shape(shape)
        dtype = _check_dtype(meta['data_type'])
        chunk_grid = meta['chunk_grid']
        if chunk_grid['type'] != 'regular':
            raise NotImplementedError
        chunk_shape = tuple(chunk_grid['chunk_shape'])
        _check_chunk_shape(chunk_shape, shape)
        chunk_memory_layout = meta['chunk_memory_layout']
        if chunk_memory_layout != 'C':
            raise NotImplementedError
        compressor = _decode_codec_metadata(meta['compressor'])
        fill_value = meta['fill_value']
        for spec in meta['extensions']:
            if spec['must_understand']:
                raise NotImplementedError(spec)
        attrs = meta['attributes']

        # instantiate array
        a = Array(store=self.store, path=path, owner=self, shape=shape,
                  dtype=dtype, chunk_shape=chunk_shape, compressor=compressor,
                  fill_value=fill_value, attrs=attrs)

        return a

    def get_explicit_group(self, path: str) -> ExplicitGroup:
        _check_path(path)

        # retrieve and parse group metadata document
        if path == '/':
            meta_key = 'meta/root.group'
        else:
            meta_key = f'meta/root{path}.group'
        try:
            meta_doc = self.store[meta_key]
        except KeyError:
            raise NodeNotFoundError(path=path)
        meta = _json_decode(meta_doc)

        # check metadata
        attrs = meta['attributes']

        # instantiate explicit group
        g = ExplicitGroup(store=self.store, path=path, owner=self, attrs=attrs)

        return g

    def get_implicit_group(self, path: str) -> ImplicitGroup:
        _check_path(path)

        # attempt to list directory
        if path == '/':
            key_prefix = 'meta/root/'
        else:
            key_prefix = f'meta/root{path}/'
        result = self.store.list_dir(key_prefix)
        if not (result.contents or result.prefixes):
            raise NodeNotFoundError(path=path)

        # instantiate implicit group
        g = ImplicitGroup(store=self.store, path=path, owner=self)

        return g

    def __getitem__(self, path: str) -> Node:

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
        yield from []  # TODO

    def __repr__(self) -> str:
        return f'<Hierarchy at {repr(self.store)}>'
    
    def list_children(self, path: str) -> List[Dict]:
        _check_path(path)
        
        # attempt to list directory
        if path == '/':
            key_prefix = 'meta/root/'
        else:
            key_prefix = f'meta/root{path}/'
        result = self.store.list_dir(key_prefix)
        
        # compile children
        children = []
        names = set()

        # find explicit children
        for n in sorted(result.contents):
            if n.endswith('.array'):
                node_type = 'array'
                name = n[:-len('.array')]
            elif n.endswith('.group'):
                node_type = 'explicit_group'
                name = n[:-len('.group')]
            else:
                # ignore
                continue
            children.append({'name': name, 'type': node_type})
            names.add(name)

        # find implicit children
        for n in sorted(result.prefixes):
            if n not in names:
                children.append({'name': n, 'type': 'implicit_group'})

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
        return self.path.split('/')[-1]


class Group(Node, Mapping):

    def __init__(self, store: Store, path: str, owner: Hierarchy):
        super().__init__(store=store, path=path, owner=owner)

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __iter__(self) -> Iterator[str]:
        yield from []  # TODO all child names

    def list_children(self) -> List[Dict]:
        return self.owner.list_children(path=self.path)

    def _dereference_path(self, path: str) -> str:
        assert isinstance(path, str)
        if path[0] != '/':
            # treat as relative path
            if self.path == '/':
                # special case root group
                path = f'/{path}'
            else:
                path = f'{self.path}/{path}'
        if len(path) > 1:
            assert path[-1] != '/'
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
        return f'<Group {path}>'


class ImplicitGroup(Group):

    def __init__(self, store: Store, path: str, owner: Hierarchy):
        super().__init__(store=store, path=path, owner=owner)

    def __repr__(self) -> str:
        path = self.path
        return f'<Group {path} (implied)>'


class Array(Node):

    def __init__(self,
                 store: Store,
                 path: str,
                 owner: Hierarchy,
                 shape: Tuple[int, ...],
                 dtype: Any,
                 chunk_shape: Tuple[int, ...],
                 compressor: Optional[Codec],
                 fill_value: Any = None,
                 attrs: Optional[Mapping] = None):
        super().__init__(store=store, path=path, owner=owner)
        self.shape = shape
        self.dtype = dtype
        self.chunk_shape = chunk_shape
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
        out = np.zeros(indexer.shape, dtype=self.dtype, order='C')

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
        suffix = '.'.join(map(str, chunk_coords))
        if self.path == '/':
            # special case array as root node
            key = f'data/{suffix}'
        else:
            key = f'data{self.path}/{suffix}'
        return key

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
        chunk = chunk.reshape(-1, order='A')
        chunk = chunk.reshape(self.chunk_shape, order='C')

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
            if not hasattr(value, 'shape'):
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
                chunk = np.empty(self.chunk_shape, dtype=self.dtype, order='C')
                chunk.fill(value)

            else:

                # ensure array is contiguous
                chunk = value.astype(self.dtype, order='C', copy=False)

        else:
            # partially replace the contents of this chunk

            try:

                # obtain compressed data for chunk
                encoded_chunk_data = self.store[chunk_key]

            except KeyError:

                # chunk not initialized
                if self.fill_value is not None:
                    chunk = np.empty(self.chunk_shape, dtype=self.dtype, order='C')
                    chunk.fill(self.fill_value)
                else:
                    # N.B., use zeros here so any region beyond the array has consistent
                    # and compressible data
                    chunk = np.zeros(self.chunk_shape, dtype=self.dtype, order='C')

            else:

                # decode chunk
                chunk = self._decode_chunk(encoded_chunk_data)
                if not chunk.flags.writeable:
                    chunk = chunk.copy(order='K')

            # modify
            chunk[chunk_selection] = value

        # encode chunk
        encoded_chunk_data = self._encode_chunk(chunk)

        # store
        self.store[chunk_key] = encoded_chunk_data

    def _encode_chunk(self, chunk):

        # compress
        if self.compressor is not None:
            encoded_chunk_data = self.compressor.encode(chunk)
        else:
            encoded_chunk_data = chunk

        return encoded_chunk_data

    def __repr__(self):
        path = self.path
        return f'<Array {path}>'


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
        raise TypeError('expected slice or tuple of slices, found %r' % item)


def _ensure_tuple(v):
    if not isinstance(v, tuple):
        v = (v,)
    return v


def _err_too_many_indices(selection, shape):
    raise IndexError('too many indices for array; expected {}, got {}'
                     .format(len(shape), len(selection)))


def _err_boundscheck(dim_len):
    raise IndexError('index out of bounds for dimension with length {}'
                     .format(dim_len))


def _err_negative_step():
    raise IndexError('only slices with step >= 1 are supported')


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
                raise IndexError('unsupported selection item for basic indexing; '
                                 'expected integer or slice, got {!r}'
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

    def __iter__(self) -> Iterator[str]:
        raise NotImplementedError

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def list_pre(self, prefix: str) -> Iterator[str]:
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
        self.root = root.rstrip('/')

    def __getitem__(self, key: str, default: Optional[bytes] = None) -> bytes:
        assert isinstance(key, str)
        path = f'{self.root}/{key}'

        try:
            value = self.fs.cat(path)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            if default is not None:
                return default
            raise KeyError(key)

        return value

    def __setitem__(self, key: str, value: bytes) -> None:
        assert isinstance(key, str)
        path = f'{self.root}/{key}'

        # ensure parent folder exists
        # noinspection PyProtectedMember
        self.fs.mkdirs(self.fs._parent(path), exist_ok=True)

        # write data
        with self.fs.open(path, 'wb') as f:
            f.write(value)

    def __delitem__(self, key: str) -> None:
        assert isinstance(key, str)
        # TODO
        pass

    def __iter__(self) -> Iterator[str]:
        yield from []  # TODO

    def list_pre(self, prefix: str) -> Iterator[str]:
        raise NotImplementedError

    def list_dir(self, prefix: str = '') -> ListDirResult:
        assert isinstance(prefix, str)

        # setup result
        contents: List[str] = []
        prefixes: List[str] = []

        # attempt to list directory
        path = f'{self.root}/{prefix}'
        try:
            ls = self.fs.ls(path, detail=True)
        except FileNotFoundError:
            return ListDirResult(contents=contents, prefixes=prefixes)

        # build result
        for item in ls:
            name = item['name'].split(path)[1]
            if item['type'] == 'file':
                contents.append(name)
            elif item['type'] == 'directory':
                prefixes.append(name)

        return ListDirResult(contents=contents, prefixes=prefixes)

    def __repr__(self) -> str:
        protocol = self.fs.protocol
        if isinstance(protocol, tuple):
            protocol = protocol[-1]
        return f'{protocol}://{self.root}'
