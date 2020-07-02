import json
import fsspec
import numpy as np
import numcodecs
from numcodecs.abc import Codec


def _json_encode(o):
    s = json.dumps(o, ensure_ascii=False, allow_nan=False, indent=4,
                   sort_keys=False)
    b = s.encode('utf8')
    return b


def _json_decode(b):
    assert isinstance(b, bytes)
    o = json.loads(b)
    return o


def _check_store(store):

    # if store arg is a string, assume it's an fsspec-style URL
    if isinstance(store, str):
        store = FileSystemStore(store)

    assert isinstance(store, Store)

    return store


def create_hierarchy(store):

    # sanity checks
    store = _check_store(store)

    # create entry point metadata document
    meta = {
        'zarr_format': 'https://purl.org/zarr/spec/protocol/core/3.0',
        'metadata_encoding': 'application/json',
        'extensions': [],
    }

    # serialise and store metadata document
    meta_doc = _json_encode(meta)
    meta_key = 'zarr.json'
    store[meta_key] = meta_doc

    # instantiate a hierarchy
    hierarchy = Hierarchy(store=store)

    return hierarchy


def get_hierarchy(store):

    # sanity checks
    store = _check_store(store)

    # retrieve and parse entry point metadata document
    meta_key = 'zarr.json'
    meta_doc = store[meta_key]
    meta = _json_decode(meta_doc)

    # check protocol version
    zarr_format = meta['zarr_format']
    protocol_version = zarr_format.split('/')[-1]
    protocol_major_version = protocol_version.split('.')[0]
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


def _check_path(path):
    assert isinstance(path, str)
    assert path[0] == '/'
    assert path[-1] != '/'


def _check_attrs(attrs):
    assert attrs is None or isinstance(attrs, dict)


def _check_shape(shape):
    assert isinstance(shape, tuple)
    assert all([isinstance(s, int) for s in shape])


def _check_dtype(dtype):
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


def _check_chunks(chunks, shape):
    assert isinstance(chunks, tuple)
    assert all([isinstance(c, int) for c in chunks])
    assert len(chunks) == len(shape)


def _check_compressor(compressor):
    assert isinstance(compressor, numcodecs.abc.Codec)


def _get_codec_metadata(codec):
    # only support gzip for now
    assert codec.codec_id in {'gzip'}
    config = codec.get_config()
    del config['codec_id']
    meta = {
        'codec': 'https://purl.org/zarr/spec/codec/gzip/1.0',
        'configuration': config,
    }
    return meta


class Hierarchy:

    def __init__(self, store):
        self.store = store

    def create_group(self, path, attrs=None):

        # sanity checks
        _check_path(path)
        _check_attrs(attrs)

        # create group metadata
        meta = {
            'extensions': [],
            'attributes': attrs,
        }

        # serialise and store metadata document
        meta_doc = _json_encode(meta)
        if path == '/':
            # special case root path
            meta_key_suffix = 'root.group'
        else:
            meta_key_suffix = 'root' + path + '.group'
        meta_key = 'meta/' + meta_key_suffix
        self.store[meta_key] = meta_doc

        # instantiate group
        group = ExplicitGroup(store=self.store, path=path, owner=self, 
                              attrs=attrs)

        return group

    def create_array(self, path, shape, dtype, chunks, compressor, attrs=None):

        # sanity checks
        _check_path(path)
        _check_shape(shape)
        dtype = _check_dtype(dtype)
        _check_chunks(chunks, shape)
        _check_compressor(compressor)
        _check_attrs(attrs)

        # encode data type
        if dtype == np.bool:
            data_type = 'bool'
        else:
            data_type = dtype.str

        # create array metadata
        meta = {
            'shape': shape,
            'data_type': data_type,
            'chunk_grid': {
                'type': 'regular',
                'chunk_shape': chunks,
            },
            'chunk_memory_layout': 'C',
            'compressor': _get_codec_metadata(compressor),
            'fill_value': None,
            'extensions': [],
            'attributes': attrs,
        }

        # serialise and store metadata document
        meta_doc = _json_encode(meta)
        if path == '/':
            # special case root path
            meta_key_suffix = 'root.array'
        else:
            meta_key_suffix = 'root' + path + '.array'
        meta_key = 'meta/' + meta_key_suffix
        self.store[meta_key] = meta_doc

        # instantiate array
        array = Array(store=self.store, path=path, owner=self,
                      shape=shape, dtype=dtype, chunks=chunks,
                      compressor=compressor, attrs=attrs)

        return array

    def get_array(self, path):
        _check_path(path)

        # TODO retrieve and parse array metadata document

        # TODO decode and check metadata

        # TODO instantiate array

        pass

    def get_explicit_group(self, path):
        _check_path(path)

        # TODO retrieve and parse group metadata document

        # TODO check metadata

        # TODO instantiate explicit group

        pass

    def get_implicit_group(self, path):
        _check_path(path)

        # TODO attempt to list directory

        # TODO instantiate implicit group

        pass

    def __getitem__(self, path):

        # try array
        try:
            return self.get_array(path)
        except NodeNotFoundError:
            pass

        # try explicit group
        try:
            return self.get_explicit_group(path)
        except NodeNotFoundError:
            pass

        # try implicit group
        try:
            return self.get_implicit_group(path)
        except NodeNotFoundError:
            pass

        raise KeyError(path)

    def __repr__(self):
        return '<zarr_v3 Hierarchy>'


class NodeNotFoundError(Exception):
    pass


class Group:

    def __init__(self, store, path, owner):
        self.store = store
        self.path = path
        self.owner = owner

    def list_children(self, path):
        pass

    def __getitem__(self, path):
        # pass through to owner
        _check_path(path)
        return self.owner[self.path + path]

    def create_group(self, path, **kwargs):
        # pass through to owner
        _check_path(path)
        return self.owner.create_group(self.path + path, **kwargs)

    def create_array(self, path, **kwargs):
        # pass through to owner
        _check_path(path)
        return self.owner.create_array(self.path + path, **kwargs)
    
    def get_array(self, path):
        # pass through to owner
        _check_path(path)
        return self.owner.get_array(self.path + path)

    def get_explicit_group(self, path):
        # pass through to owner
        _check_path(path)
        return self.owner.get_explicit_group(self.path + path)

    def get_implicit_group(self, path):
        # pass through to owner
        _check_path(path)
        return self.owner.get_implicit_group(self.path + path)


class ExplicitGroup(Group):

    # TODO persist changes to attrs

    def __init__(self, store, path, owner, attrs):
        super().__init__(store=store, path=path, owner=owner)
        self.attrs = attrs

    def __repr__(self):
        path = self.path
        return f'<zarr_v3 Group {path}>'


class ImplicitGroup(Group):

    def __init__(self, store, path, owner):
        super().__init__(store=store, path=path, owner=owner)

    def __repr__(self):
        path = self.path
        return f'<zarr_v3 Group (implied) {path}>'


class Array:

    # TODO persist changes to attrs

    def __init__(self, store, path, owner, shape, dtype, chunks,
                 compressor, attrs):
        self.store = store
        self.path = path
        self.owner = owner
        self.shape = shape
        self.dtype = dtype
        self.chunks = chunks
        self.compressor = compressor
        self.attrs = attrs

    def __getitem__(self, item):
        # TODO
        pass

    def __setitem__(self, item, value):
        # TODO
        pass

    def __repr__(self):
        path = self.path
        return f'<zarr_v3 Array {path}>'


class Store:

    def __getitem__(self, key):
        raise NotImplementedError

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __delitem__(self, key):
        raise NotImplementedError

    def keys(self):
        raise NotImplementedError

    def values(self):
        raise NotImplementedError

    def items(self):
        raise NotImplementedError

    def __iter__(self):
        return self.keys()

    def list_pre(self, prefix):
        raise NotImplementedError

    def list_dir(self, prefix):
        raise NotImplementedError


class FileSystemStore(Store):

    def __init__(self, fs, prefix=''):
        if isinstance(fs, str):
            # TODO instantiate file system
            pass
        assert isinstance(prefix, str)
        self.fs = fs
        self.prefix = prefix

    def __getitem__(self, key):
        assert isinstance(key, str)
        # TODO
        pass

    def __setitem__(self, key, value):
        assert isinstance(key, str)
        assert isinstance(value, bytes)
        # TODO
        pass

    def __delitem__(self, key):
        assert isinstance(key, str)
        # TODO
        pass

    def keys(self):
        # TODO
        pass

    def values(self):
        # TODO
        pass

    def items(self):
        # TODO
        pass

    def __iter__(self):
        return self.keys()

    def list_pre(self, prefix):
        assert isinstance(prefix, str)
        # TODO
        pass

    def list_dir(self, prefix):
        assert isinstance(prefix, str)
        # TODO
        pass
