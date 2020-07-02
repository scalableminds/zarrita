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


def _check_path(path):
    assert isinstance(path, str)
    assert path[0] == '/'
    if len(path) > 1:
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


def _check_chunk_shape(chunk_shape, shape):
    assert isinstance(chunk_shape, tuple)
    assert all([isinstance(c, int) for c in chunk_shape])
    assert len(chunk_shape) == len(shape)


def _check_compressor(compressor):
    assert compressor is None or isinstance(compressor, numcodecs.abc.Codec)


def _encode_codec(codec):
    if codec is None:
        return None

    # only support gzip for now
    assert codec.codec_id in {'gzip'}
    config = codec.get_config()
    del config['codec_id']
    meta = {
        'codec': 'https://purl.org/zarr/spec/codec/gzip/1.0',
        'configuration': config,
    }
    return meta


def _decode_codec(meta):
    if meta is None:
        return None

    # only support gzip for now
    if meta['codec'] != 'https://purl.org/zarr/spec/codec/gzip/1.0':
        raise NotImplementedError
    codec = numcodecs.GZip(level=meta['level'])
    return codec


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
            meta_key = 'meta/root.group'
        else:
            meta_key = f'meta/root{path}.group'
        self.store[meta_key] = meta_doc

        # instantiate group
        group = ExplicitGroup(store=self.store, path=path, owner=self, 
                              attrs=attrs)

        return group

    def create_array(self, path, shape, dtype, chunk_shape, compressor,
                     fill_value=None, attrs=None):

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
        meta = {
            'shape': shape,
            'data_type': data_type,
            'chunk_grid': {
                'type': 'regular',
                'chunk_shape': chunk_shape,
            },
            'chunk_memory_layout': 'C',
            'compressor': _encode_codec(compressor),
            'fill_value': fill_value,
            'extensions': [],
            'attributes': attrs,
        }

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

    def get_array(self, path):
        _check_path(path)

        # retrieve and parse array metadata document
        if path == '/':
            meta_key = 'meta/root.array'
        else:
            meta_key = f'meta/root{path}.array'
        try:
            meta_doc = self.store[meta_key]
        except KeyError:
            raise NodeNotFoundError(path)
        meta = _json_decode(meta_doc)

        # decode and check metadata
        shape = tuple(meta['shape'])
        dtype = np.dtype(meta['data_type'])
        chunk_grid = meta['chunk_grid']
        if chunk_grid['type'] != 'regular':
            raise NotImplementedError
        chunk_shape = tuple(chunk_grid['chunk_shape'])
        chunk_memory_layout = meta['chunk_memory_layout']
        if chunk_memory_layout != 'C':
            raise NotImplementedError
        compressor = _decode_codec(meta['compressor'])
        fill_value = meta['fill_value']
        for spec in meta['extensions']:
            if spec['must_understand']:
                raise NotImplementedError
        attrs = meta['attributes']

        # instantiate array
        a = Array(store=self.store, path=path, owner=self, shape=shape,
                  dtype=dtype, chunk_shape=chunk_shape, compressor=compressor,
                  fill_value=fill_value, attrs=attrs)

        return a

    def get_explicit_group(self, path):
        _check_path(path)

        # retrieve and parse group metadata document
        if path == '/':
            meta_key = 'meta/root.group'
        else:
            meta_key = f'meta/root{path}.group'
        try:
            meta_doc = self.store[meta_key]
        except KeyError:
            raise NodeNotFoundError(path)
        meta = _json_decode(meta_doc)

        # check metadata
        attrs = meta['attributes']

        # instantiate explicit group
        g = ExplicitGroup(store=self.store, path=path, owner=self, attrs=attrs)

        return g

    def get_implicit_group(self, path):
        _check_path(path)

        # attempt to list directory
        if path == '/':
            key_prefix = 'meta/root/'
        else:
            key_prefix = f'meta/root{path}/'
        result = self.store.list_dir(key_prefix)
        if not (result['contents'] or result['prefixes']):
            raise NodeNotFoundError(path)

        # instantiate implicit group
        g = ImplicitGroup(store=self.store, path=path, owner=self)

        return g

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
    
    def list_children(self, path):
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
        for n in sorted(result['contents']):
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
        for n in sorted(result['prefixes']):
            if n not in names:
                children.append({'name': n, 'type': 'implicit_group'})

        return children


class NodeNotFoundError(Exception):

    def __init__(self, path):
        self.path = path


class Group:

    def __init__(self, store, path, owner):
        self.store = store
        self.path = path
        self.owner = owner

    def list_children(self):
        return self.owner.list_children(self.path)

    def _dereference_path(self, path):
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

    def __getitem__(self, path):
        path = self._dereference_path(path)
        return self.owner[path]

    def create_group(self, path, **kwargs):
        path = self._dereference_path(path)
        return self.owner.create_group(self.path + path, **kwargs)

    def create_array(self, path, **kwargs):
        path = self._dereference_path(path)
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
        return f'<zarr_v3 Group {path} (implied)>'


class Array:

    # TODO persist changes to attrs

    def __init__(self, store, path, owner, shape, dtype, chunk_shape,
                 compressor, fill_value, attrs):
        self.store = store
        self.path = path
        self.owner = owner
        self.shape = shape
        self.dtype = dtype
        self.chunk_shape = chunk_shape
        self.compressor = compressor
        self.fill_value = fill_value
        self.attrs = attrs

    @property
    def ndim(self):
        return len(self.shape)

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

    # TODO ultimately replace this with the fsspec FSMap class, but for now roll
    # our own implementation in order to be able to add some extra methods for
    # listing keys.

    def __init__(self, url, **kwargs):
        assert isinstance(url, str)

        # instantiate file system
        fs, root = fsspec.core.url_to_fs(url, **kwargs)
        self.fs = fs
        self.root = root.rstrip('/')

    def __getitem__(self, key, default=None):
        assert isinstance(key, str)
        path = f'{self.root}/{key}'

        try:
            value = self.fs.cat(path)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            if default is not None:
                return default
            raise KeyError(key)

        return value

    def __setitem__(self, key, value):
        assert isinstance(key, str)
        assert isinstance(value, bytes)
        path = f'{self.root}/{key}'

        # ensure parent folder exists
        # noinspection PyProtectedMember
        self.fs.mkdirs(self.fs._parent(path), exist_ok=True)

        # write data
        with self.fs.open(path, 'wb') as f:
            f.write(value)

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

    def list_dir(self, prefix=''):
        assert isinstance(prefix, str)

        # setup result
        result = {
            'contents': [],
            'prefixes': [],
        }

        # attempt to list directory
        path = f'{self.root}/{prefix}'
        try:
            ls = self.fs.ls(path, detail=True)
        except FileNotFoundError:
            return result

        # build result
        for item in ls:
            name = item['name'].split(path)[1]
            if item['type'] == 'file':
                result['contents'].append(name)
            elif item['type'] == 'directory':
                result['prefixes'].append(name)

        return result
