**Here be dragons.** Zarrita is a minimal, exploratory implementation of the [Zarr version 3.0 core protocol](https://zarr-specs.readthedocs.io/en/core-protocol-v3.0-dev/protocol/core/v3.0.html). This is a technical spike only, not for production use.

This README contains a doctest suite to verify basic functionality. Both local and remote file systems are supported via [fsspec](https://filesystem-spec.readthedocs.io/en/latest/).

Ensure blank slate:

```python
>>> import shutil
>>> shutil.rmtree('test.zr3', ignore_errors=True)
 
```

## Create a hierarchy

Create a new hierarchy stored on the local file system:

```python
>>> import zarrita
>>> h = zarrita.create_hierarchy('test.zr3')
>>> h  # doctest: +ELLIPSIS
<Hierarchy at file://.../test.zr3>
>>> from sh import tree, cat
>>> tree('test.zr3', '-n', '--noreport')  # doctest: +NORMALIZE_WHITESPACE
test.zr3
└── zarr.json
>>> cat('test.zr3/zarr.json')
{
    "zarr_format": "https://purl.org/zarr/spec/protocol/core/3.0",
    "metadata_encoding": "https://purl.org/zarr/spec/protocol/core/3.0",
    "metadata_key_suffix": ".json",
    "extensions": []
}

```

## Open a hierarchy

Access a previously created hierarchy:

```python
>>> h = zarrita.get_hierarchy('test.zr3')
>>> h  # doctest: +ELLIPSIS
<Hierarchy at file://.../test.zr3>

```

## Create an array

```python
>>> from numcodecs import GZip
>>> compressor = GZip(level=1)
>>> attrs = {'question': 'life', 'answer': 42}
>>> a = h.create_array('/arthur/dent', shape=(5, 10), dtype='i4', chunk_shape=(2, 5), compressor=compressor, attrs=attrs)
>>> a
<Array /arthur/dent>
>>> a.path
'/arthur/dent'
>>> a.name
'dent'
>>> a.ndim
2
>>> a.shape
(5, 10)
>>> a.dtype
dtype('int32')
>>> a.chunk_shape
(2, 5)
>>> a.compressor
GZip(level=1)
>>> a.attrs
{'question': 'life', 'answer': 42}
>>> tree('test.zr3', '-n', '--noreport')  # doctest: +NORMALIZE_WHITESPACE
test.zr3
├── meta
│   └── root
│       └── arthur
│           └── dent.array.json
└── zarr.json
>>> cat('test.zr3/meta/root/arthur/dent.array.json')
{
    "shape": [
        5,
        10
    ],
    "data_type": "<i4",
    "chunk_grid": {
        "type": "regular",
        "chunk_shape": [
            2,
            5
        ],
        "separator": "/"
    },
    "chunk_memory_layout": "C",
    "fill_value": null,
    "extensions": [],
    "attributes": {
        "question": "life",
        "answer": 42
    },
    "compressor": {
        "codec": "https://purl.org/zarr/spec/codec/gzip/1.0",
        "configuration": {
            "level": 1
        }
    }
}

```

## Create an array with no compressor

```python
>>> a = h.create_array('/deep/thought', shape=7_500_000, dtype='>f2', chunk_shape=42, compressor=None)
>>> a
<Array /deep/thought>
>>> a.compressor is None
True
>>> a.attrs
{}
>>> tree('test.zr3', '-n', '--noreport')  # doctest: +NORMALIZE_WHITESPACE
test.zr3
├── meta
│   └── root
│       ├── arthur
│       │   └── dent.array.json
│       └── deep
│           └── thought.array.json
└── zarr.json
>>> cat('test.zr3/meta/root/deep/thought.array.json')
{
    "shape": [
        7500000
    ],
    "data_type": ">f2",
    "chunk_grid": {
        "type": "regular",
        "chunk_shape": [
            42
        ],
        "separator": "/"
    },
    "chunk_memory_layout": "C",
    "fill_value": null,
    "extensions": [],
    "attributes": {}
}

## Create a group

```python
>>> attrs = {'heart': 'gold', 'improbability': 'infinite'}
>>> g = h.create_group('/tricia/mcmillan', attrs=attrs)
>>> g
<Group /tricia/mcmillan>
>>> g.path
'/tricia/mcmillan'
>>> g.name
'mcmillan'
>>> g.attrs
{'heart': 'gold', 'improbability': 'infinite'}
>>> tree('test.zr3', '-n', '--noreport')  # doctest: +NORMALIZE_WHITESPACE
test.zr3
├── meta
│   └── root
│       ├── arthur
│       │   └── dent.array.json
│       ├── deep
│       │   └── thought.array.json
│       └── tricia
│           └── mcmillan.group.json
└── zarr.json
>>> cat('test.zr3/meta/root/tricia/mcmillan.group.json')
{
    "extensions": [],
    "attributes": {
        "heart": "gold",
        "improbability": "infinite"
    }
}

```


## Create nodes via groups

```python
>>> h.create_group('marvin')
<Group /marvin>
>>> h['marvin'].create_group('paranoid')
<Group /marvin/paranoid>
>>> h['marvin'].create_array('android', shape=(42, 42), dtype=bool, chunk_shape=(2, 2))
<Array /marvin/android>
>>> tree('test.zr3', '-n', '--noreport')  # doctest: +NORMALIZE_WHITESPACE
test.zr3
├── meta
│   └── root
│       ├── arthur
│       │   └── dent.array.json
│       ├── deep
│       │   └── thought.array.json
│       ├── marvin
│       │   ├── android.array.json
│       │   └── paranoid.group.json
│       ├── marvin.group.json
│       └── tricia
│           └── mcmillan.group.json
└── zarr.json

```


## Access an array

```python
>>> a = h['/arthur/dent']
>>> a
<Array /arthur/dent>
>>> a.shape
(5, 10)
>>> a.dtype
dtype('int32')
>>> a.chunk_shape
(2, 5)
>>> a.compressor
GZip(level=1)
>>> a.attrs
{'question': 'life', 'answer': 42}

```

## Access an explicit group

```python
>>> g = h['/tricia/mcmillan']
>>> g
<Group /tricia/mcmillan>
>>> g.attrs
{'heart': 'gold', 'improbability': 'infinite'}

```

## Access implicit groups

```python
>>> h['/']
<Group / (implied)>
>>> h['/arthur']
<Group /arthur (implied)>
>>> h['/tricia']
<Group /tricia (implied)>

```

## Access nodes via groups

```python
>>> root = h['/']
>>> root
<Group / (implied)>
>>> arthur = root['arthur']
>>> arthur
<Group /arthur (implied)>
>>> arthur['dent']
<Array /arthur/dent>
>>> tricia = root['tricia']
>>> tricia
<Group /tricia (implied)>
>>> tricia['mcmillan']
<Group /tricia/mcmillan>

```


## Access nodes - convenience

Zarrita treats a hierarchy and its root node as separate objects. E.g.:

```python
>>> h  # doctest: +ELLIPSIS
<Hierarchy at file://.../test.zr3>
>>> h['/']
<Group / (implied)>
>>> h.root  # alternative way to access the root node
<Group / (implied)>

```

For convenience and familiarity, any relative path accessed via a 
hierarchy is treated as relative to the root group. E.g.:

```python
>>> h['/arthur']
<Group /arthur (implied)>
>>> h['arthur']
<Group /arthur (implied)>
>>> h['/tricia/mcmillan']
<Group /tricia/mcmillan>
>>> h['tricia']['mcmillan']
<Group /tricia/mcmillan>

```

## Explore the hierarchy

Explore the hierarchy top-down:

```python
>>> h.get_children('/')  # doctest: +NORMALIZE_WHITESPACE
{'arthur': 'implicit_group', 
 'deep': 'implicit_group',
 'marvin': 'explicit_group', 
 'tricia': 'implicit_group'}
>>> h.get_children('/tricia')
{'mcmillan': 'explicit_group'}
>>> h.get_children('/tricia/mcmillan')
{}
>>> h.get_children('/arthur')
{'dent': 'array'}

```

Alternative way to explore the hierarchy:

```python
>>> h.get_children()  # doctest: +NORMALIZE_WHITESPACE
{'arthur': 'implicit_group', 
 'deep': 'implicit_group',
 'marvin': 'explicit_group', 
 'tricia': 'implicit_group'}
>>> h['tricia'].get_children()
{'mcmillan': 'explicit_group'}
>>> h['tricia']['mcmillan'].get_children()
{}
>>> h['arthur'].get_children()
{'dent': 'array'}

```

View the whole hierarchy in one go:

```python
>>> h.get_nodes()  # doctest: +NORMALIZE_WHITESPACE
{'/': 'implicit_group',
 '/arthur': 'implicit_group',
 '/arthur/dent': 'array', 
 '/deep': 'implicit_group',
 '/deep/thought': 'array',
 '/marvin': 'explicit_group', 
 '/marvin/android': 'array', 
 '/marvin/paranoid': 'explicit_group', 
 '/tricia': 'implicit_group', 
 '/tricia/mcmillan': 'explicit_group'}

```


## Check existence of nodes in a hierarchy

```python
>>> '/' in h
True
>>> '/arthur' in h
True
>>> '/arthur/dent' in h
True
>>> '/zaphod' in h
False
>>> '/zaphod/beeblebrox' in h
False
>>> '/tricia' in h
True
>>> '/tricia/mcmillan' in h
True

```


## Check existence of children in a group

```python
>>> 'arthur' in h.root
True
>>> 'tricia' in h.root
True
>>> 'zaphod' in h.root
False
>>> g = h.root['arthur']
>>> 'dent' in g
True
>>> g = h.root['tricia']
>>> 'mcmillan' in g
True
>>> 'beeblebrox' in g
False

```

## Iterating over a hierarchy or group

Iterating over a group returns names of all child nodes:

```python
>>> sorted(h['/'])
['arthur', 'deep', 'marvin', 'tricia']
>>> sorted(h['/arthur'])
['dent']
>>> sorted(h['/tricia'])
['mcmillan']
>>> sorted(h['/tricia/mcmillan'])
[]
>>> sorted(h['/marvin'])
['android', 'paranoid']

```

Iterate over a hierarchy returns paths for all explicit nodes:

```python
>>> sorted(h)  # doctest: +NORMALIZE_WHITESPACE
['/', 
 '/arthur', 
 '/arthur/dent',
 '/deep',
 '/deep/thought', 
 '/marvin', 
 '/marvin/android', 
 '/marvin/paranoid', 
 '/tricia', 
 '/tricia/mcmillan']

```


## Read and write array data

```python
>>> import numpy as np
>>> a = h['/arthur/dent']
>>> a
<Array /arthur/dent>
>>> tree('test.zr3', '-n', '--noreport')  # doctest: +NORMALIZE_WHITESPACE
test.zr3
├── meta
│   └── root
│       ├── arthur
│       │   └── dent.array.json
│       ├── deep
│       │   └── thought.array.json
│       ├── marvin
│       │   ├── android.array.json
│       │   └── paranoid.group.json
│       ├── marvin.group.json
│       └── tricia
│           └── mcmillan.group.json
└── zarr.json
>>> a[:, :]
array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int32)
>>> a[...]
array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int32)
>>> a[:]
array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int32)
>>> a[0, :] = 42
>>> a[:]
array([[42, 42, 42, 42, 42, 42, 42, 42, 42, 42],
       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0]], dtype=int32)
>>> tree('test.zr3/data/root/arthur/dent', '-n', '--noreport')  # doctest: +NORMALIZE_WHITESPACE
test.zr3/data/root/arthur/dent
└── c0
    ├── 0
    └── 1
>>> a[:, 0] = 42
>>> a[:]
array([[42, 42, 42, 42, 42, 42, 42, 42, 42, 42],
       [42,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       [42,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       [42,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       [42,  0,  0,  0,  0,  0,  0,  0,  0,  0]], dtype=int32)
>>> tree('test.zr3/data/root/arthur/dent', '-n', '--noreport')  # doctest: +NORMALIZE_WHITESPACE
test.zr3/data/root/arthur/dent
├── c0
│   ├── 0
│   └── 1
├── c1
│   └── 0
└── c2
    └── 0
>>> a[:] = 42
>>> a[:]
array([[42, 42, 42, 42, 42, 42, 42, 42, 42, 42],
       [42, 42, 42, 42, 42, 42, 42, 42, 42, 42],
       [42, 42, 42, 42, 42, 42, 42, 42, 42, 42],
       [42, 42, 42, 42, 42, 42, 42, 42, 42, 42],
       [42, 42, 42, 42, 42, 42, 42, 42, 42, 42]], dtype=int32)
>>> tree('test.zr3/data/root/arthur/dent', '-n', '--noreport')  # doctest: +NORMALIZE_WHITESPACE
test.zr3/data/root/arthur/dent
├── c0
│   ├── 0
│   └── 1
├── c1
│   ├── 0
│   └── 1
└── c2
    ├── 0
    └── 1
>>> a[0, :] = np.arange(10)
>>> a[:]
array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
       [42, 42, 42, 42, 42, 42, 42, 42, 42, 42],
       [42, 42, 42, 42, 42, 42, 42, 42, 42, 42],
       [42, 42, 42, 42, 42, 42, 42, 42, 42, 42],
       [42, 42, 42, 42, 42, 42, 42, 42, 42, 42]], dtype=int32)
>>> a[:, 0] = np.arange(0, 50, 10)
>>> a[:]
array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
       [10, 42, 42, 42, 42, 42, 42, 42, 42, 42],
       [20, 42, 42, 42, 42, 42, 42, 42, 42, 42],
       [30, 42, 42, 42, 42, 42, 42, 42, 42, 42],
       [40, 42, 42, 42, 42, 42, 42, 42, 42, 42]], dtype=int32)
>>> a[:] = np.arange(50).reshape(5, 10)
>>> a[:]
array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
       [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
       [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
       [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]], dtype=int32)
>>> a[:, 0]
array([ 0, 10, 20, 30, 40], dtype=int32)
>>> a[:, 1]
array([ 1, 11, 21, 31, 41], dtype=int32)
>>> a[0, :]
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int32)
>>> a[1, :]
array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=int32)
>>> a[:, 0:7]
array([[ 0,  1,  2,  3,  4,  5,  6],
       [10, 11, 12, 13, 14, 15, 16],
       [20, 21, 22, 23, 24, 25, 26],
       [30, 31, 32, 33, 34, 35, 36],
       [40, 41, 42, 43, 44, 45, 46]], dtype=int32)
>>> a[0:3, :]
array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
       [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]], dtype=int32)
>>> a[0:3, 0:7]
array([[ 0,  1,  2,  3,  4,  5,  6],
       [10, 11, 12, 13, 14, 15, 16],
       [20, 21, 22, 23, 24, 25, 26]], dtype=int32)
>>> a[1:4, 2:7]
array([[12, 13, 14, 15, 16],
       [22, 23, 24, 25, 26],
       [32, 33, 34, 35, 36]], dtype=int32)
>>> b = h['/deep/thought']  # no compressor
>>> b
<Array /deep/thought>
>>> b[:10]
array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float16)
>>> b[:5] = 1
>>> b[:10]
array([1., 1., 1., 1., 1., 0., 0., 0., 0., 0.], dtype=float16)
>>> tree('test.zr3/data/root/deep/thought', '-n', '--noreport')  # doctest: +NORMALIZE_WHITESPACE
test.zr3/data/root/deep/thought
└── c0
>>> import numpy as np
>>> with open('test.zr3/data/root/deep/thought/c0', mode='rb') as f:
...     np.frombuffer(f.read(), dtype='>f2')[:10]
array([1., 1., 1., 1., 1., 0., 0., 0., 0., 0.], dtype=float16)

```


## Invalid node names

```python
>>> bad_paths = '', '//', '/foo//bar', ' ', '.', '/../', '/foo/./bar', '/foo/../bar', '/ ', 'Καλημέρα', 'a' * 256
>>> for p in bad_paths:
...     try:
...         h.create_group(p)
...     except ValueError:
...         pass

```

## Use cloud storage
 
Read data previously copied to GCS:

```python
>>> cloud_h = zarrita.get_hierarchy('gs://zarr-demo/v3/test.zr3', token='anon')
>>> cloud_h
<Hierarchy at gs://zarr-demo/v3/test.zr3>
>>> sorted(cloud_h)
['/', '/arthur', '/arthur/dent', '/deep', '/deep/thought', '/marvin', '/marvin/android', '/marvin/paranoid', '/tricia', '/tricia/mcmillan']
>>> cloud_h.get_children('/')  # doctest: +NORMALIZE_WHITESPACE
{'arthur': 'implicit_group', 
 'deep': 'implicit_group',
 'marvin': 'explicit_group', 
 'tricia': 'implicit_group'}
>>> cloud_h.get_children('/tricia')
{'mcmillan': 'explicit_group'}
>>> cloud_h.get_children('/tricia/mcmillan')
{}
>>> cloud_h.get_children('/arthur')
{'dent': 'array'}
>>> cloud_h['/']
<Group / (implied)>
>>> cloud_h['/tricia']
<Group /tricia (implied)>
>>> g = cloud_h['/tricia/mcmillan']
>>> g
<Group /tricia/mcmillan>
>>> g.attrs
{'heart': 'gold', 'improbability': 'infinite'}
>>> a = cloud_h['/arthur/dent']
>>> a
<Array /arthur/dent>
>>> a.shape
(5, 10)
>>> a.dtype
dtype('int32')
>>> a.attrs
{'question': 'life', 'answer': 42}
>>> a[:]
array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
       [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
       [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
       [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]], dtype=int32)

```