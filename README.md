**Here be dragons.** Zarrita is a minimal, exploratory implementation of the [Zarr version 3.0 core protocol](https://zarr-specs.readthedocs.io/en/core-protocol-v3.0-dev/protocol/core/v3.0.html). This is a technical spike only, not for production use.

This README contains a doctest suite to verify basic functionality using local file system storage. 

Ensure blank slate:

```python
>>> import shutil
>>> shutil.rmtree('test.zr3', ignore_errors=True)
 
```

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
    "metadata_encoding": "application/json",
    "extensions": []
}

```

Access a previously created hierarchy:

```python
>>> h = zarrita.get_hierarchy('test.zr3')
>>> h  # doctest: +ELLIPSIS
<Hierarchy at file://.../test.zr3>

```

Create an array:

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
│           └── dent.array
└── zarr.json
>>> cat('test.zr3/meta/root/arthur/dent.array')
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
        ]
    },
    "chunk_memory_layout": "C",
    "compressor": {
        "codec": "https://purl.org/zarr/spec/codec/gzip/1.0",
        "configuration": {
            "level": 1
        }
    },
    "fill_value": null,
    "extensions": [],
    "attributes": {
        "question": "life",
        "answer": 42
    }
}

```

Create a group:

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
│       │   └── dent.array
│       └── tricia
│           └── mcmillan.group
└── zarr.json
>>> cat('test.zr3/meta/root/tricia/mcmillan.group')
{
    "extensions": [],
    "attributes": {
        "heart": "gold",
        "improbability": "infinite"
    }
}

```

Access an array:

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

Access an explicit group:

```python
>>> g = h['/tricia/mcmillan']
>>> g
<Group /tricia/mcmillan>
>>> g.attrs
{'heart': 'gold', 'improbability': 'infinite'}

```

Access implicit groups:

```python
>>> h['/']
<Group / (implied)>
>>> h['/arthur']
<Group /arthur (implied)>
>>> h['/tricia']
<Group /tricia (implied)>

```

Access nodes via groups:

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

Explore the hierarchy:

```python
>>> h.list_children('/')
[{'name': 'arthur', 'type': 'implicit_group'}, {'name': 'tricia', 'type': 'implicit_group'}]
>>> h.list_children('/tricia')
[{'name': 'mcmillan', 'type': 'explicit_group'}]
>>> h.list_children('/tricia/mcmillan')
[]
>>> h.list_children('/arthur')
[{'name': 'dent', 'type': 'array'}]

```

Alternative way to explore the hierarchy:

```python
>>> root = h['/']
>>> root
<Group / (implied)>
>>> root.list_children()
[{'name': 'arthur', 'type': 'implicit_group'}, {'name': 'tricia', 'type': 'implicit_group'}]
>>> root['tricia'].list_children()
[{'name': 'mcmillan', 'type': 'explicit_group'}]
>>> root['tricia']['mcmillan'].list_children()
[]
>>> root['arthur'].list_children()
[{'name': 'dent', 'type': 'array'}]

```

Read and write data into an array:

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
│       │   └── dent.array
│       └── tricia
│           └── mcmillan.group
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
>>> tree('test.zr3', '-n', '--noreport')  # doctest: +NORMALIZE_WHITESPACE
test.zr3
├── data
│   └── arthur
│       └── dent
│           ├── 0.0
│           └── 0.1
├── meta
│   └── root
│       ├── arthur
│       │   └── dent.array
│       └── tricia
│           └── mcmillan.group
└── zarr.json
>>> a[:, 0] = 42
>>> a[:]
array([[42, 42, 42, 42, 42, 42, 42, 42, 42, 42],
       [42,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       [42,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       [42,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       [42,  0,  0,  0,  0,  0,  0,  0,  0,  0]], dtype=int32)
>>> tree('test.zr3', '-n', '--noreport')  # doctest: +NORMALIZE_WHITESPACE
test.zr3
├── data
│   └── arthur
│       └── dent
│           ├── 0.0
│           ├── 0.1
│           ├── 1.0
│           └── 2.0
├── meta
│   └── root
│       ├── arthur
│       │   └── dent.array
│       └── tricia
│           └── mcmillan.group
└── zarr.json
>>> a[:] = 42
>>> a[:]
array([[42, 42, 42, 42, 42, 42, 42, 42, 42, 42],
       [42, 42, 42, 42, 42, 42, 42, 42, 42, 42],
       [42, 42, 42, 42, 42, 42, 42, 42, 42, 42],
       [42, 42, 42, 42, 42, 42, 42, 42, 42, 42],
       [42, 42, 42, 42, 42, 42, 42, 42, 42, 42]], dtype=int32)
>>> tree('test.zr3', '-n', '--noreport')  # doctest: +NORMALIZE_WHITESPACE
test.zr3
├── data
│   └── arthur
│       └── dent
│           ├── 0.0
│           ├── 0.1
│           ├── 1.0
│           ├── 1.1
│           ├── 2.0
│           └── 2.1
├── meta
│   └── root
│       ├── arthur
│       │   └── dent.array
│       └── tricia
│           └── mcmillan.group
└── zarr.json
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

```
