**Here be dragons.**

Ensure blank slate:

```
>>> import shutil
>>> shutil.rmtree('test.zr3', ignore_errors=True)
 
```

Create a new hierarchy stored on the local file system:

```
>>> import zarr_v3
>>> h = zarr_v3.create_hierarchy('test.zr3')
>>> h
<zarr_v3 Hierarchy>
>>> from sh import tree, cat
>>> tree('test.zr3', '-n')
test.zr3
└── zarr.json
<BLANKLINE>
0 directories, 1 file
<BLANKLINE>

>>> cat('test.zr3/zarr.json')
{
    "zarr_format": "https://purl.org/zarr/spec/protocol/core/3.0",
    "metadata_encoding": "application/json",
    "extensions": []
}

```

Access a previously created hierarchy:

```
>>> h = zarr_v3.get_hierarchy('test.zr3')
>>> h
<zarr_v3 Hierarchy>

```

Create an array:

```
>>> a = h.create_array('/arthur/dent', shape=(100, 10), dtype='i4', chunk_shape=(20, 5), compressor=None, attrs={'question': 'life', 'answer': 42})
>>> a
<zarr_v3 Array /arthur/dent>
>>> a.path
'/arthur/dent'
>>> a.ndim
2
>>> a.shape
(100, 10)
>>> a.dtype
dtype('int32')
>>> a.chunk_shape
(20, 5)
>>> a.compressor is None
True
>>> a.attrs
{'question': 'life', 'answer': 42}
>>> tree('test.zr3', '-n')
test.zr3
├── meta
│   └── root
│       └── arthur
│           └── dent.array
└── zarr.json
<BLANKLINE>
3 directories, 2 files
<BLANKLINE>

>>> cat('test.zr3/meta/root/arthur/dent.array')
{
    "shape": [
        100,
        10
    ],
    "data_type": "<i4",
    "chunk_grid": {
        "type": "regular",
        "chunk_shape": [
            20,
            5
        ]
    },
    "chunk_memory_layout": "C",
    "compressor": null,
    "fill_value": null,
    "extensions": [],
    "attributes": {
        "question": "life",
        "answer": 42
    }
}

```

Create a group:

```
>>> g = h.create_group('/tricia/mcmillan', attrs={'heart': 'gold', 'improbability': 'infinite'})
>>> g
<zarr_v3 Group /tricia/mcmillan>
>>> g.attrs
{'heart': 'gold', 'improbability': 'infinite'}
>>> tree('test.zr3', '-n')
test.zr3
├── meta
│   └── root
│       ├── arthur
│       │   └── dent.array
│       └── tricia
│           └── mcmillan.group
└── zarr.json
<BLANKLINE>
4 directories, 3 files
<BLANKLINE>

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

```
>>> a = h['/arthur/dent']
>>> a
<zarr_v3 Array /arthur/dent>
>>> a.shape
(100, 10)
>>> a.dtype
dtype('int32')
>>> a.chunk_shape
(20, 5)
>>> a.compressor is None
True
>>> a.attrs
{'question': 'life', 'answer': 42}

```

Access an explicit group:

```
>>> g = h['/tricia/mcmillan']
>>> g
<zarr_v3 Group /tricia/mcmillan>
>>> g.attrs
{'heart': 'gold', 'improbability': 'infinite'}

```

Access implicit groups:

```
>>> h['/']
<zarr_v3 Group / (implied)>
>>> h['/arthur']
<zarr_v3 Group /arthur (implied)>
>>> h['/tricia']
<zarr_v3 Group /tricia (implied)>

```

Access nodes via groups:

```
>>> root = h['/']
>>> root
<zarr_v3 Group / (implied)>
>>> arthur = root['arthur']
>>> arthur
<zarr_v3 Group /arthur (implied)>
>>> arthur['dent']
<zarr_v3 Array /arthur/dent>
>>> tricia = root['tricia']
>>> tricia
<zarr_v3 Group /tricia (implied)>
>>> tricia['mcmillan']
<zarr_v3 Group /tricia/mcmillan>

```

Explore the hierarchy:

```
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

```
>>> root = h['/']
>>> root
<zarr_v3 Group / (implied)>
>>> root.list_children()
[{'name': 'arthur', 'type': 'implicit_group'}, {'name': 'tricia', 'type': 'implicit_group'}]
>>> root['tricia'].list_children()
[{'name': 'mcmillan', 'type': 'explicit_group'}]
>>> root['tricia']['mcmillan'].list_children()
[]
>>> root['arthur'].list_children()
[{'name': 'dent', 'type': 'array'}]

```