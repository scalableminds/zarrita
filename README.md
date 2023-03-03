# zarrita

**Here be dragons.** Zarrita is a minimal, exploratory implementation of the [Zarr version 3.0 core protocol](https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html). Includes an implementation of sharding. This is a technical spike only, not for production use.

## Setup

```python
import zarrita
import numpy as np

store = zarrita.FileSystemStore('file://./testdata')
```

## Create an array

```python
a = zarrita.Array.create(
    store,
    'array',
    shape=(6, 10),
    dtype=np.dtype('int32'),
    chunk_shape=(2, 5),
    codecs=[zarrita.Array.gzip_codec(level=1)],
    attributes={'question': 'life', 'answer': 42}
)
a[:, :] = np.ones((6, 10), dtype='int32')
```

## Open an array

```python
a = zarrita.Array.open(store, 'array')
assert a[:, :] == np.ones((5, 10), dtype='int32')
```

## Create an array with sharding

```python
a = zarrita.Array.create(
    '/arthur/dent',
    shape=(16, 16),
    dtype=np.dtype('int32'),
    chunk_shape=(16, 16),
    codecs=[
        zarrita.Array.sharding_codec(
            chunk_shape=(8, 8),
            codecs=[zarrita.Array.gzip_codec(level=1)]
        ),
    ],
)
a[:, :] = np.arange(0, 16 * 16, dtype='int32').reshape((16, 16))
```

# Create a group

```python
g = zarrita.Group.create(store, 'group')
g2 = g.create_group('group2')
a = g2.create_array(
    'array',
    shape=(16, 16),
    dtype=np.dtype('int32'),
    chunk_shape=(16, 16),
)
a[:, :] = np.arange(0, 16 * 16, dtype='int32').reshape((16, 16))
```

# Open a group

```python
g = zarrita.Group.open(store, 'group')
g2 = g['group2']
a = g['group2/array']
assert a[:, :] == np.arange(0, 16 * 16, dtype='int32').reshape((16, 16))
```
