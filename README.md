# Zarrita

Zarrita is an experimental implementation of [Zarr v3](https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html) including [sharding](https://zarr.dev/zeps/draft/ZEP0002.html). This is only a technical proof of concept meant for generating sample datasets. Not recommended for production use.

## Setup

```python
import zarrita
import numpy as np

store = zarrita.LocalStore('testoutput') # or zarrita.RemoteStore('s3://bucket/test')

testdata = np.arange(0, 16 * 16, dtype='int32').reshape((16, 16))
```

## Create an array

```python
a = zarrita.Array.create(
    store / 'array',
    shape=(16, 16),
    dtype='int32',
    chunk_shape=(2, 8),
    codecs=[
        zarrita.codecs.bytes_codec(),
        zarrita.codecs.blosc_codec(typesize=4),
    ],
    attributes={'question': 'life', 'answer': 42}
)
a[:, :] = testdata
```

## Open an array

```python
a = zarrita.Array.open(store / 'array')
assert np.array_equal(a[:, :], testdata)
```

## Create an array with sharding

```python
a = zarrita.Array.create(
    store / 'sharding',
    shape=(16, 16),
    dtype='int32',
    chunk_shape=(16, 16),
    chunk_key_encoding=('v2', '.'),
    codecs=[
        zarrita.codecs.sharding_codec(
            chunk_shape=(8, 8),
            codecs=[
                zarrita.codecs.bytes_codec(),
                zarrita.codecs.blosc_codec(typesize=4),
            ]
        ),
    ],
)
a[:, :] = testdata
assert np.array_equal(a[:, :], testdata)
```

## Create a group

```python
g = zarrita.Group.create(store / 'group')
g2 = g.create_group('group2')
a = g2.create_array(
    'array',
    shape=(16, 16),
    dtype='int32',
    chunk_shape=(16, 16),
)
a[:, :] = testdata
```

## Open a group

```python
g = zarrita.Group.open(store / 'group')
g2 = g['group2']
a = g['group2']['array']
assert np.array_equal(a[:, :], testdata)
```

## Resize array

```python
a.resize((10, 10))
```

## Update attributes

```python
a.update_attributes({'question': 'life', 'answer': 0})
```

## Zarr v2

```python
a = zarrita.ArrayV2.create(
    store / 'array',
    shape=(16, 16),
    dtype='int32',
    chunks=(2, 8),
)
a[:, :] = testdata

a3 = a.convert_to_v3()
assert a3.metadata.shape == a.shape
```

## Async methods

```python
a = await zarrita.Array.create_async(
    store / 'array_async',
    shape=(16, 16),
    dtype='int32',
    chunk_shape=(2, 8),
)
await a.async_[:, :].set(testdata)
assert np.array_equal(await a.async_[:, :].get(), testdata)
```

# Credits

This is a largely-rewritten fork of `zarrita` by [@alimanfoo](https://github.com/alimanfoo). It implements the Zarr v3 draft specification created by [@alimanfoo](https://github.com/alimanfoo), [@jstriebel](https://github.com/jstriebel), [@jbms](https://github.com/jbms) et al.

Licensed under MIT
