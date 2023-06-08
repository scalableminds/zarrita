# Zarrita

Zarrita is an experimental implementation of [Zarr v3](https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html) including [sharding](https://zarr.dev/zeps/draft/ZEP0002.html). This is only a technical proof of concept meant for generating sample datasets. Not recommended for production use.

## Setup

```python
import zarrita
import numpy as np

store = zarrita.LocalStore('testdata') # or zarrita.RemoteStore('s3://bucket/test')
```

## Create an array

```python
a = await zarrita.Array.create_async(
    store,
    'array',
    shape=(6, 10),
    dtype='int32',
    chunk_shape=(2, 5),
    codecs=[zarrita.codecs.blosc_codec()],
    attributes={'question': 'life', 'answer': 42}
)
await a.async_[:, :].set(np.ones((6, 10), dtype='int32'))
```

## Open an array

```python
a = await zarrita.Array.open_async(store, 'array')
assert np.array_equal(await a.async_[:, :].get(), np.ones((6, 10), dtype='int32'))
```

## Create an array with sharding

```python
a = await zarrita.Array.create_async(
    store,
    'sharding',
    shape=(16, 16),
    dtype='int32',
    chunk_shape=(16, 16),
    chunk_key_encoding=('v2', '.'),
    codecs=[
        zarrita.codecs.sharding_codec(
            chunk_shape=(8, 8),
            codecs=[zarrita.codecs.blosc_codec()]
        ),
    ],
)
data = np.arange(0, 16 * 16, dtype='int32').reshape((16, 16))
await a.async_[:, :].set(data)
assert np.array_equal(await a.async_[:, :].get(), data)
```

## Create a group

```python
g = await zarrita.Group.create_async(store, 'group')
g2 = await g.create_group_async('group2')
a = await g2.create_array_async(
    'array',
    shape=(16, 16),
    dtype='int32',
    chunk_shape=(16, 16),
)
await a.async_[:, :].set(np.arange(0, 16 * 16, dtype='int32').reshape((16, 16)))
```

## Open a group

```python
g = await zarrita.Group.open_async(store, 'group')
g2 = g['group2']
a = g['group2/array']
assert np.array_equal(await a.asnyc_[:, :].get(), np.arange(0, 16 * 16, dtype='int32').reshape((16, 16)))
```

# Credits

This is a largely-rewritten fork of `zarrita` by [@alimanfoo](https://github.com/alimanfoo). It implements the Zarr v3 draft specification created by [@alimanfoo](https://github.com/alimanfoo), [@jstriebel](https://github.com/jstriebel), [@jbms](https://github.com/jbms) et al.

Licensed under MIT
