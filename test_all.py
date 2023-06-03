from shutil import rmtree

import numpy as np
import pytest
import webknossos as wk
import zarr
from pytest import fixture

from zarrita import *
from zarrita.indexing import morton_order_iter


@fixture
def store() -> zarrita.Store:
    rmtree("testdata", ignore_errors=True)
    return zarrita.FileSystemStore("file://./testdata")


def test_sharding(store):
    data = (
        wk.Dataset.open("l4_sample")
        .get_layer("color")
        .get_mag(1)
        .read()[0][:128, :128, :128]
    )

    a = zarrita.Array.create(
        store,
        "l4_sample/color/1",
        shape=data.shape,
        chunk_shape=(64, 64, 64),
        dtype=data.dtype,
        fill_value=0,
        codecs=[
            zarrita.codecs.sharding_codec(
                (32, 32, 32),
                [
                    zarrita.codecs.transpose_codec("F"),
                    zarrita.codecs.blosc_codec("lz4"),
                ],
            )
        ],
    )

    a[:, :, :] = data

    read_data = a[0 : data.shape[0], 0 : data.shape[1], 0 : data.shape[2]]
    assert data.shape == read_data.shape
    assert np.array_equal(data, read_data)


def test_sharding_partial(store):
    data = (
        wk.Dataset.open("l4_sample")
        .get_layer("color")
        .get_mag(1)
        .read()[0][:128, :128, :128]
    )

    a = zarrita.Array.create(
        store,
        "l4_sample/color/1",
        shape=tuple(a + 10 for a in data.shape),
        chunk_shape=(512, 512, 512),
        dtype=data.dtype,
        fill_value=0,
        codecs=[
            zarrita.codecs.sharding_codec(
                (32, 32, 32),
                [
                    zarrita.codecs.transpose_codec("F"),
                    zarrita.codecs.blosc_codec("lz4"),
                ],
            )
        ],
    )

    a[10:, 10:, 10:] = data

    read_data = a[0:10, 0:10, 0:10]
    assert np.all(read_data == 0)

    read_data = a[10:, 10:, 10:]
    assert data.shape == read_data.shape
    assert np.array_equal(data, read_data)


def test_order_F(store):
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16), order="F")

    a = zarrita.Array.create(
        store,
        "order_F",
        shape=data.shape,
        chunk_shape=(16, 16),
        dtype=data.dtype,
        fill_value=0,
        codecs=[zarrita.codecs.transpose_codec("F")],
        runtime_configuration=zarrita.runtime_configuration(order="F"),
    )

    a[:, :] = data
    read_data = a[:, :]
    assert np.array_equal(data, read_data)
    assert read_data.flags["F_CONTIGUOUS"]
    assert not read_data.flags["C_CONTIGUOUS"]


def test_order_C(store):
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16), order="C")

    a = zarrita.Array.create(
        store,
        "order_C",
        shape=data.shape,
        chunk_shape=(16, 16),
        dtype=data.dtype,
        fill_value=0,
        codecs=[zarrita.codecs.transpose_codec("C")],
    )

    a[:, :] = data
    read_data = a[:, :]
    assert np.array_equal(data, read_data)
    assert read_data.flags["C_CONTIGUOUS"]
    assert not read_data.flags["F_CONTIGUOUS"]


def test_order_implicitC(store):
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16), order="F")

    a = zarrita.Array.create(
        store,
        "order_implicitC",
        shape=data.shape,
        chunk_shape=(16, 16),
        dtype=data.dtype,
        fill_value=0,
    )

    a[:, :] = data
    read_data = a[:, :]
    assert np.array_equal(data, read_data)
    assert read_data.flags["C_CONTIGUOUS"]
    assert not read_data.flags["F_CONTIGUOUS"]


def test_open(store):
    a = zarrita.Array.create(
        store,
        "open",
        shape=(16, 16),
        chunk_shape=(16, 16),
        dtype="int32",
        fill_value=0,
    )
    b = zarrita.Array.open(store, "open")
    assert a.metadata == b.metadata


def test_open_sharding(store):
    a = zarrita.Array.create(
        store,
        "open_sharding",
        shape=(16, 16),
        chunk_shape=(16, 16),
        dtype="int32",
        fill_value=0,
        codecs=[
            zarrita.codecs.sharding_codec(
                (8, 8),
                [zarrita.codecs.transpose_codec("F"), zarrita.codecs.blosc_codec()],
            )
        ],
    )
    b = zarrita.Array.open(store, "open_sharding")
    assert a.metadata == b.metadata


def test_simple(store):
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))

    a = zarrita.Array.create(
        store,
        "simple",
        shape=data.shape,
        chunk_shape=(16, 16),
        dtype=data.dtype,
        fill_value=0,
    )

    a[:, :] = data
    assert np.array_equal(data, a[:, :])


def test_morton(store):
    assert list(morton_order_iter((2, 2))) == [(0, 0), (1, 0), (0, 1), (1, 1)]
    assert list(morton_order_iter((2, 2, 2))) == [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    ]
    assert list(morton_order_iter((2, 2, 2, 2))) == [
        (0, 0, 0, 0),
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (1, 1, 0, 0),
        (0, 0, 1, 0),
        (1, 0, 1, 0),
        (0, 1, 1, 0),
        (1, 1, 1, 0),
        (0, 0, 0, 1),
        (1, 0, 0, 1),
        (0, 1, 0, 1),
        (1, 1, 0, 1),
        (0, 0, 1, 1),
        (1, 0, 1, 1),
        (0, 1, 1, 1),
        (1, 1, 1, 1),
    ]


def test_group(store):
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))

    g = zarrita.Group.create(store, "group")
    g.create_array(
        "array",
        shape=data.shape,
        chunk_shape=(16, 16),
        dtype=data.dtype,
        fill_value=0,
    )
    g.create_group("group2")

    assert isinstance(g["array"], zarrita.Array)
    assert isinstance(g["group2"], zarrita.Group)


def test_write_partial_chunks(store):
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))

    a = zarrita.Array.create(
        store,
        "write_partial_chunks",
        shape=data.shape,
        chunk_shape=(20, 20),
        dtype=data.dtype,
        fill_value=1,
    )
    a[0:16, 0:16] = data
    assert np.array_equal(a[0:16, 0:16], data)


def test_write_full_chunks(store):
    data = np.arange(0, 16 * 16, dtype="uint16").reshape((16, 16))

    a = zarrita.Array.create(
        store,
        "write_full_chunks",
        shape=(16, 16),
        chunk_shape=(20, 20),
        dtype=data.dtype,
        fill_value=1,
    )
    a[0:16, 0:16] = data
    assert np.array_equal(a[0:16, 0:16], data)

    a = zarrita.Array.create(
        store,
        "write_full_chunks",
        shape=(20, 20),
        chunk_shape=(20, 20),
        dtype=data.dtype,
        fill_value=1,
    )
    assert np.all(a[16:20, 16:20] == 1)


def test_write_partial_sharded_chunks(store):
    data = np.arange(0, 16 * 16, dtype="uint16").reshape((16, 16))

    a = zarrita.Array.create(
        store,
        "write_partial_sharded_chunks",
        shape=(40, 40),
        chunk_shape=(20, 20),
        dtype=data.dtype,
        fill_value=1,
        codecs=[
            zarrita.codecs.sharding_codec(
                chunk_shape=(10, 10), codecs=[zarrita.codecs.blosc_codec()]
            )
        ],
    )
    a[0:16, 0:16] = data
    assert np.array_equal(a[0:16, 0:16], data)


@pytest.mark.asyncio
async def test_delete_empty_chunks(store):
    data = np.ones((16, 16))

    a = await zarrita.Array.create_async(
        store,
        "delete_empty_chunks",
        shape=data.shape,
        chunk_shape=(32, 32),
        dtype=data.dtype,
        fill_value=1,
    )
    await a.async_[:16, :16].set(np.zeros((16, 16)))
    await a.async_[:16, :16].set(data)
    assert np.array_equal(await a.async_[:16, :16].get(), data)
    assert await store.get_async("delete_empty_chunks/c0/0") == None


@pytest.mark.asyncio
async def test_delete_empty_sharded_chunks(store):
    a = await zarrita.Array.create_async(
        store,
        "delete_empty_sharded_chunks",
        shape=(16, 16),
        chunk_shape=(8, 16),
        dtype="uint16",
        fill_value=1,
        codecs=[zarrita.codecs.sharding_codec(chunk_shape=(8, 8))],
    )
    await a.async_[:, :].set(np.zeros((16, 16)))
    await a.async_[8:, :].set(np.ones((8, 16)))
    await a.async_[:, 8:].set(np.ones((16, 8)))
    # chunk (0, 0) is full
    # chunks (0, 1), (1, 0), (1, 1) are empty
    # shard (0, 0) is half-full
    # shard (1, 0) is empty

    data = np.ones((16, 16), dtype="uint16")
    data[:8, :8] = 0
    assert np.array_equal(data, await a.async_[:, :].get())
    assert await store.get_async("delete_empty_sharded_chunks/c/1/0") == None
    assert (
        len(await store.get_async("delete_empty_sharded_chunks/c/0/0"))
        == 16 * 2 + 8 * 8 * 2 + 4
    )


@pytest.mark.asyncio
async def test_zarr_compat(store):
    data = np.zeros((16, 18), dtype="uint16")

    a = await zarrita.Array.create_async(
        store,
        "zarr_compat3",
        shape=data.shape,
        chunk_shape=(10, 10),
        dtype=data.dtype,
        chunk_key_encoding=("v2", "."),
        fill_value=1,
    )

    z2 = zarr.create(
        shape=data.shape,
        chunks=(10, 10),
        dtype=data.dtype,
        compressor=None,
        fill_value=1,
        store="testdata/zarr_compat2",
    )

    await a.async_[:16, :18].set(data)
    z2[:16, :18] = data
    assert np.array_equal(data, await a.async_[:16, :18].get())
    assert np.array_equal(data, z2[:16, :18])

    assert await store.get_async("zarr_compat2/0.0") == await store.get_async(
        "zarr_compat3/0.0"
    )
    assert await store.get_async("zarr_compat2/0.1") == await store.get_async(
        "zarr_compat3/0.1"
    )
    assert await store.get_async("zarr_compat2/1.0") == await store.get_async(
        "zarr_compat3/1.0"
    )
    assert await store.get_async("zarr_compat2/1.1") == await store.get_async(
        "zarr_compat3/1.1"
    )


@pytest.mark.asyncio
async def test_zarr_compat_F(store):
    data = np.zeros((16, 18), dtype="uint16", order="F")

    a = await zarrita.Array.create_async(
        store,
        "zarr_compatF3",
        shape=data.shape,
        chunk_shape=(10, 10),
        dtype=data.dtype,
        chunk_key_encoding=("v2", "."),
        fill_value=1,
        codecs=[zarrita.codecs.transpose_codec("F")],
    )

    z2 = zarr.create(
        shape=data.shape,
        chunks=(10, 10),
        dtype=data.dtype,
        compressor=None,
        order="F",
        fill_value=1,
        store="testdata/zarr_compatF2",
    )

    await a.async_[:16, :18].set(data)
    z2[:16, :18] = data
    assert np.array_equal(data, await a.async_[:16, :18].get())
    assert np.array_equal(data, z2[:16, :18])

    assert await store.get_async("zarr_compatF2/0.0") == await store.get_async(
        "zarr_compatF3/0.0"
    )
    assert await store.get_async("zarr_compatF2/0.1") == await store.get_async(
        "zarr_compatF3/0.1"
    )
    assert await store.get_async("zarr_compatF2/1.0") == await store.get_async(
        "zarr_compatF3/1.0"
    )
    assert await store.get_async("zarr_compatF2/1.1") == await store.get_async(
        "zarr_compatF3/1.1"
    )


def test_dimension_names(store):
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))

    a = zarrita.Array.create(
        store,
        "dimension_names",
        shape=data.shape,
        chunk_shape=(16, 16),
        dtype=data.dtype,
        fill_value=0,
        dimension_names=("x", "y"),
    )

    assert zarrita.Array.open(store, "dimension_names").metadata.dimension_names == (
        "x",
        "y",
    )

    a = zarrita.Array.create(
        store,
        "dimension_names",
        shape=data.shape,
        chunk_shape=(16, 16),
        dtype=data.dtype,
        fill_value=0,
    )

    assert zarrita.Array.open(store, "dimension_names").metadata.dimension_names == None


def test_gzip(store):
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))

    a = zarrita.Array.create(
        store,
        "gzip",
        shape=data.shape,
        chunk_shape=(16, 16),
        dtype=data.dtype,
        fill_value=0,
        codecs=[zarrita.codecs.gzip_codec()],
    )

    a[:, :] = data
    assert np.array_equal(data, a[:, :])


def test_endian(store):
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))

    a = zarrita.Array.create(
        store,
        "endian",
        shape=data.shape,
        chunk_shape=(16, 16),
        dtype=data.dtype,
        fill_value=0,
        codecs=[zarrita.codecs.endian_codec("big")],
    )

    a[:, :] = data
    readback_data = a[:, :]
    assert np.array_equal(data, readback_data)


def test_invalid_metadata(store):
    with pytest.raises(AssertionError):
        a = zarrita.Array.create(
            store,
            "invalid",
            shape=(16, 16, 16),
            chunk_shape=(16, 16),
            dtype=np.dtype("uint8"),
            fill_value=0,
        )

    with pytest.raises(AssertionError):
        a = zarrita.Array.create(
            store,
            "invalid",
            shape=(16, 16),
            chunk_shape=(16, 16),
            dtype=np.dtype("uint8"),
            fill_value=0,
            codecs=[
                zarrita.codecs.endian_codec("big"),
                zarrita.codecs.transpose_codec("F"),
            ],
        )

    with pytest.raises(AssertionError):
        a = zarrita.Array.create(
            store,
            "invalid",
            shape=(16, 16),
            chunk_shape=(16, 16),
            dtype=np.dtype("uint8"),
            fill_value=0,
            codecs=[
                zarrita.codecs.sharding_codec(chunk_shape=(8,)),
            ],
        )
    with pytest.raises(AssertionError):
        a = zarrita.Array.create(
            store,
            "invalid",
            shape=(16, 16),
            chunk_shape=(16, 16),
            dtype=np.dtype("uint8"),
            fill_value=0,
            codecs=[
                zarrita.codecs.sharding_codec(chunk_shape=(8, 7)),
            ],
        )

    with pytest.warns(UserWarning):
        a = zarrita.Array.create(
            store,
            "invalid",
            shape=(16, 16),
            chunk_shape=(16, 16),
            dtype=np.dtype("uint8"),
            fill_value=0,
            codecs=[
                zarrita.codecs.sharding_codec(chunk_shape=(8, 8)),
                zarrita.codecs.gzip_codec(),
            ],
        )
