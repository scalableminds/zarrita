from pathlib import Path
from shutil import rmtree

import numpy as np
import pytest
import webknossos as wk
import zarr
from pytest import fixture

from zarrita import Array, Group, LocalStore, Store, codecs, runtime_configuration
from zarrita.indexing import morton_order_iter


@fixture
def store() -> Store:
    path = Path("testdata")
    rmtree(path, ignore_errors=True)
    return LocalStore(path)


def test_sharding(store):
    data = (
        wk.Dataset.open("l4_sample")
        .get_layer("color")
        .get_mag(1)
        .read()[0][:128, :128, :128]
    )

    a = Array.create(
        store,
        "l4_sample/color/1",
        shape=data.shape,
        chunk_shape=(64, 64, 64),
        dtype=data.dtype,
        fill_value=0,
        codecs=[
            codecs.sharding_codec(
                (32, 32, 32),
                [
                    codecs.transpose_codec("F"),
                    codecs.blosc_codec("lz4"),
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

    a = Array.create(
        store,
        "l4_sample/color/1",
        shape=tuple(a + 10 for a in data.shape),
        chunk_shape=(512, 512, 512),
        dtype=data.dtype,
        fill_value=0,
        codecs=[
            codecs.sharding_codec(
                (32, 32, 32),
                [
                    codecs.transpose_codec("F"),
                    codecs.blosc_codec("lz4"),
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


@pytest.mark.parametrize("input_order", ["F", "C"])
@pytest.mark.parametrize("store_order", ["F", "C"])
@pytest.mark.parametrize("runtime_order", ["F", "C"])
@pytest.mark.asyncio
async def test_order(store, input_order, store_order, runtime_order):
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16), order=input_order)

    a = await Array.create_async(
        store,
        "order",
        shape=data.shape,
        chunk_shape=(16, 16),
        dtype=data.dtype,
        fill_value=0,
        chunk_key_encoding=("v2", "."),
        codecs=[codecs.transpose_codec(store_order)],
    )

    await a.async_[:, :].set(data)
    read_data = await a.async_[:, :].get()
    assert np.array_equal(data, read_data)

    a = await Array.open_async(
        store,
        "order",
        runtime_configuration=runtime_configuration(order=runtime_order),
    )
    read_data = await a.async_[:, :].get()
    assert np.array_equal(data, read_data)

    if runtime_order == "F":
        assert read_data.flags["F_CONTIGUOUS"]
        assert not read_data.flags["C_CONTIGUOUS"]
    else:
        assert not read_data.flags["F_CONTIGUOUS"]
        assert read_data.flags["C_CONTIGUOUS"]

    # Compare with zarr-python
    z = zarr.create(
        shape=data.shape,
        chunks=(16, 16),
        dtype="<u2",
        order=store_order,
        compressor=None,
        fill_value=1,
        store="testdata/order_zarr",
    )
    z[:, :] = data
    assert await store.get_async("order/0.0") == await store.get_async("order_zarr/0.0")


def test_order_implicitC(store):
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16), order="F")

    a = Array.create(
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
    a = Array.create(
        store,
        "open",
        shape=(16, 16),
        chunk_shape=(16, 16),
        dtype="int32",
        fill_value=0,
    )
    b = Array.open(store, "open")
    assert a.metadata == b.metadata


def test_open_sharding(store):
    a = Array.create(
        store,
        "open_sharding",
        shape=(16, 16),
        chunk_shape=(16, 16),
        dtype="int32",
        fill_value=0,
        codecs=[
            codecs.sharding_codec(
                (8, 8),
                [codecs.transpose_codec("F"), codecs.blosc_codec()],
            )
        ],
    )
    b = Array.open(store, "open_sharding")
    assert a.metadata == b.metadata


def test_simple(store):
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))

    a = Array.create(
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

    g = Group.create(store, "group")
    g.create_array(
        "array",
        shape=data.shape,
        chunk_shape=(16, 16),
        dtype=data.dtype,
        fill_value=0,
    )
    g.create_group("group2")

    assert isinstance(g["array"], Array)
    assert isinstance(g["group2"], Group)


def test_write_partial_chunks(store):
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))

    a = Array.create(
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

    a = Array.create(
        store,
        "write_full_chunks",
        shape=(16, 16),
        chunk_shape=(20, 20),
        dtype=data.dtype,
        fill_value=1,
    )
    a[0:16, 0:16] = data
    assert np.array_equal(a[0:16, 0:16], data)

    a = Array.create(
        store,
        "write_full_chunks2",
        shape=(20, 20),
        chunk_shape=(20, 20),
        dtype=data.dtype,
        fill_value=1,
    )
    assert np.all(a[16:20, 16:20] == 1)


def test_write_partial_sharded_chunks(store):
    data = np.arange(0, 16 * 16, dtype="uint16").reshape((16, 16))

    a = Array.create(
        store,
        "write_partial_sharded_chunks",
        shape=(40, 40),
        chunk_shape=(20, 20),
        dtype=data.dtype,
        fill_value=1,
        codecs=[
            codecs.sharding_codec(chunk_shape=(10, 10), codecs=[codecs.blosc_codec()])
        ],
    )
    a[0:16, 0:16] = data
    assert np.array_equal(a[0:16, 0:16], data)


@pytest.mark.asyncio
async def test_delete_empty_chunks(store):
    data = np.ones((16, 16))

    a = await Array.create_async(
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
    assert await store.get_async("delete_empty_chunks/c0/0") is None


@pytest.mark.asyncio
async def test_delete_empty_sharded_chunks(store):
    a = await Array.create_async(
        store,
        "delete_empty_sharded_chunks",
        shape=(16, 16),
        chunk_shape=(8, 16),
        dtype="uint16",
        fill_value=1,
        codecs=[codecs.sharding_codec(chunk_shape=(8, 8))],
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
    assert await store.get_async("delete_empty_sharded_chunks/c/1/0") is None
    assert (
        len(await store.get_async("delete_empty_sharded_chunks/c/0/0"))
        == 16 * 2 + 8 * 8 * 2 + 4
    )


@pytest.mark.asyncio
async def test_zarr_compat(store):
    data = np.zeros((16, 18), dtype="uint16")

    a = await Array.create_async(
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

    a = await Array.create_async(
        store,
        "zarr_compatF3",
        shape=data.shape,
        chunk_shape=(10, 10),
        dtype=data.dtype,
        chunk_key_encoding=("v2", "."),
        fill_value=1,
        codecs=[codecs.transpose_codec("F")],
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

    Array.create(
        store,
        "dimension_names",
        shape=data.shape,
        chunk_shape=(16, 16),
        dtype=data.dtype,
        fill_value=0,
        dimension_names=("x", "y"),
    )

    assert Array.open(store, "dimension_names").metadata.dimension_names == (
        "x",
        "y",
    )

    Array.create(
        store,
        "dimension_names2",
        shape=data.shape,
        chunk_shape=(16, 16),
        dtype=data.dtype,
        fill_value=0,
    )

    assert Array.open(store, "dimension_names2").metadata.dimension_names is None


def test_gzip(store):
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))

    a = Array.create(
        store,
        "gzip",
        shape=data.shape,
        chunk_shape=(16, 16),
        dtype=data.dtype,
        fill_value=0,
        codecs=[codecs.gzip_codec()],
    )

    a[:, :] = data
    assert np.array_equal(data, a[:, :])


@pytest.mark.parametrize("endian", ["big", "little"])
@pytest.mark.asyncio
async def test_endian(store, endian):
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))

    a = await Array.create_async(
        store,
        "endian",
        shape=data.shape,
        chunk_shape=(16, 16),
        dtype=data.dtype,
        fill_value=0,
        chunk_key_encoding=("v2", "."),
        codecs=[codecs.endian_codec(endian)],
    )

    await a.async_[:, :].set(data)
    readback_data = await a.async_[:, :].get()
    assert np.array_equal(data, readback_data)

    # Compare with zarr-python
    z = zarr.create(
        shape=data.shape,
        chunks=(16, 16),
        dtype=">u2" if endian == "big" else "<u2",
        compressor=None,
        fill_value=1,
        store="testdata/endian_zarr",
    )
    z[:, :] = data
    assert await store.get_async("endian/0.0") == await store.get_async(
        "endian_zarr/0.0"
    )


@pytest.mark.parametrize("dtype_input_endian", [">u2", "<u2"])
@pytest.mark.parametrize("dtype_store_endian", ["big", "little"])
@pytest.mark.asyncio
async def test_endian_write(store, dtype_input_endian, dtype_store_endian):
    data = np.arange(0, 256, dtype=dtype_input_endian).reshape((16, 16))

    a = await Array.create_async(
        store,
        "endian",
        shape=data.shape,
        chunk_shape=(16, 16),
        dtype="uint16",
        fill_value=0,
        chunk_key_encoding=("v2", "."),
        codecs=[codecs.endian_codec(dtype_store_endian)],
    )

    await a.async_[:, :].set(data)
    readback_data = await a.async_[:, :].get()
    assert np.array_equal(data, readback_data)

    # Compare with zarr-python
    z = zarr.create(
        shape=data.shape,
        chunks=(16, 16),
        dtype=">u2" if dtype_store_endian == "big" else "<u2",
        compressor=None,
        fill_value=1,
        store="testdata/endian_zarr",
    )
    z[:, :] = data
    assert await store.get_async("endian/0.0") == await store.get_async(
        "endian_zarr/0.0"
    )


def test_invalid_metadata(store):
    with pytest.raises(AssertionError):
        Array.create(
            store,
            "invalid",
            shape=(16, 16, 16),
            chunk_shape=(16, 16),
            dtype=np.dtype("uint8"),
            fill_value=0,
        )

    with pytest.raises(AssertionError):
        Array.create(
            store,
            "invalid",
            shape=(16, 16),
            chunk_shape=(16, 16),
            dtype=np.dtype("uint8"),
            fill_value=0,
            codecs=[
                codecs.endian_codec("big"),
                codecs.transpose_codec("F"),
            ],
        )

    with pytest.raises(AssertionError):
        Array.create(
            store,
            "invalid",
            shape=(16, 16),
            chunk_shape=(16, 16),
            dtype=np.dtype("uint8"),
            fill_value=0,
            codecs=[
                codecs.sharding_codec(chunk_shape=(8,)),
            ],
        )
    with pytest.raises(AssertionError):
        Array.create(
            store,
            "invalid",
            shape=(16, 16),
            chunk_shape=(16, 16),
            dtype=np.dtype("uint8"),
            fill_value=0,
            codecs=[
                codecs.sharding_codec(chunk_shape=(8, 7)),
            ],
        )

    with pytest.warns(UserWarning):
        Array.create(
            store,
            "invalid",
            shape=(16, 16),
            chunk_shape=(16, 16),
            dtype=np.dtype("uint8"),
            fill_value=0,
            codecs=[
                codecs.sharding_codec(chunk_shape=(8, 8)),
                codecs.gzip_codec(),
            ],
        )


@pytest.mark.asyncio
async def test_reshape(store):
    data = np.zeros((16, 18), dtype="uint16")

    a = await Array.create_async(
        store,
        "reshape",
        shape=data.shape,
        chunk_shape=(10, 10),
        dtype=data.dtype,
        chunk_key_encoding=("v2", "."),
        fill_value=1,
    )

    await a.async_[:16, :18].set(data)
    assert await store.get_async("reshape/0.0") is not None
    assert await store.get_async("reshape/0.1") is not None
    assert await store.get_async("reshape/1.0") is not None
    assert await store.get_async("reshape/1.1") is not None

    a = await a.reshape_async((10, 12))
    assert a.metadata.shape == (10, 12)
    assert await store.get_async("reshape/0.0") is not None
    assert await store.get_async("reshape/0.1") is not None
    assert await store.get_async("reshape/1.0") is None
    assert await store.get_async("reshape/1.1") is None


def test_exists_ok(store):
    Array.create(
        store,
        "exists_ok",
        shape=(16, 16),
        chunk_shape=(16, 16),
        dtype=np.dtype("uint8"),
    )
    with pytest.raises(AssertionError):
        Array.create(
            store,
            "exists_ok",
            shape=(16, 16),
            chunk_shape=(16, 16),
            dtype=np.dtype("uint8"),
        )
    Array.create(
        store,
        "exists_ok",
        shape=(16, 16),
        chunk_shape=(16, 16),
        dtype=np.dtype("uint8"),
        exists_ok=True,
    )
