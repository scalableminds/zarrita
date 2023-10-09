import json
from pathlib import Path
from shutil import rmtree
from typing import Iterator, List, Literal, Optional

import numpy as np
import pytest
import wkw
import zarr
from pytest import fixture

from zarrita import Array, Group, LocalStore, Store, codecs, runtime_configuration
from zarrita.indexing import morton_order_iter
from zarrita.metadata import CodecMetadata


@fixture
def l4_sample_data() -> np.ndarray:
    return wkw.Dataset.open("l4_sample/color/1").read(
        (3072, 3072, 512), (128, 128, 128)
    )[0]


@fixture
def store() -> Iterator[Store]:
    path = Path("testdata")
    rmtree(path, ignore_errors=True)
    try:
        yield LocalStore(path)
    finally:
        # rmtree(path, ignore_errors=True)
        pass


def test_sharding(store: Store, l4_sample_data: np.ndarray):
    data = l4_sample_data

    a = Array.create(
        store / "l4_sample" / "color" / "1",
        shape=data.shape,
        chunk_shape=(64, 64, 64),
        dtype=data.dtype,
        fill_value=0,
        codecs=[
            codecs.sharding_codec(
                (32, 32, 32),
                [
                    codecs.transpose_codec("F"),
                    codecs.bytes_codec(),
                    codecs.blosc_codec(typesize=data.dtype.itemsize, cname="lz4"),
                ],
            )
        ],
    )

    a[:, :, :] = data

    read_data = a[0 : data.shape[0], 0 : data.shape[1], 0 : data.shape[2]]
    assert data.shape == read_data.shape
    assert np.array_equal(data, read_data)


def test_sharding_partial(store: Store, l4_sample_data: np.ndarray):
    data = l4_sample_data

    a = Array.create(
        store / "l4_sample" / "color" / "1",
        shape=tuple(a + 10 for a in data.shape),
        chunk_shape=(512, 512, 512),
        dtype=data.dtype,
        fill_value=0,
        codecs=[
            codecs.sharding_codec(
                (32, 32, 32),
                [
                    codecs.transpose_codec("F"),
                    codecs.bytes_codec(),
                    codecs.blosc_codec(typesize=data.dtype.itemsize, cname="lz4"),
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


def test_sharding_partial_read(store: Store, l4_sample_data: np.ndarray):
    data = l4_sample_data

    a = Array.create(
        store / "l4_sample" / "color" / "1",
        shape=tuple(a + 10 for a in data.shape),
        chunk_shape=(512, 512, 512),
        dtype=data.dtype,
        fill_value=1,
        codecs=[
            codecs.sharding_codec(
                (32, 32, 32),
                [
                    codecs.transpose_codec("F"),
                    codecs.bytes_codec(),
                    codecs.blosc_codec(typesize=data.dtype.itemsize, cname="lz4"),
                ],
            )
        ],
    )

    read_data = a[0:10, 0:10, 0:10]
    assert np.all(read_data == 1)


def test_sharding_partial_overwrite(store: Store, l4_sample_data: np.ndarray):
    data = l4_sample_data[:10, :10, :10]

    a = Array.create(
        store / "l4_sample" / "color" / "1",
        shape=tuple(a + 10 for a in data.shape),
        chunk_shape=(512, 512, 512),
        dtype=data.dtype,
        fill_value=1,
        codecs=[
            codecs.sharding_codec(
                (32, 32, 32),
                [
                    codecs.transpose_codec("F"),
                    codecs.bytes_codec(),
                    codecs.blosc_codec(typesize=data.dtype.itemsize, cname="lz4"),
                ],
            )
        ],
    )

    a[:10, :10, :10] = data

    read_data = a[0:10, 0:10, 0:10]
    assert np.array_equal(data, read_data)

    data = data + 10
    a[:10, :10, :10] = data
    read_data = a[0:10, 0:10, 0:10]
    assert np.array_equal(data, read_data)


def test_nested_sharding(store: Store, l4_sample_data: np.ndarray):
    data = l4_sample_data

    a = Array.create(
        store / "l4_sample" / "color" / "1",
        shape=data.shape,
        chunk_shape=(64, 64, 64),
        dtype=data.dtype,
        fill_value=0,
        codecs=[
            codecs.sharding_codec(
                (32, 32, 32),
                [codecs.sharding_codec((16, 16, 16))],
            )
        ],
    )

    a[:, :, :] = data

    read_data = a[0 : data.shape[0], 0 : data.shape[1], 0 : data.shape[2]]
    assert data.shape == read_data.shape
    assert np.array_equal(data, read_data)


@pytest.mark.parametrize("input_order", ["F", "C"])
@pytest.mark.parametrize("store_order", ["F", "C"])
@pytest.mark.parametrize("runtime_write_order", ["F", "C"])
@pytest.mark.parametrize("runtime_read_order", ["F", "C"])
@pytest.mark.parametrize("with_sharding", [True, False])
@pytest.mark.asyncio
async def test_order(
    store: Store,
    input_order: Literal["F", "C"],
    store_order: Literal["F", "C"],
    runtime_write_order: Literal["F", "C"],
    runtime_read_order: Literal["F", "C"],
    with_sharding: bool,
):
    data = np.arange(0, 256, dtype="uint16").reshape((32, 8), order=input_order)

    codecs_: List[CodecMetadata] = (
        [
            codecs.sharding_codec(
                (16, 8),
                codecs=[codecs.transpose_codec(store_order), codecs.bytes_codec()],
            )
        ]
        if with_sharding
        else [codecs.transpose_codec(store_order), codecs.bytes_codec()]
    )

    a = await Array.create_async(
        store / "order",
        shape=data.shape,
        chunk_shape=(32, 8),
        dtype=data.dtype,
        fill_value=0,
        chunk_key_encoding=("v2", "."),
        codecs=codecs_,
        runtime_configuration=runtime_configuration(runtime_write_order),
    )

    await a.async_[:, :].set(data)
    read_data = await a.async_[:, :].get()
    assert np.array_equal(data, read_data)

    a = await Array.open_async(
        store / "order",
        runtime_configuration=runtime_configuration(order=runtime_read_order),
    )
    read_data = await a.async_[:, :].get()
    assert np.array_equal(data, read_data)

    if runtime_read_order == "F":
        assert read_data.flags["F_CONTIGUOUS"]
        assert not read_data.flags["C_CONTIGUOUS"]
    else:
        assert not read_data.flags["F_CONTIGUOUS"]
        assert read_data.flags["C_CONTIGUOUS"]

    if not with_sharding:
        # Compare with zarr-python
        z = zarr.create(
            shape=data.shape,
            chunks=(32, 8),
            dtype="<u2",
            order=store_order,
            compressor=None,
            fill_value=1,
            store="testdata/order_zarr",
        )
        z[:, :] = data
        assert await store.get_async("order/0.0") == await store.get_async(
            "order_zarr/0.0"
        )


@pytest.mark.parametrize("input_order", ["F", "C"])
@pytest.mark.parametrize("runtime_write_order", ["F", "C"])
@pytest.mark.parametrize("runtime_read_order", ["F", "C"])
@pytest.mark.parametrize("with_sharding", [True, False])
def test_order_implicit(
    store: Store,
    input_order: Literal["F", "C"],
    runtime_write_order: Literal["F", "C"],
    runtime_read_order: Literal["F", "C"],
    with_sharding: bool,
):
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16), order=input_order)

    codecs_: Optional[List[CodecMetadata]] = (
        [codecs.sharding_codec((8, 8))] if with_sharding else None
    )

    a = Array.create(
        store / "order_implicit",
        shape=data.shape,
        chunk_shape=(16, 16),
        dtype=data.dtype,
        fill_value=0,
        codecs=codecs_,
        runtime_configuration=runtime_configuration(runtime_write_order),
    )

    a[:, :] = data

    a = Array.open(
        store / "order_implicit",
        runtime_configuration=runtime_configuration(order=runtime_read_order),
    )
    read_data = a[:, :]
    assert np.array_equal(data, read_data)

    if runtime_read_order == "F":
        assert read_data.flags["F_CONTIGUOUS"]
        assert not read_data.flags["C_CONTIGUOUS"]
    else:
        assert not read_data.flags["F_CONTIGUOUS"]
        assert read_data.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize("input_order", ["F", "C"])
@pytest.mark.parametrize("runtime_write_order", ["F", "C"])
@pytest.mark.parametrize("runtime_read_order", ["F", "C"])
@pytest.mark.parametrize("with_sharding", [True, False])
@pytest.mark.asyncio
async def test_transpose(
    store: Store,
    input_order: Literal["F", "C"],
    runtime_write_order: Literal["F", "C"],
    runtime_read_order: Literal["F", "C"],
    with_sharding: bool,
):
    data = np.arange(0, 256, dtype="uint16").reshape((1, 32, 8), order=input_order)

    codecs_: List[CodecMetadata] = (
        [
            codecs.sharding_codec(
                (1, 16, 8),
                codecs=[codecs.transpose_codec((2, 1, 0)), codecs.bytes_codec()],
            )
        ]
        if with_sharding
        else [codecs.transpose_codec((2, 1, 0)), codecs.bytes_codec()]
    )

    a = await Array.create_async(
        store / "transpose",
        shape=data.shape,
        chunk_shape=(1, 32, 8),
        dtype=data.dtype,
        fill_value=0,
        chunk_key_encoding=("v2", "."),
        codecs=codecs_,
        runtime_configuration=runtime_configuration(runtime_write_order),
    )

    await a.async_[:, :].set(data)
    read_data = await a.async_[:, :].get()
    assert np.array_equal(data, read_data)

    a = await Array.open_async(
        store / "transpose",
        runtime_configuration=runtime_configuration(runtime_read_order),
    )
    read_data = await a.async_[:, :].get()
    assert np.array_equal(data, read_data)

    if runtime_read_order == "F":
        assert read_data.flags["F_CONTIGUOUS"]
        assert not read_data.flags["C_CONTIGUOUS"]
    else:
        assert not read_data.flags["F_CONTIGUOUS"]
        assert read_data.flags["C_CONTIGUOUS"]

    if not with_sharding:
        # Compare with zarr-python
        z = zarr.create(
            shape=data.shape,
            chunks=(1, 32, 8),
            dtype="<u2",
            order="F",
            compressor=None,
            fill_value=1,
            store="testdata/transpose_zarr",
        )
        z[:, :] = data
        assert await store.get_async("transpose/0.0") == await store.get_async(
            "transpose_zarr/0.0"
        )


def test_transpose_invalid(
    store: Store,
):
    data = np.arange(0, 256, dtype="uint16").reshape((1, 32, 8))

    for order in [(1, 0), (3, 2, 1), (3, 3, 1)]:
        with pytest.raises(AssertionError):
            Array.create(
                store / "transpose_invalid",
                shape=data.shape,
                chunk_shape=(1, 32, 8),
                dtype=data.dtype,
                fill_value=0,
                chunk_key_encoding=("v2", "."),
                codecs=[codecs.transpose_codec(order), codecs.bytes_codec()],
            )


def test_open(store: Store):
    a = Array.create(
        store / "open",
        shape=(16, 16),
        chunk_shape=(16, 16),
        dtype="int32",
        fill_value=0,
    )
    b = Array.open(store / "open")
    assert a.metadata == b.metadata


def test_open_sharding(store: Store):
    a = Array.create(
        store / "open_sharding",
        shape=(16, 16),
        chunk_shape=(16, 16),
        dtype="int32",
        fill_value=0,
        codecs=[
            codecs.sharding_codec(
                (8, 8),
                [
                    codecs.transpose_codec("F"),
                    codecs.bytes_codec(),
                    codecs.blosc_codec(typesize=4),
                ],
            )
        ],
    )
    b = Array.open(store / "open_sharding")
    assert a.metadata == b.metadata


def test_simple(store: Store):
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))

    a = Array.create(
        store / "simple",
        shape=data.shape,
        chunk_shape=(16, 16),
        dtype=data.dtype,
        fill_value=0,
    )

    a[:, :] = data
    assert np.array_equal(data, a[:, :])


def test_fill_value(store: Store):
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))

    a = Array.create(
        store / "fill_value1",
        shape=data.shape,
        chunk_shape=(16, 16),
        dtype=data.dtype,
    )

    assert a.metadata.fill_value == 0
    assert np.array_equiv(0, a[:, :])

    b = Array.create(
        store / "fill_value2",
        shape=data.shape,
        chunk_shape=(16, 16),
        dtype=np.dtype("bool"),
    )

    assert b.metadata.fill_value is False
    assert np.array_equiv(False, b[:, :])

    c = Array.create(
        store / "fill_value3",
        shape=data.shape,
        chunk_shape=(16, 16),
        dtype=data.dtype,
        fill_value=4,
    )

    assert c.metadata.fill_value == 4
    assert np.array_equiv(4, c[:, :])


def test_morton(store: Store):
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


def test_group(store: Store):
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))

    g = Group.create(store / "group")
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


def test_write_partial_chunks(store: Store):
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))

    a = Array.create(
        store / "write_partial_chunks",
        shape=data.shape,
        chunk_shape=(20, 20),
        dtype=data.dtype,
        fill_value=1,
    )
    a[0:16, 0:16] = data
    assert np.array_equal(a[0:16, 0:16], data)


def test_write_full_chunks(store: Store):
    data = np.arange(0, 16 * 16, dtype="uint16").reshape((16, 16))

    a = Array.create(
        store / "write_full_chunks",
        shape=(16, 16),
        chunk_shape=(20, 20),
        dtype=data.dtype,
        fill_value=1,
    )
    a[0:16, 0:16] = data
    assert np.array_equal(a[0:16, 0:16], data)

    a = Array.create(
        store / "write_full_chunks2",
        shape=(20, 20),
        chunk_shape=(20, 20),
        dtype=data.dtype,
        fill_value=1,
    )
    assert np.all(a[16:20, 16:20] == 1)


def test_write_partial_sharded_chunks(store: Store):
    data = np.arange(0, 16 * 16, dtype="uint16").reshape((16, 16))

    a = Array.create(
        store / "write_partial_sharded_chunks",
        shape=(40, 40),
        chunk_shape=(20, 20),
        dtype=data.dtype,
        fill_value=1,
        codecs=[
            codecs.sharding_codec(
                chunk_shape=(10, 10),
                codecs=[
                    codecs.bytes_codec(),
                    codecs.blosc_codec(typesize=data.dtype.itemsize),
                ],
            )
        ],
    )
    a[0:16, 0:16] = data
    assert np.array_equal(a[0:16, 0:16], data)


@pytest.mark.asyncio
async def test_delete_empty_chunks(store: Store):
    data = np.ones((16, 16))

    a = await Array.create_async(
        store / "delete_empty_chunks",
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
async def test_delete_empty_sharded_chunks(store: Store):
    a = await Array.create_async(
        store / "delete_empty_sharded_chunks",
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
    chunk_bytes = await store.get_async("delete_empty_sharded_chunks/c/0/0")
    assert chunk_bytes is not None and len(chunk_bytes) == 16 * 2 + 8 * 8 * 2 + 4


@pytest.mark.asyncio
async def test_zarr_compat(store: Store):
    data = np.zeros((16, 18), dtype="uint16")

    a = await Array.create_async(
        store / "zarr_compat3",
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
async def test_zarr_compat_F(store: Store):
    data = np.zeros((16, 18), dtype="uint16", order="F")

    a = await Array.create_async(
        store / "zarr_compatF3",
        shape=data.shape,
        chunk_shape=(10, 10),
        dtype=data.dtype,
        chunk_key_encoding=("v2", "."),
        fill_value=1,
        codecs=[codecs.transpose_codec("F"), codecs.bytes_codec()],
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


@pytest.mark.asyncio
async def test_dimension_names(store: Store):
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))

    await Array.create_async(
        store / "dimension_names",
        shape=data.shape,
        chunk_shape=(16, 16),
        dtype=data.dtype,
        fill_value=0,
        dimension_names=("x", "y"),
    )

    assert (
        await Array.open_async(store / "dimension_names")
    ).metadata.dimension_names == (
        "x",
        "y",
    )

    await Array.create_async(
        store / "dimension_names2",
        shape=data.shape,
        chunk_shape=(16, 16),
        dtype=data.dtype,
        fill_value=0,
    )

    assert (
        await Array.open_async(store / "dimension_names2")
    ).metadata.dimension_names is None
    zarr_json_bytes = await (store / "dimension_names2" / "zarr.json").get_async()
    assert zarr_json_bytes is not None
    assert "dimension_names" not in json.loads(zarr_json_bytes)


def test_gzip(store: Store):
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))

    a = Array.create(
        store / "gzip",
        shape=data.shape,
        chunk_shape=(16, 16),
        dtype=data.dtype,
        fill_value=0,
        codecs=[codecs.bytes_codec(), codecs.gzip_codec()],
    )

    a[:, :] = data
    assert np.array_equal(data, a[:, :])


@pytest.mark.parametrize("checksum", [True, False])
def test_zstd(store: Store, checksum: bool):
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))

    a = Array.create(
        store / "zstd",
        shape=data.shape,
        chunk_shape=(16, 16),
        dtype=data.dtype,
        fill_value=0,
        codecs=[codecs.bytes_codec(), codecs.zstd_codec(level=0, checksum=checksum)],
    )

    a[:, :] = data
    assert np.array_equal(data, a[:, :])


@pytest.mark.parametrize("endian", ["big", "little"])
@pytest.mark.asyncio
async def test_endian(store: Store, endian: Literal["big", "little"]):
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))

    a = await Array.create_async(
        store / "endian",
        shape=data.shape,
        chunk_shape=(16, 16),
        dtype=data.dtype,
        fill_value=0,
        chunk_key_encoding=("v2", "."),
        codecs=[codecs.bytes_codec(endian)],
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
async def test_endian_write(
    store: Store,
    dtype_input_endian: Literal[">u2", "<u2"],
    dtype_store_endian: Literal["big", "little"],
):
    data = np.arange(0, 256, dtype=dtype_input_endian).reshape((16, 16))

    a = await Array.create_async(
        store / "endian",
        shape=data.shape,
        chunk_shape=(16, 16),
        dtype="uint16",
        fill_value=0,
        chunk_key_encoding=("v2", "."),
        codecs=[codecs.bytes_codec(dtype_store_endian)],
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


def test_invalid_metadata(store: Store):
    with pytest.raises(AssertionError):
        Array.create(
            store / "invalid",
            shape=(16, 16, 16),
            chunk_shape=(16, 16),
            dtype=np.dtype("uint8"),
            fill_value=0,
        )

    with pytest.raises(AssertionError):
        Array.create(
            store / "invalid",
            shape=(16, 16),
            chunk_shape=(16, 16),
            dtype=np.dtype("uint8"),
            fill_value=0,
            codecs=[
                codecs.bytes_codec("big"),
                codecs.transpose_codec("F"),
            ],
        )

    with pytest.raises(AssertionError):
        Array.create(
            store / "invalid",
            shape=(16, 16),
            chunk_shape=(16, 16),
            dtype=np.dtype("uint8"),
            fill_value=0,
            codecs=[
                codecs.transpose_codec("F"),
            ],
        )

    with pytest.raises(AssertionError):
        Array.create(
            store / "invalid",
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
            store / "invalid",
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
            store / "invalid",
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
async def test_resize(store: Store):
    data = np.zeros((16, 18), dtype="uint16")

    a = await Array.create_async(
        store / "resize",
        shape=data.shape,
        chunk_shape=(10, 10),
        dtype=data.dtype,
        chunk_key_encoding=("v2", "."),
        fill_value=1,
    )

    await a.async_[:16, :18].set(data)
    assert await store.get_async("resize/0.0") is not None
    assert await store.get_async("resize/0.1") is not None
    assert await store.get_async("resize/1.0") is not None
    assert await store.get_async("resize/1.1") is not None

    a = await a.resize_async((10, 12))
    assert a.metadata.shape == (10, 12)
    assert await store.get_async("resize/0.0") is not None
    assert await store.get_async("resize/0.1") is not None
    assert await store.get_async("resize/1.0") is None
    assert await store.get_async("resize/1.1") is None


def test_exists_ok(store: Store):
    Array.create(
        store / "exists_ok",
        shape=(16, 16),
        chunk_shape=(16, 16),
        dtype=np.dtype("uint8"),
    )
    with pytest.raises(AssertionError):
        Array.create(
            store / "exists_ok",
            shape=(16, 16),
            chunk_shape=(16, 16),
            dtype=np.dtype("uint8"),
        )
    Array.create(
        store / "exists_ok",
        shape=(16, 16),
        chunk_shape=(16, 16),
        dtype=np.dtype("uint8"),
        exists_ok=True,
    )


def test_update_attributes_array(store: Store):
    data = np.zeros((16, 18), dtype="uint16")

    a = Array.create(
        store / "update_attributes",
        shape=data.shape,
        chunk_shape=(10, 10),
        dtype=data.dtype,
        fill_value=1,
        attributes={"hello": "world"},
    )

    a = Array.open(store / "update_attributes")
    assert a.metadata.attributes["hello"] == "world"

    a.update_attributes({"hello": "zarrita"})

    a = Array.open(store / "update_attributes")
    assert a.metadata.attributes["hello"] == "zarrita"


def test_update_attributes_group(store: Store):
    g = Group.create(store / "update_attributes_group", attributes={"hello": "world"})

    g = Group.open(store / "update_attributes_group")
    assert g.metadata.attributes["hello"] == "world"

    g.update_attributes({"hello": "zarrita"})

    g = Group.open(store / "update_attributes_group")
    assert g.metadata.attributes["hello"] == "zarrita"
