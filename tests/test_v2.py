import json
from shutil import rmtree

import numcodecs
import numpy as np
import pytest
import zarr
from pytest import fixture

from zarrita import *


@fixture
def store() -> zarrita.Store:
    rmtree("testdata", ignore_errors=True)
    return zarrita.FileSystemStore("file://./testdata")


async def do_test(store, data, zarrita_kwargs, zarr_kwargs):
    # setup
    zarrita_array = await zarrita.ArrayV2.create_async(
        store,
        "zarrita_v2",
        shape=data.shape,
        dtype=data.dtype,
        chunks=(10, 10),
        **{
            **dict(dimension_separator=".", compressor=None, filters=None),
            **zarrita_kwargs,
        },
    )

    zarr_attrs = zarr_kwargs.pop("attributes", None)
    zarr_array = zarr.create(
        store="testdata/zarr_v2",
        shape=data.shape,
        dtype=data.dtype,
        chunks=(10, 10),
        **{
            **dict(
                dimension_separator=".",
                compressor=None,
            ),
            **zarr_kwargs,
        },
    )
    if zarr_attrs:
        zarr_array.attrs.update(**zarr_attrs)

    # write
    await zarrita_array.async_[:, :].set(data)
    zarr_array[:, :] = data

    # read
    assert np.array_equal(data, await zarrita_array.async_[:16, :18].get())
    assert np.array_equal(data, zarr_array[:16, :18])

    # open again
    zarrita_array = await zarrita.ArrayV2.open_async(store, "zarrita_v2")
    assert np.array_equal(data, await zarrita_array.async_[:16, :18].get())

    # open zarrita array with zarr
    zarrita_array_via_zarr = zarr.open_array("testdata/zarrita_v2")
    assert np.array_equal(data, zarrita_array_via_zarr[:16, :18])

    # open zarr array via zarrita
    zarr_array_via_zarrita = await zarrita.ArrayV2.open_async(store, "zarr_v2")
    assert np.array_equal(data, zarr_array_via_zarrita[:16, :18])

    await check_zarr2_files(store, "zarrita_v2", "zarr_v2")

    return zarrita_array, zarr_array


async def check_zarr2_files(store, folder_a, folder_b):
    assert json.loads(await store.get_async(f"{folder_a}/.zarray")) == json.loads(
        await store.get_async(f"{folder_b}/.zarray")
    )
    zarr_zattrs_bytes = await store.get_async(f"{folder_b}/.zattrs")
    if zarr_zattrs_bytes is not None:
        assert json.loads(await store.get_async(f"{folder_a}/.zattrs")) == json.loads(
            zarr_zattrs_bytes
        )
    assert await store.get_async(f"{folder_a}/0.0") == await store.get_async(
        f"{folder_b}/0.0"
    )
    assert await store.get_async(f"{folder_a}/0.1") == await store.get_async(
        f"{folder_b}/0.1"
    )
    assert await store.get_async(f"{folder_a}/1.0") == await store.get_async(
        f"{folder_b}/1.0"
    )
    assert await store.get_async(f"{folder_a}/1.1") == await store.get_async(
        f"{folder_b}/1.1"
    )


@pytest.mark.asyncio
async def test_simple(store):
    data = np.arange(0, 16 * 18, dtype="uint16").reshape(16, 18)

    await do_test(
        store,
        data,
        dict(fill_value=1),
        dict(fill_value=1),
    )


@pytest.mark.asyncio
async def test_compressor(store):
    data = np.arange(0, 16 * 18, dtype="uint16").reshape(16, 18)

    await do_test(
        store,
        data,
        dict(
            fill_value=1,
            compressor={"id": "blosc", "cname": "zstd", "clevel": 5},
        ),
        dict(
            fill_value=1,
            compressor=numcodecs.get_codec(
                {"id": "blosc", "cname": "zstd", "clevel": 5}
            ),
        ),
    )


@pytest.mark.asyncio
async def test_filters(store):
    data = np.arange(0, 16 * 18, dtype="uint16").reshape(16, 18)

    await do_test(
        store,
        data,
        dict(fill_value=1, filters=[{"id": "delta", "dtype": data.dtype}]),
        dict(
            fill_value=1,
            compressor=None,
            filters=[numcodecs.get_codec({"id": "delta", "dtype": "<u2"})],
        ),
    )


@pytest.mark.asyncio
async def test_filters_and_compressors(store):
    data = np.arange(0, 16 * 18, dtype="uint16").reshape(16, 18)

    await do_test(
        store,
        data,
        dict(
            fill_value=1,
            compressor={"id": "blosc", "cname": "zstd", "clevel": 5},
            filters=[{"id": "delta", "dtype": data.dtype}],
        ),
        dict(
            fill_value=1,
            compressor=numcodecs.get_codec(
                {"id": "blosc", "cname": "zstd", "clevel": 5}
            ),
            filters=[numcodecs.get_codec({"id": "delta", "dtype": "<u2"})],
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("endian", ["<", ">"])
async def test_endian(store, endian):
    data = np.arange(0, 16 * 18, dtype=f"{endian}u2").reshape(16, 18)

    await do_test(
        store,
        data,
        dict(
            fill_value=1,
            compressor={"id": "blosc", "cname": "zstd", "clevel": 5},
            filters=[{"id": "delta", "dtype": data.dtype}],
        ),
        dict(
            fill_value=1,
            compressor=numcodecs.get_codec(
                {"id": "blosc", "cname": "zstd", "clevel": 5}
            ),
            filters=[numcodecs.get_codec({"id": "delta", "dtype": data.dtype})],
        ),
    )


@pytest.mark.asyncio
async def test_f_order(store):
    data = np.arange(0, 16 * 18, dtype="<u2").reshape(16, 18, order="F")

    a, z2 = await do_test(
        store,
        data,
        dict(
            fill_value=1,
            order="F",
            compressor={"id": "blosc", "cname": "zstd", "clevel": 5},
            filters=[{"id": "delta", "dtype": data.dtype}],
        ),
        dict(
            fill_value=1,
            order="F",
            compressor=numcodecs.get_codec(
                {"id": "blosc", "cname": "zstd", "clevel": 5}
            ),
            filters=[numcodecs.get_codec({"id": "delta", "dtype": data.dtype})],
        ),
    )

    assert (await a.async_[:16, :18].get()).flags.f_contiguous
    assert z2[:16, :18].flags.f_contiguous


@pytest.mark.asyncio
async def test_attributes(store):
    data = np.arange(0, 16 * 18, dtype="<u2").reshape(16, 18)

    await do_test(
        store,
        data,
        dict(fill_value=1, attributes={"hello": "world"}),
        dict(fill_value=1, attributes={"hello": "world"}),
    )
