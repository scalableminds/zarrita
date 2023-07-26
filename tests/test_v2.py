import json
from pathlib import Path
from shutil import rmtree
from typing import Any, Literal

import numcodecs
import numpy as np
import pytest
import zarr
from pytest import fixture

from zarrita import Array, ArrayV2, Group, GroupV2, LocalStore, Store, open_auto_async
from zarrita.codecs import BloscCodec, EndianCodec, TransposeCodec


@fixture
def store() -> Store:
    path = Path("testdata")
    rmtree(path, ignore_errors=True)
    return LocalStore(path)


async def do_test(
    store: Store,
    data: np.ndarray,
    zarrita_kwargs: Any,
    zarr_kwargs: Any,
):
    # setup
    zarrita_dtype = zarrita_kwargs.pop("dtype", data.dtype)
    zarrita_array = await ArrayV2.create_async(
        store / "zarrita_v2",
        shape=data.shape,
        dtype=zarrita_dtype,
        chunks=(10, 10),
        **{  # type: ignore
            **dict(dimension_separator=".", compressor=None, filters=None),
            **zarrita_kwargs,
        },
    )

    zarr_dtype = zarr_kwargs.pop("dtype", data.dtype)
    zarr_attrs = zarr_kwargs.pop("attributes", None)
    zarr_array = zarr.create(
        store="testdata/zarr_v2",
        shape=data.shape,
        dtype=zarr_dtype,
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
    zarrita_array = await ArrayV2.open_async(store / "zarrita_v2")
    assert np.array_equal(data, await zarrita_array.async_[:16, :18].get())

    # open zarrita array with zarr
    zarrita_array_via_zarr = zarr.open_array("testdata/zarrita_v2")
    assert np.array_equal(data, zarrita_array_via_zarr[:16, :18])

    # open zarr array via zarrita
    zarr_array_via_zarrita = await ArrayV2.open_async(store / "zarr_v2")
    assert np.array_equal(data, zarr_array_via_zarrita[:16, :18])

    await check_zarr2_files(store, "zarrita_v2", "zarr_v2")

    return zarrita_array, zarr_array


async def check_zarr2_files(store: Store, folder_a: str, folder_b: str):
    zarray_a_bytes = await store.get_async(f"{folder_a}/.zarray")
    zarray_b_bytes = await store.get_async(f"{folder_b}/.zarray")
    assert zarray_a_bytes is not None
    assert zarray_b_bytes is not None
    assert json.loads(zarray_a_bytes) == json.loads(zarray_b_bytes)

    zattrs_a_bytes = await store.get_async(f"{folder_a}/.zattrs")
    zattrs_b_bytes = await store.get_async(f"{folder_b}/.zattrs")
    if zattrs_b_bytes is not None:
        assert zattrs_a_bytes is not None
        assert json.loads(zattrs_a_bytes) == json.loads(zattrs_b_bytes)
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
async def test_simple(store: Store):
    data = np.arange(0, 16 * 18, dtype="uint16").reshape(16, 18)

    await do_test(
        store,
        data,
        dict(fill_value=1),
        dict(fill_value=1),
    )


@pytest.mark.asyncio
async def test_compressor(store: Store):
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
async def test_filters(store: Store):
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
async def test_filters_and_compressors(store: Store):
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
@pytest.mark.parametrize("input_endian", ["<u2", ">u2"])
@pytest.mark.parametrize("store_endian", ["<u2", ">u2"])
async def test_endian(
    store: Store,
    input_endian: Literal["<u2", ">u2"],
    store_endian: Literal["<u2", ">u2"],
):
    data = np.arange(0, 16 * 18, dtype=input_endian).reshape(16, 18)

    await do_test(
        store,
        data,
        dict(
            dtype=np.dtype(store_endian),
            fill_value=1,
            compressor={"id": "blosc", "cname": "zstd", "clevel": 5},
            filters=[{"id": "delta", "dtype": store_endian}],
        ),
        dict(
            dtype=np.dtype(store_endian),
            fill_value=1,
            compressor=numcodecs.get_codec(
                {"id": "blosc", "cname": "zstd", "clevel": 5}
            ),
            filters=[numcodecs.get_codec({"id": "delta", "dtype": store_endian})],
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("input_order", ["F", "C"])
@pytest.mark.parametrize("store_order", ["F", "C"])
async def test_order(
    store: Store, input_order: Literal["F", "C"], store_order: Literal["F", "C"]
):
    data = np.arange(0, 16 * 18, dtype="<u2").reshape(16, 18, order=input_order)

    zarrita_array, zarr_array = await do_test(
        store,
        data,
        dict(
            fill_value=1,
            order=store_order,
            compressor={"id": "blosc", "cname": "zstd", "clevel": 5},
            filters=[{"id": "delta", "dtype": data.dtype}],
        ),
        dict(
            fill_value=1,
            order=store_order,
            compressor=numcodecs.get_codec(
                {"id": "blosc", "cname": "zstd", "clevel": 5}
            ),
            filters=[numcodecs.get_codec({"id": "delta", "dtype": data.dtype})],
        ),
    )

    zarrita_data = await zarrita_array.async_[:16, :18].get()
    zarr_data = zarr_array[:16, :18]
    if store_order == "F":
        assert zarrita_data.flags.f_contiguous
        assert not zarrita_data.flags.c_contiguous
        assert zarr_data.flags.f_contiguous
        assert not zarr_data.flags.c_contiguous
    else:
        assert not zarrita_data.flags.f_contiguous
        assert zarrita_data.flags.c_contiguous
        assert not zarr_data.flags.f_contiguous
        assert zarr_data.flags.c_contiguous


@pytest.mark.asyncio
async def test_attributes(store: Store):
    data = np.arange(0, 16 * 18, dtype="<u2").reshape(16, 18)

    await do_test(
        store,
        data,
        dict(fill_value=1, attributes={"hello": "world"}),
        dict(fill_value=1, attributes={"hello": "world"}),
    )


@pytest.mark.asyncio
async def test_no_attributes(store: Store):
    data = np.arange(0, 16 * 18, dtype="<u2").reshape(16, 18)

    await do_test(
        store,
        data,
        dict(fill_value=1),
        dict(fill_value=1),
    )

    assert await store.get_async("zarrita_v2/.zattrs") is None
    assert await store.get_async("zarr_v2/.zattrs") is None


@pytest.mark.asyncio
async def test_empty_attributes(store: Store):
    data = np.arange(0, 16 * 18, dtype="<u2").reshape(16, 18)

    await do_test(
        store,
        data,
        dict(fill_value=1, attributes={}),
        dict(fill_value=1, attributes={}),
    )

    assert await store.get_async("zarrita_v2/.zattrs") is None
    assert await store.get_async("zarr_v2/.zattrs") is None


def test_group(store: Store):
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))

    g = GroupV2.create(store / "group")
    g.create_array(
        "array",
        shape=data.shape,
        chunks=(16, 16),
        dtype=data.dtype,
        fill_value=0,
    )
    g.create_group("group2")

    assert isinstance(g["array"], ArrayV2)
    assert isinstance(g["group2"], GroupV2)


@pytest.mark.asyncio
async def test_open_auto(store: Store):
    data = np.arange(0, 16 * 18, dtype="<u2").reshape(16, 18)

    group_v2 = await GroupV2.create_async(store / "group_v2")
    await group_v2.create_array_async(
        "zarrita_v2",
        shape=data.shape,
        dtype=data.dtype,
        chunks=(10, 10),
    )
    group_v3 = await Group.create_async(store / "group_v3")
    await group_v3.create_array_async(
        "zarrita_v3",
        shape=data.shape,
        dtype=data.dtype,
        chunk_shape=(10, 10),
    )

    assert isinstance(
        await Array.open_auto_async(store / "group_v2/zarrita_v2"), ArrayV2
    )
    assert isinstance(await Array.open_auto_async(store / "group_v3/zarrita_v3"), Array)

    assert isinstance(await open_auto_async(store / "group_v2"), GroupV2)
    assert isinstance(await open_auto_async(store / "group_v2/zarrita_v2"), ArrayV2)
    assert isinstance(await open_auto_async(store / "group_v3"), Group)
    assert isinstance(await open_auto_async(store / "group_v3/zarrita_v3"), Array)


def test_exists_ok(store: Store):
    ArrayV2.create(
        store / "exists_ok",
        shape=(16, 16),
        chunks=(16, 16),
        dtype=np.dtype("uint8"),
    )
    with pytest.raises(AssertionError):
        ArrayV2.create(
            store / "exists_ok",
            shape=(16, 16),
            chunks=(16, 16),
            dtype=np.dtype("uint8"),
        )
    ArrayV2.create(
        store / "exists_ok",
        shape=(16, 16),
        chunks=(16, 16),
        dtype=np.dtype("uint8"),
        exists_ok=True,
    )


@pytest.mark.asyncio
async def test_resize(store: Store):
    data = np.zeros((16, 18), dtype="uint16")

    a = await ArrayV2.create_async(
        store / "resize",
        shape=data.shape,
        chunks=(10, 10),
        dtype=data.dtype,
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


def test_update_attributes_array(store: Store):
    data = np.zeros((16, 18), dtype="uint16")

    a = ArrayV2.create(
        store / "update_attributes",
        shape=data.shape,
        chunks=(10, 10),
        dtype=data.dtype,
        fill_value=1,
        attributes={"hello": "world"},
    )

    a = ArrayV2.open(store / "update_attributes")
    assert a.attributes is not None and a.attributes["hello"] == "world"

    a.update_attributes({"hello": "zarrita"})

    a = ArrayV2.open(store / "update_attributes")
    assert a.attributes is not None and a.attributes["hello"] == "zarrita"


def test_update_attributes_group(store: Store):
    g = GroupV2.create(store / "update_attributes_group", attributes={"hello": "world"})

    g = GroupV2.open(store / "update_attributes_group")
    assert g.attributes is not None and g.attributes["hello"] == "world"

    g.update_attributes({"hello": "zarrita"})

    g = GroupV2.open(store / "update_attributes_group")
    assert g.attributes is not None and g.attributes["hello"] == "zarrita"


def test_convert_to_v3_array(store: Store):
    data = np.zeros((16, 18), dtype="uint16")

    a = ArrayV2.create(
        store / "convert_v3",
        shape=data.shape,
        chunks=(10, 10),
        dtype=data.dtype,
        fill_value=1,
        dimension_separator="/",
        attributes={"hello": "world"},
        order="F",
        compressor={
            "id": "blosc",
            "cname": "zstd",
            "clevel": 3,
            "blocksize": 1,
            "shuffle": 1,
        },
    )

    a.convert_to_v3()

    # Load anew
    a3 = Array.open(store / "convert_v3")

    assert a3.metadata.shape == a.metadata.shape
    assert a3.metadata.chunk_grid.configuration.chunk_shape == a.metadata.chunks
    assert a3.metadata.data_type.to_numpy_shortname() == a.metadata.dtype.str[1:]
    assert a3.metadata.fill_value == a.metadata.fill_value
    assert a3.metadata.chunk_key_encoding.name == "v2"
    assert a3.metadata.chunk_key_encoding.configuration.separator == "/"
    assert a3.metadata.attributes["hello"] == "world"
    assert any(
        isinstance(c, TransposeCodec) and c.configuration.order == "F"
        for c in a3.codec_pipeline.codecs
    )
    assert any(
        isinstance(c, EndianCodec) and c.configuration.endian == "little"
        for c in a3.codec_pipeline.codecs
    )
    assert any(
        isinstance(c, BloscCodec)
        and c.configuration.cname == "zstd"
        and c.configuration.clevel == 3
        and c.configuration.blocksize == 1
        and c.configuration.typesize == 2
        and c.configuration.shuffle == "shuffle"
        for c in a3.codec_pipeline.codecs
    )


def test_convert_to_v3_group(store: Store):
    g = GroupV2.create(store / "convert_v3_group", attributes={"hello": "world"})
    g.convert_to_v3()
    g3 = Group.open(store / "convert_v3_group")
    assert g3.metadata.attributes["hello"] == "world"
