from shutil import rmtree

import numpy as np
import webknossos as wk
import zarr
from pytest import fixture

import zarrita
from zarrita.sharding import morton_order_iter


@fixture(autouse=True, scope="module")
def clean_folder():
    rmtree("testdata", ignore_errors=True)


def test_sharding():
    ds = wk.Dataset.open("l4_sample")

    def copy(data, path):
        s = zarrita.FileSystemStore("file://./testdata")
        a = zarrita.Array.create(
            s,
            path,
            shape=data.shape,
            chunk_shape=(512, 512, 512),
            dtype=data.dtype,
            fill_value=0,
            codecs=[
                zarrita.codecs.sharding_codec(
                    (32, 32, 32),
                    [zarrita.codecs.transpose_codec("F"), zarrita.codecs.blosc_codec()],
                )
            ],
        )

        a[:, :, :] = data
        read_data = a[:, :, :]
        assert np.array_equal(data, read_data)

    copy(ds.get_layer("color").get_mag(1).read()[0], "l4_sample/color/1")
    # copy(ds.get_layer("segmentation").get_mag(1).read()[0], "l4_sample/segmentation/1")


def test_order_F():
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16), order="F")

    s = zarrita.FileSystemStore("file://./testdata")
    a = zarrita.Array.create(
        s,
        "order_F",
        shape=data.shape,
        chunk_shape=(16, 16),
        dtype=data.dtype,
        fill_value=0,
        codecs=[zarrita.codecs.transpose_codec("F")],
    )

    a[:, :] = data
    read_data = a[:, :]
    assert np.array_equal(data, read_data)
    assert read_data.flags["F_CONTIGUOUS"]
    assert not read_data.flags["C_CONTIGUOUS"]


def test_order_C():
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16), order="C")

    s = zarrita.FileSystemStore("file://./testdata")
    a = zarrita.Array.create(
        s,
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


def test_order_implicitC():
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16), order="F")

    s = zarrita.FileSystemStore("file://./testdata")
    a = zarrita.Array.create(
        s,
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


def test_open():
    s = zarrita.FileSystemStore("file://./testdata")
    a = zarrita.Array.create(
        s,
        "open",
        shape=(16, 16),
        chunk_shape=(16, 16),
        dtype="int32",
        fill_value=0,
    )
    b = zarrita.Array.open(s, "open")
    assert a.metadata == b.metadata


def test_open_sharding():
    s = zarrita.FileSystemStore("file://./testdata")
    a = zarrita.Array.create(
        s,
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
    b = zarrita.Array.open(s, "open_sharding")
    assert a.metadata == b.metadata


def test_simple():
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))
    s = zarrita.FileSystemStore("file://./testdata")
    a = zarrita.Array.create(
        s,
        "simple",
        shape=data.shape,
        chunk_shape=(16, 16),
        dtype=data.dtype,
        fill_value=0,
    )

    a[:, :] = data
    assert np.array_equal(data, a[:, :])


def test_morton():
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


def test_group():
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))
    s = zarrita.FileSystemStore("file://./testdata")
    g = zarrita.Group.create(s, "group")
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


def test_write_partial_chunks():
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))
    s = zarrita.FileSystemStore("file://./testdata")
    a = zarrita.Array.create(
        s,
        "write_partial_chunks",
        shape=data.shape,
        chunk_shape=(20, 20),
        dtype=data.dtype,
        fill_value=1,
    )
    a[0:16, 0:16] = data
    assert np.array_equal(a[0:16, 0:16], data)


def test_write_full_chunks():
    data = np.arange(0, 16 * 16, dtype="uint16").reshape((16, 16))
    s = zarrita.FileSystemStore("file://./testdata")
    a = zarrita.Array.create(
        s,
        "write_full_chunks",
        shape=(16, 16),
        chunk_shape=(20, 20),
        dtype=data.dtype,
        fill_value=1,
    )
    a[0:16, 0:16] = data
    assert np.array_equal(a[0:16, 0:16], data)

    a = zarrita.Array.create(
        s,
        "write_full_chunks",
        shape=(20, 20),
        chunk_shape=(20, 20),
        dtype=data.dtype,
        fill_value=1,
    )
    assert np.all(a[16:20, 16:20] == 1)


def test_write_partial_sharded_chunks():
    data = np.arange(0, 16 * 16, dtype="uint16").reshape((16, 16))
    s = zarrita.FileSystemStore("file://./testdata")
    a = zarrita.Array.create(
        s,
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


def test_delete_empty_chunks():
    data = np.ones((16, 16))
    s = zarrita.FileSystemStore("file://./testdata")
    a = zarrita.Array.create(
        s,
        "delete_empty_chunks",
        shape=data.shape,
        chunk_shape=(32, 32),
        dtype=data.dtype,
        fill_value=1,
    )
    a[0:16, 0:16] = np.zeros((16, 16))
    a[0:16, 0:16] = data
    assert np.array_equal(a[0:16, 0:16], data)
    assert s.get("delete_empty_chunks/c0/0") == None


def test_delete_empty_sharded_chunks():
    s = zarrita.FileSystemStore("file://./testdata")
    a = zarrita.Array.create(
        s,
        "delete_empty_sharded_chunks",
        shape=(16, 16),
        chunk_shape=(8, 16),
        dtype="uint16",
        fill_value=1,
        codecs=[zarrita.codecs.sharding_codec(chunk_shape=(8, 8))],
    )
    a[:, :] = np.zeros((16, 16))
    a[8:, :] = np.ones((8, 16))
    a[:, 8:] = np.ones((16, 8))
    # chunk (0, 0) is full
    # chunks (0, 1), (1, 0), (1, 1) are empty
    # shard (0, 0) is half-full
    # shard (1, 0) is empty

    data = np.ones((16, 16), dtype="uint16")
    data[:8, :8] = 0
    assert np.array_equal(data, a[:, :])
    assert s.get("delete_empty_sharded_chunks/c1/0") == None
    assert len(s.get("delete_empty_sharded_chunks/c0/0")) == 16 * 2 + 8 * 8 * 2


def test_zarr_compat():
    data = np.zeros((16, 18), dtype="uint16")
    s = zarrita.FileSystemStore("file://./testdata")
    a = zarrita.Array.create(
        s,
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

    a[:16, :18] = data
    z2[:16, :18] = data
    assert np.array_equal(data, a[:16, :18])
    assert np.array_equal(data, z2[:16, :18])

    assert s.get("zarr_compat2/0.0") == s.get("zarr_compat3/0.0")
    assert s.get("zarr_compat2/0.1") == s.get("zarr_compat3/0.1")
    assert s.get("zarr_compat2/1.0") == s.get("zarr_compat3/1.0")
    assert s.get("zarr_compat2/1.1") == s.get("zarr_compat3/1.1")


def test_zarr_compat_F():
    data = np.zeros((16, 18), dtype="uint16", order="F")
    s = zarrita.FileSystemStore("file://./testdata")
    a = zarrita.Array.create(
        s,
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

    a[:16, :18] = data
    z2[:16, :18] = data
    assert np.array_equal(data, a[:16, :18])
    assert np.array_equal(data, z2[:16, :18])

    assert s.get("zarr_compatF2/0.0") == s.get("zarr_compatF3/0.0")
    assert s.get("zarr_compatF2/0.1") == s.get("zarr_compatF3/0.1")
    assert s.get("zarr_compatF2/1.0") == s.get("zarr_compatF3/1.0")
    assert s.get("zarr_compatF2/1.1") == s.get("zarr_compatF3/1.1")


def test_dimension_names():
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))
    s = zarrita.FileSystemStore("file://./testdata")
    a = zarrita.Array.create(
        s,
        "dimension_names",
        shape=data.shape,
        chunk_shape=(16, 16),
        dtype=data.dtype,
        fill_value=0,
        dimension_names=("x", "y"),
    )

    assert zarrita.Array.open(s, "dimension_names").metadata.dimension_names == (
        "x",
        "y",
    )

    a = zarrita.Array.create(
        s,
        "dimension_names",
        shape=data.shape,
        chunk_shape=(16, 16),
        dtype=data.dtype,
        fill_value=0,
    )

    assert zarrita.Array.open(s, "dimension_names").metadata.dimension_names == None
