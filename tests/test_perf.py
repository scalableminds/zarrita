import os
from pathlib import Path
from shutil import rmtree
from timeit import default_timer
from typing import List, Tuple

import numpy as np
import pytest
import webknossos as wk
import wkw
import zarr
from numcodecs import Blosc
from pytest import fixture

from zarrita import Array, LocalStore, Store, codecs
from zarrita.array import runtime_configuration

TEST_SIZE = int(os.environ.get("TEST_SIZE", "1024"))

TESTDATA: List[Tuple[str, np.ndarray]] = [
    (
        "color",
        wk.Dataset.open("l4_sample")
        .get_layer("color")
        .get_mag(1)
        .read()[0][:TEST_SIZE, :TEST_SIZE, :TEST_SIZE],
    ),
    (
        "segmentation",
        wk.Dataset.open("l4_sample")
        .get_layer("segmentation")
        .get_mag(1)
        .read()[0][:TEST_SIZE, :TEST_SIZE, :TEST_SIZE],
    ),
]


@fixture
def folder() -> Path:
    path = Path("testdata")
    rmtree(path, ignore_errors=True)
    return path


@fixture
def store(folder: Path) -> Store:
    return LocalStore(folder)


def folder_disk_usage(folder: Path) -> int:
    return sum(f.stat().st_size for f in folder.glob("**/*") if f.is_file())


def folder_inodes(folder: Path) -> int:
    return sum(1 for _ in folder.glob("**/*"))


@pytest.mark.parametrize("layer_name, testdata", TESTDATA)
def test_zarrita_sharding(store: Store, layer_name: str, testdata: np.ndarray):
    print("")
    a = Array.create(
        store / "l4_sample_zarrita_sharding" / layer_name,
        shape=testdata.shape,
        chunk_shape=(1024, 1024, 1024),
        dtype=testdata.dtype,
        fill_value=0,
        codecs=[
            codecs.sharding_codec(
                (32, 32, 32),
                [
                    codecs.transpose_codec("F"),
                    codecs.blosc_codec(),
                ],
            )
        ],
        runtime_configuration=runtime_configuration("F"),
    )

    start = default_timer()
    a[:, :, :] = testdata
    print(f"  zarrita WRITE {layer_name} - {default_timer() - start:.2f}s")

    start = default_timer()
    readback_data = a[
        0 : testdata.shape[0], 0 : testdata.shape[1], 0 : testdata.shape[2]
    ]
    print(f"  zarrita READ {layer_name} - {default_timer() - start:.2f}s")

    path = Path("testdata") / "l4_sample_zarrita_sharding" / layer_name
    print(
        f"  zarrita STORAGE {folder_disk_usage(path)/1000000:,.2f} MB "
        + f"- {folder_inodes(path)} inodes"
    )

    assert np.array_equal(readback_data, testdata)


@pytest.mark.parametrize("layer_name,testdata", TESTDATA)
@pytest.mark.parametrize("codec", ["lz4", "lz4hc"])
def test_wkw(folder: Path, layer_name: str, testdata: np.ndarray, codec: str):
    print("")
    path = folder / "l4_sample_wkw" / codec / layer_name

    with wkw.Dataset.create(
        str(path),
        wkw.Header(
            testdata.dtype,
            block_type=wkw.Header.BLOCK_TYPE_LZ4
            if codec == "lz4"
            else wkw.Header.BLOCK_TYPE_LZ4HC,
            file_len=TEST_SIZE // 32,
        ),
    ) as ds:
        start = default_timer()
        ds.write((0, 0, 0), testdata)
        print(f"  wkw WRITE {layer_name} - {default_timer() - start:.2f}s")

        start = default_timer()
        readback_data = ds.read((0, 0, 0), testdata.shape)[0]
        print(f"  wkw READ {layer_name} - {default_timer() - start:.2f}s")

        print(
            f"  wkw STORAGE {folder_disk_usage(path)/1000000:,.2f} MB - "
            + f"{folder_inodes(path)} inodes"
        )

        assert np.array_equal(readback_data, testdata)


@pytest.mark.parametrize("layer_name,testdata", TESTDATA)
def test_zarr(folder: Path, layer_name: str, testdata: np.ndarray):
    print("")
    path = folder / "l4_sample_zarr" / layer_name
    a = zarr.create(
        shape=testdata.shape,
        chunks=(32, 32, 32),
        dtype=testdata.dtype,
        compressor=Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE),
        fill_value=0,
        order="F",
        store=str(path),
    )
    start = default_timer()
    a[:, :, :] = testdata
    print(f"  zarr WRITE {layer_name} - {default_timer() - start:.2f}s")

    start = default_timer()
    readback_data = a[
        0 : testdata.shape[0], 0 : testdata.shape[1], 0 : testdata.shape[2]
    ]
    print(f"  zarr READ {layer_name} - {default_timer() - start:.2f}s")

    print(
        f"  zarr STORAGE {folder_disk_usage(path)/1000000:,.2f} MB - "
        + f"{folder_inodes(path)} inodes"
    )

    assert np.array_equal(readback_data, testdata)
