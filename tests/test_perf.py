import os
from pathlib import Path
from shutil import rmtree
from timeit import default_timer
from typing import List, Tuple

import numpy as np
import pytest
import wkw
import zarr
from numcodecs import Blosc
from pytest import fixture

from zarrita import Array, LocalStore, Store, codecs, runtime_configuration

TEST_SIZE = int(os.environ.get("TEST_SIZE", "1024"))
CHUNK_SIZE = 32
PARTIAL_SELECTION = (
    slice(32, 32 + 32),
    slice(32, 32 + 32),
    slice(32, 32 + 32),
)

TESTDATA: List[Tuple[str, np.ndarray]] = [
    (
        "color",
        wkw.Dataset.open("l4_sample/color/1").read(
            (3072, 3072, 512), (TEST_SIZE, TEST_SIZE, TEST_SIZE)
        )[0],
    ),
    (
        "segmentation",
        wkw.Dataset.open("l4_sample/segmentation/1").read(
            (3072, 3072, 512), (TEST_SIZE, TEST_SIZE, TEST_SIZE)
        )[0],
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
        chunk_shape=(512, 512, 512),
        dtype=testdata.dtype,
        fill_value=0,
        codecs=[
            codecs.sharding_codec(
                (CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE),
                [
                    codecs.transpose_codec("F"),
                    codecs.bytes_codec(),
                    codecs.blosc_codec(
                        cname="zstd",
                        clevel=5,
                        shuffle="noshuffle",
                        blocksize=0,
                        typesize=testdata.dtype.itemsize,
                    ),
                ],
            )
        ],
        runtime_configuration=runtime_configuration("F"),
    )

    start = default_timer()
    a[:, :, :] = testdata
    print(f"  zarrita WRITE {layer_name} - {default_timer() - start:.2f}s")

    path = Path("testdata") / "l4_sample_zarrita_sharding" / layer_name
    print(
        f"  zarrita STORAGE {folder_disk_usage(path)/1000000:,.2f} MB "
        + f"- {folder_inodes(path)} inodes"
    )

    start = default_timer()
    readback_data = a[
        0 : testdata.shape[0], 0 : testdata.shape[1], 0 : testdata.shape[2]
    ]
    print(f"  zarrita READ {layer_name} - {default_timer() - start:.2f}s")
    assert np.array_equal(readback_data, testdata)

    start = default_timer()
    partial_data = a[PARTIAL_SELECTION]
    print(f"  zarrita PARTIAL READ {layer_name} - {default_timer() - start:.5f}s")
    assert np.array_equal(partial_data, testdata[PARTIAL_SELECTION])


@pytest.mark.parametrize("layer_name,testdata", TESTDATA)
@pytest.mark.parametrize("codec", ["lz4"])
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
            file_len=TEST_SIZE // CHUNK_SIZE,
            block_len=CHUNK_SIZE,
        ),
    ) as ds:
        start = default_timer()
        ds.write((0, 0, 0), testdata)
        print(f"  wkw WRITE {layer_name} - {default_timer() - start:.2f}s")

        print(
            f"  wkw STORAGE {folder_disk_usage(path)/1000000:,.2f} MB - "
            + f"{folder_inodes(path)} inodes"
        )

        start = default_timer()
        readback_data = ds.read((0, 0, 0), testdata.shape)[0]
        print(f"  wkw READ {layer_name} - {default_timer() - start:.2f}s")
        assert np.array_equal(readback_data, testdata)

        start = default_timer()
        partial_data = ds.read(
            tuple(a.start for a in PARTIAL_SELECTION),
            tuple(a.stop - a.start for a in PARTIAL_SELECTION),
        )[0]
        print(f"  wkw PARTIAL READ {layer_name} - {default_timer() - start:.5f}s")
        assert np.array_equal(partial_data, testdata[PARTIAL_SELECTION])


@pytest.mark.parametrize("layer_name,testdata", TESTDATA)
def test_zarr(folder: Path, layer_name: str, testdata: np.ndarray):
    print("")
    path = folder / "l4_sample_zarr" / layer_name
    a = zarr.create(
        shape=testdata.shape,
        chunks=(512, 512, 512),
        dtype=testdata.dtype,
        compressor=Blosc(
            cname="zstd",
            clevel=5,
            shuffle=Blosc.NOSHUFFLE,
            blocksize=0,
        ),
        fill_value=0,
        order="F",
        store=str(path),
    )
    start = default_timer()
    a[:, :, :] = testdata
    print(f"  zarr WRITE {layer_name} - {default_timer() - start:.2f}s")

    print(
        f"  zarr STORAGE {folder_disk_usage(path)/1000000:,.2f} MB - "
        + f"{folder_inodes(path)} inodes"
    )

    start = default_timer()
    readback_data = a[
        0 : testdata.shape[0], 0 : testdata.shape[1], 0 : testdata.shape[2]
    ]
    print(f"  zarr READ {layer_name} - {default_timer() - start:.2f}s")
    assert np.array_equal(readback_data, testdata)

    start = default_timer()
    partial_data = a[PARTIAL_SELECTION]
    print(f"  zarr PARTIAL READ {layer_name} - {default_timer() - start:.5f}s")
    assert np.array_equal(partial_data, testdata[PARTIAL_SELECTION])
