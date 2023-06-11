from pathlib import Path

from s3fs import S3FileSystem
from upath import UPath

from zarrita.store import LocalStore, RemoteStore, make_store_path


def test_make_store_path():
    # str
    store_path = make_store_path("testdata")
    assert isinstance(store_path.store, LocalStore)
    assert store_path.store.root == Path("testdata")

    # local store
    store_path = make_store_path(LocalStore("testdata"))
    assert isinstance(store_path.store, LocalStore)
    assert store_path.store.root == Path("testdata")

    # remote store
    store_path = make_store_path(RemoteStore("s3://bucket/test"))
    assert isinstance(store_path.store, RemoteStore)
    assert isinstance(store_path.store.fs, S3FileSystem)
    assert store_path.store.root == "bucket/test"

    # path
    store_path = make_store_path(Path("testdata"))
    assert isinstance(store_path.store, LocalStore)
    assert store_path.store.root == Path("testdata")

    # upath
    store_path = make_store_path(UPath("s3://bucket/test"))
    assert isinstance(store_path.store, RemoteStore)
    assert isinstance(store_path.store.fs, S3FileSystem)
    assert store_path.store.root == "bucket/test"

    # store path
    store_path = make_store_path(LocalStore("testdata") / "test")
    assert isinstance(store_path.store, LocalStore)
    assert store_path.store.root == Path("testdata")
    assert store_path.path == "test"


def test_path_building():
    store = LocalStore("testdata")
    assert str(store / "test") == "file://testdata/test"
    assert repr(store / "test") == "StorePath(LocalStore, 'file://testdata/test')"

    store = RemoteStore("s3://bucket")
    assert str(store / "test2") == "s3://bucket/test2"
    assert repr(store / "test2") == "StorePath(RemoteStore, 's3://bucket/test2')"

    store = LocalStore("testdata")
    assert str(store / "test/test") == "file://testdata/test/test"
    assert (
        repr(store / "test/test")
        == "StorePath(LocalStore, 'file://testdata/test/test')"
    )
