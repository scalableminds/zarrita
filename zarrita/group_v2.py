import json
from typing import Any, Dict, Literal, Optional, Union

from attr import asdict, frozen

from zarrita.array import Array, ArrayRuntimeConfiguration
from zarrita.common import ZARR_JSON, make_cattr
from zarrita.store import Store
from zarrita.sync import sync


@frozen
class GroupV2Metadata:
    zarr_format: Literal[2] = 2


@frozen
class GroupV2:
    metadata: GroupV2Metadata
    store: Store
    path: str
    attributes: Optional[Dict[str, Any]] = None

    @classmethod
    async def create_async(
        cls, store: "Store", path: str, *, attributes: Optional[Dict[str, Any]] = None
    ) -> "GroupV2":
        group = cls(
            metadata=GroupV2Metadata(), attributes=attributes, store=store, path=path
        )
        await group._save_metadata()
        return group

    @classmethod
    def create(
        cls, store: "Store", path: str, *, attributes: Optional[Dict[str, Any]] = None
    ) -> "GroupV2":
        return sync(cls.create_async(store, path, attributes=attributes))

    @classmethod
    async def open_async(cls, store: "Store", path: str) -> "GroupV2":
        zarr_json_bytes = await store.get_async(f"{path}/{ZARR_JSON}")
        assert zarr_json_bytes is not None
        return cls.from_json(store, path, json.loads(zarr_json_bytes))

    @classmethod
    def open(cls, store: "Store", path: str) -> "GroupV2":
        return sync(cls.open_async(store, path))

    @classmethod
    def from_json(cls, store: Store, path: str, zarr_json: Any) -> "GroupV2":
        group = cls(
            metadata=make_cattr().structure(zarr_json, GroupV2Metadata),
            store=store,
            path=path,
        )
        return group

    @staticmethod
    async def open_or_array(store: Store, path: str) -> Union[Array, "GroupV2"]:
        zarr_json_bytes = await store.get_async(f"{path}/{ZARR_JSON}")
        assert zarr_json_bytes is not None
        zarr_json = json.loads(zarr_json_bytes)
        if zarr_json["node_type"] == "groupV2":
            return GroupV2.from_json(store, path, zarr_json)
        if zarr_json["node_type"] == "array":
            return Array.from_json(store, path, zarr_json, ArrayRuntimeConfiguration())
        raise KeyError

    async def _save_metadata(self) -> None:
        await self.store.set_async(
            f"{self.path}/{ZARR_JSON}",
            json.dumps(asdict(self.metadata)).encode(),
        )

    def _dereference_path(self, path: str) -> str:
        assert isinstance(path, str)
        path = f"{self.path}/{path}"
        if len(path) > 1:
            assert path[-1] != "/"
        return path

    async def get_async(self, path: str) -> Union[Array, "GroupV2"]:
        path = self._dereference_path(path)
        return await GroupV2.open_or_array(self.store, path)

    def __getitem__(self, path: str) -> Union[Array, "GroupV2"]:
        return sync(self.get_async(path))

    async def create_group_async(self, path: str, **kwargs) -> "GroupV2":
        path = self._dereference_path(path)
        return await GroupV2.create_async(self.store, path, **kwargs)

    def create_group(self, path: str, **kwargs) -> "GroupV2":
        return sync(self.create_group_async(path))

    async def create_array_async(self, path: str, **kwargs) -> Array:
        path = self._dereference_path(path)
        return await Array.create_async(self.store, path, **kwargs)

    def create_array(self, path: str, **kwargs) -> "Array":
        return sync(self.create_array_async(path, **kwargs))

    def __repr__(self):
        path = self.path
        return f"<Group_v2 {path}>"
