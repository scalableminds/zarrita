import json
from typing import Any, Dict, Literal, Optional, Union

from attr import asdict, field, frozen

from zarrita.array import Array, ArrayRuntimeConfiguration
from zarrita.common import ZARR_JSON, make_cattr
from zarrita.store import Store
from zarrita.sync import sync


@frozen
class GroupMetadata:
    attributes: Dict[str, Any] = field(factory=dict)
    zarr_format: Literal[3] = 3
    node_type: Literal["group"] = "group"


@frozen
class Group:
    metadata: GroupMetadata
    store: Store
    path: str

    @classmethod
    async def create_async(
        cls,
        store: "Store",
        path: str,
        *,
        attributes: Optional[Dict[str, Any]] = None,
        exists_ok: bool = False,
    ) -> "Group":
        if not exists_ok:
            assert not await store.exists_async(f"{path}/{ZARR_JSON}")
        group = cls(
            metadata=GroupMetadata(attributes=attributes or {}),
            store=store,
            path=path,
        )
        await group._save_metadata()
        return group

    @classmethod
    def create(
        cls,
        store: "Store",
        path: str,
        *,
        attributes: Optional[Dict[str, Any]] = None,
        exists_ok: bool = False,
    ) -> "Group":
        return sync(
            cls.create_async(store, path, attributes=attributes, exists_ok=exists_ok)
        )

    @classmethod
    async def open_async(cls, store: "Store", path: str) -> "Group":
        zarr_json_bytes = await store.get_async(f"{path}/{ZARR_JSON}")
        assert zarr_json_bytes is not None
        return cls.from_json(store, path, json.loads(zarr_json_bytes))

    @classmethod
    def open(cls, store: "Store", path: str) -> "Group":
        return sync(cls.open_async(store, path))

    @classmethod
    def from_json(cls, store: Store, path: str, zarr_json: Any) -> "Group":
        group = cls(
            metadata=make_cattr().structure(zarr_json, GroupMetadata),
            store=store,
            path=path,
        )
        return group

    @classmethod
    async def open_or_array(
        cls,
        store: Store,
        path: str,
        runtime_configuration: Optional[ArrayRuntimeConfiguration] = None,
    ) -> Union[Array, "Group"]:
        zarr_json_bytes = await store.get_async(f"{path}/{ZARR_JSON}")
        if zarr_json_bytes is None:
            raise KeyError
        zarr_json = json.loads(zarr_json_bytes)
        if zarr_json["node_type"] == "group":
            return cls.from_json(store, path, zarr_json)
        if zarr_json["node_type"] == "array":
            return Array.from_json(
                store,
                path,
                zarr_json,
                runtime_configuration=runtime_configuration
                or ArrayRuntimeConfiguration(),
            )
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

    async def get_async(self, path: str) -> Union[Array, "Group"]:
        path = self._dereference_path(path)
        return await self.__class__.open_or_array(self.store, path)

    def __getitem__(self, path: str) -> Union[Array, "Group"]:
        return sync(self.get_async(path))

    async def create_group_async(self, path: str, **kwargs) -> "Group":
        path = self._dereference_path(path)
        return await self.__class__.create_async(self.store, path, **kwargs)

    def create_group(self, path: str, **kwargs) -> "Group":
        return sync(self.create_group_async(path))

    async def create_array_async(self, path: str, **kwargs) -> Array:
        path = self._dereference_path(path)
        return await Array.create_async(self.store, path, **kwargs)

    def create_array(self, path: str, **kwargs) -> Array:
        return sync(self.create_array_async(path, **kwargs))

    def __repr__(self):
        path = self.path
        return f"<Group {path}>"
