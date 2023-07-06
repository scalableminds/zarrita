from __future__ import annotations

import json
from typing import Any, Dict, Literal, Optional, Union

from attr import asdict, field, frozen

from zarrita.array import Array, ArrayRuntimeConfiguration
from zarrita.common import ZARR_JSON, make_cattr
from zarrita.store import StoreLike, StorePath, make_store_path
from zarrita.sync import sync


@frozen
class GroupMetadata:
    attributes: Dict[str, Any] = field(factory=dict)
    zarr_format: Literal[3] = 3
    node_type: Literal["group"] = "group"


@frozen
class Group:
    metadata: GroupMetadata
    store_path: StorePath

    @classmethod
    async def create_async(
        cls,
        store: StoreLike,
        *,
        attributes: Optional[Dict[str, Any]] = None,
        exists_ok: bool = False,
    ) -> Group:
        store_path = make_store_path(store)
        if not exists_ok:
            assert not await (store_path / ZARR_JSON).exists_async()
        group = cls(
            metadata=GroupMetadata(attributes=attributes or {}),
            store_path=store_path,
        )
        await group._save_metadata()
        return group

    @classmethod
    def create(
        cls,
        store: StoreLike,
        *,
        attributes: Optional[Dict[str, Any]] = None,
        exists_ok: bool = False,
    ) -> Group:
        return sync(cls.create_async(store, attributes=attributes, exists_ok=exists_ok))

    @classmethod
    async def open_async(cls, store: StoreLike) -> Group:
        store_path = make_store_path(store)
        zarr_json_bytes = await (store_path / ZARR_JSON).get_async()
        assert zarr_json_bytes is not None
        return cls.from_json(store_path, json.loads(zarr_json_bytes))

    @classmethod
    def open(cls, store: StoreLike) -> Group:
        return sync(cls.open_async(store))

    @classmethod
    def from_json(cls, store_path: StorePath, zarr_json: Any) -> Group:
        group = cls(
            metadata=make_cattr().structure(zarr_json, GroupMetadata),
            store_path=store_path,
        )
        return group

    @classmethod
    async def open_or_array(
        cls,
        store: StoreLike,
        runtime_configuration: Optional[ArrayRuntimeConfiguration] = None,
    ) -> Union[Array, Group]:
        store_path = make_store_path(store)
        zarr_json_bytes = await (store_path / ZARR_JSON).get_async()
        if zarr_json_bytes is None:
            raise KeyError
        zarr_json = json.loads(zarr_json_bytes)
        if zarr_json["node_type"] == "group":
            return cls.from_json(store_path, zarr_json)
        if zarr_json["node_type"] == "array":
            return Array.from_json(
                store_path,
                zarr_json,
                runtime_configuration=runtime_configuration
                or ArrayRuntimeConfiguration(),
            )
        raise KeyError

    async def _save_metadata(self) -> None:
        await (self.store_path / ZARR_JSON).set_async(
            json.dumps(asdict(self.metadata)).encode(),
        )

    async def get_async(self, path: str) -> Union[Array, Group]:
        return await self.__class__.open_or_array(self.store_path / path)

    def __getitem__(self, path: str) -> Union[Array, Group]:
        return sync(self.get_async(path))

    async def create_group_async(self, path: str, **kwargs) -> Group:
        return await self.__class__.create_async(self.store_path / path, **kwargs)

    def create_group(self, path: str, **kwargs) -> Group:
        return sync(self.create_group_async(path))

    async def create_array_async(self, path: str, **kwargs) -> Array:
        return await Array.create_async(self.store_path / path, **kwargs)

    def create_array(self, path: str, **kwargs) -> Array:
        return sync(self.create_array_async(path, **kwargs))

    def __repr__(self):
        return f"<Group {self.store_path}>"
