import asyncio
import json
from typing import Any, Dict, Literal, Optional, Union

from attr import asdict, frozen

from zarrita.array_v2 import ArrayV2
from zarrita.common import ZARRAY_JSON, ZATTRS_JSON, ZGROUP_JSON, make_cattr
from zarrita.store import StoreLike, StorePath, make_store_path
from zarrita.sync import sync


@frozen
class GroupV2Metadata:
    zarr_format: Literal[2] = 2


@frozen
class GroupV2:
    metadata: GroupV2Metadata
    store_path: StorePath
    attributes: Optional[Dict[str, Any]] = None

    @classmethod
    async def create_async(
        cls,
        store: StoreLike,
        *,
        attributes: Optional[Dict[str, Any]] = None,
        exists_ok: bool = False,
    ) -> "GroupV2":
        store_path = make_store_path(store)
        if not exists_ok:
            assert not await (store_path / ZGROUP_JSON).exists_async()
        group = cls(
            metadata=GroupV2Metadata(), attributes=attributes, store_path=store_path
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
    ) -> "GroupV2":
        return sync(cls.create_async(store, attributes=attributes, exists_ok=exists_ok))

    @classmethod
    async def open_async(cls, store: StoreLike) -> "GroupV2":
        store_path = make_store_path(store)
        zgroup_bytes = await (store_path / ZGROUP_JSON).get_async()
        assert zgroup_bytes is not None
        zattrs_bytes = await (store_path / ZATTRS_JSON).get_async()
        metadata = json.loads(zgroup_bytes)
        attributes = json.loads(zattrs_bytes) if zattrs_bytes is not None else None

        return cls.from_json(store_path, metadata, attributes)

    @classmethod
    def open(cls, store_path: StorePath) -> "GroupV2":
        return sync(cls.open_async(store_path))

    @classmethod
    def from_json(
        cls,
        store_path: StorePath,
        zarr_json: Any,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> "GroupV2":
        group = cls(
            metadata=make_cattr().structure(zarr_json, GroupV2Metadata),
            store_path=store_path,
            attributes=attributes,
        )
        return group

    @staticmethod
    async def open_or_array(store: StoreLike) -> Union[ArrayV2, "GroupV2"]:
        store_path = make_store_path(store)
        zgroup_bytes, zattrs_bytes = await asyncio.gather(
            (store_path / ZGROUP_JSON).get_async(),
            (store_path / ZATTRS_JSON).get_async(),
        )
        attributes = json.loads(zattrs_bytes) if zattrs_bytes is not None else None
        if zgroup_bytes is not None:
            return GroupV2.from_json(store_path, json.loads(zgroup_bytes), attributes)
        zarray_bytes = await (store_path / ZARRAY_JSON).get_async()
        if zarray_bytes is not None:
            return ArrayV2.from_json(store_path, json.loads(zarray_bytes), attributes)
        raise KeyError

    async def _save_metadata(self) -> None:
        await (self.store_path / ZGROUP_JSON).set_async(
            json.dumps(asdict(self.metadata)).encode(),
        )
        if self.attributes is not None and len(self.attributes) > 0:
            await (self.store_path / ZATTRS_JSON).set_async(
                json.dumps(self.attributes).encode(),
            )
        else:
            await (self.store_path / ZATTRS_JSON).delete_async()

    async def get_async(self, path: str) -> Union[ArrayV2, "GroupV2"]:
        return await self.__class__.open_or_array(self.store_path / path)

    def __getitem__(self, path: str) -> Union[ArrayV2, "GroupV2"]:
        return sync(self.get_async(path))

    async def create_group_async(self, path: str, **kwargs) -> "GroupV2":
        return await self.__class__.create_async(self.store_path / path, **kwargs)

    def create_group(self, path: str, **kwargs) -> "GroupV2":
        return sync(self.create_group_async(path))

    async def create_array_async(self, path: str, **kwargs) -> ArrayV2:
        return await ArrayV2.create_async(self.store_path / path, **kwargs)

    def create_array(self, path: str, **kwargs) -> "ArrayV2":
        return sync(self.create_array_async(path, **kwargs))

    def __repr__(self):
        return f"<Group_v2 {self.store_path}>"
