import asyncio
import json
from typing import Any, Dict, Literal, Optional, Union

from attr import asdict, frozen

from zarrita.array_v2 import ArrayV2
from zarrita.common import ZARRAY_JSON, ZATTRS_JSON, ZGROUP_JSON, make_cattr
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
        cls,
        store: "Store",
        path: str,
        *,
        attributes: Optional[Dict[str, Any]] = None,
        exists_ok: bool = False,
    ) -> "GroupV2":
        if not exists_ok:
            assert not await store.exists_async(f"{path}/{ZGROUP_JSON}")
        group = cls(
            metadata=GroupV2Metadata(), attributes=attributes, store=store, path=path
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
    ) -> "GroupV2":
        return sync(
            cls.create_async(store, path, attributes=attributes, exists_ok=exists_ok)
        )

    @classmethod
    async def open_async(cls, store: "Store", path: str) -> "GroupV2":
        zgroup_bytes = await store.get_async(f"{path}/{ZGROUP_JSON}")
        assert zgroup_bytes is not None
        zattrs_bytes = await store.get_async(f"{path}/{ZATTRS_JSON}")
        metadata = json.loads(zgroup_bytes)
        attributes = json.loads(zattrs_bytes) if zattrs_bytes is not None else None

        return cls.from_json(store, path, metadata, attributes)

    @classmethod
    def open(cls, store: "Store", path: str) -> "GroupV2":
        return sync(cls.open_async(store, path))

    @classmethod
    def from_json(
        cls,
        store: Store,
        path: str,
        zarr_json: Any,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> "GroupV2":
        group = cls(
            metadata=make_cattr().structure(zarr_json, GroupV2Metadata),
            store=store,
            path=path,
            attributes=attributes,
        )
        return group

    @staticmethod
    async def open_or_array(store: Store, path: str) -> Union[ArrayV2, "GroupV2"]:
        zgroup_bytes, zattrs_bytes = await asyncio.gather(
            store.get_async(f"{path}/{ZGROUP_JSON}"),
            store.get_async(f"{path}/{ZATTRS_JSON}"),
        )
        attributes = json.loads(zattrs_bytes) if zattrs_bytes is not None else None
        if zgroup_bytes is not None:
            return GroupV2.from_json(store, path, json.loads(zgroup_bytes), attributes)
        zarray_bytes = await store.get_async(f"{path}/{ZARRAY_JSON}")
        if zarray_bytes is not None:
            return ArrayV2.from_json(store, path, json.loads(zarray_bytes), attributes)
        raise KeyError

    async def _save_metadata(self) -> None:
        await self.store.set_async(
            f"{self.path}/{ZGROUP_JSON}",
            json.dumps(asdict(self.metadata)).encode(),
        )
        if self.attributes is not None and len(self.attributes) > 0:
            await self.store.set_async(
                f"{self.path}/{ZATTRS_JSON}",
                json.dumps(self.attributes).encode(),
            )
        else:
            await self.store.delete_async(
                f"{self.path}/{ZATTRS_JSON}",
            )

    def _dereference_path(self, path: str) -> str:
        assert isinstance(path, str)
        path = f"{self.path}/{path}"
        if len(path) > 1:
            assert path[-1] != "/"
        return path

    async def get_async(self, path: str) -> Union[ArrayV2, "GroupV2"]:
        path = self._dereference_path(path)
        return await self.__class__.open_or_array(self.store, path)

    def __getitem__(self, path: str) -> Union[ArrayV2, "GroupV2"]:
        return sync(self.get_async(path))

    async def create_group_async(self, path: str, **kwargs) -> "GroupV2":
        path = self._dereference_path(path)
        return await self.__class__.create_async(self.store, path, **kwargs)

    def create_group(self, path: str, **kwargs) -> "GroupV2":
        return sync(self.create_group_async(path))

    async def create_array_async(self, path: str, **kwargs) -> ArrayV2:
        path = self._dereference_path(path)
        return await ArrayV2.create_async(self.store, path, **kwargs)

    def create_array(self, path: str, **kwargs) -> "ArrayV2":
        return sync(self.create_array_async(path, **kwargs))

    def __repr__(self):
        path = self.path
        return f"<Group_v2 {path}>"
