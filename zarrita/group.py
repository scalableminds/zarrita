import json
from typing import Any, Dict, Literal, Optional, Union

from attr import asdict, field, frozen
from cattrs import structure

from zarrita.array import Array
from zarrita.common import ZARR_JSON
from zarrita.store import Store


@frozen
class GroupMetadata:
    attributes: Dict[str, Any] = field(factory=dict)
    zarr_format: Literal[3] = 3
    node_type: Literal["group"] = "group"


class Group:
    metadata: GroupMetadata
    store: Store
    path: str

    @classmethod
    def create(
        cls, store: "Store", path: str, *, attributes: Optional[Dict[str, Any]] = None
    ) -> "Group":
        group = cls()
        group.metadata = GroupMetadata(attributes=attributes or {})
        group.store = store
        group.path = path
        group._save_metadata()
        return group

    @classmethod
    def open(cls, store: "Store", path: str) -> "Group":
        zarr_json_bytes = store.get(f"{path}/{ZARR_JSON}")
        assert zarr_json_bytes is not None
        return cls.from_json(store, path, json.loads(zarr_json_bytes))

    @classmethod
    def from_json(cls, store: Store, path: str, zarr_json: Any) -> "Group":
        group = cls()
        group.metadata = structure(zarr_json, GroupMetadata)
        group.store = store
        group.path = path
        return group

    @staticmethod
    def open_or_array(store: Store, path: str) -> Union[Array, "Group"]:
        zarr_json_bytes = store.get(f"{path}/{ZARR_JSON}")
        assert zarr_json_bytes is not None
        zarr_json = json.loads(zarr_json_bytes)
        if zarr_json["node_type"] == "group":
            return Group.from_json(store, path, zarr_json)
        if zarr_json["node_type"] == "array":
            return Array.from_json(store, path, zarr_json)
        raise KeyError

    def _save_metadata(self) -> None:
        self.store.set(
            f"{self.path}/{ZARR_JSON}",
            json.dumps(asdict(self.metadata)).encode(),
        )

    def _dereference_path(self, path: str) -> str:
        assert isinstance(path, str)
        path = f"{self.path}/{path}"
        if len(path) > 1:
            assert path[-1] != "/"
        return path

    def __getitem__(self, path: str) -> Union[Array, "Group"]:
        path = self._dereference_path(path)
        return Group.open_or_array(self.store, path)

    def create_group(self, path: str, **kwargs) -> "Group":
        path = self._dereference_path(path)
        return Group.create(self.store, path, **kwargs)

    def create_array(self, path: str, **kwargs) -> Array:
        path = self._dereference_path(path)
        return Array.create(self.store, path, **kwargs)

    def __repr__(self):
        path = self.path
        return f"<Group {path}>"
