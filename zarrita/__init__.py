from __future__ import annotations

from typing import Optional, Union

import zarrita.codecs  # noqa: F401
from zarrita.array import (  # noqa: F401
    Array,
    ArrayRuntimeConfiguration,
    runtime_configuration,
)
from zarrita.array_v2 import ArrayV2  # noqa: F401
from zarrita.group import Group  # noqa: F401
from zarrita.group_v2 import GroupV2  # noqa: F401
from zarrita.store import (  # noqa: F401
    LocalStore,
    RemoteStore,
    Store,
    StoreLike,
    StorePath,
    make_store_path,
)
from zarrita.sync import sync as _sync


async def open_auto_async(
    store: StoreLike,
    runtime_configuration_: Optional[ArrayRuntimeConfiguration] = None,
) -> Union[Array, ArrayV2, Group, GroupV2]:
    store_path = make_store_path(store)
    try:
        return await Group.open_or_array(
            store_path, runtime_configuration=runtime_configuration_
        )
    except KeyError:
        return await GroupV2.open_or_array(store_path)


def open_auto(
    store: StoreLike,
    runtime_configuration_: Optional[ArrayRuntimeConfiguration] = None,
) -> Union[Array, ArrayV2, Group, GroupV2]:
    return _sync(open_auto_async(store, runtime_configuration_))
