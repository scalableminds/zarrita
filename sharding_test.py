import json
import os
import shutil

import zarrita


shutil.rmtree("sharding_test.zr3", ignore_errors=True)
h = zarrita.create_hierarchy("sharding_test.zr3")
a = h.create_array(
    path="testarray",
    shape=(20, 3),
    dtype="float64",
    chunk_shape=(3, 2),
    shards=(2, 2),
)

a[:10, :] = 42
a[15, 1] = 389
a[19, 2] = 1
a[0, 1] = -4.2

assert a.store._shards == (2, 2)
assert a[15, 1] == 389
assert a[19, 2] == 1
assert a[0, 1] == -4.2
assert a[0, 0] == 42

array_json = a.store["meta/root/testarray.array.json"].decode()

print(array_json)
# {
#     "shape": [
#         20,
#         3
#     ],
#     "data_type": "<f8",
#     "chunk_grid": {
#         "type": "regular",
#         "chunk_shape": [
#             3,
#             2
#         ],
#         "separator": "/"
#     },
#     "chunk_memory_layout": "C",
#     "fill_value": null,
#     "extensions": [],
#     "attributes": {},
#     "shards": [
#         2,
#         2
#     ],
#     "shard_format": "indexed"
# }

assert json.loads(array_json)["shards"] == [2, 2]

print("ONDISK")
for root, dirs, files in os.walk("sharding_test.zr3"):
    dirs.sort()
    if len(files) > 0:
        print("   ", root.ljust(40), *sorted(files))
print("UNDERLYING STORE", sorted(i.rsplit("c")[-1] for i in a.store._store if i.startswith("data")))
print("STORE", sorted(i.rsplit("c")[-1] for i in a.store if i.startswith("data")))
# ONDISK
#     sharding_test.zr3                        zarr.json
#     sharding_test.zr3/data/root/testarray/c0 0
#     sharding_test.zr3/data/root/testarray/c1 0
#     sharding_test.zr3/data/root/testarray/c2 0
#     sharding_test.zr3/data/root/testarray/c3 0
#     sharding_test.zr3/meta/root              testarray.array.json
# UNDERLYING STORE ['0/0', '1/0', '2/0', '3/0']
# STORE ['0/0', '0/1', '1/0', '1/1', '2/0', '2/1', '3/0', '3/1', '5/0', '6/1']

index_bytes = a.store._store["data/root/testarray/c0/0"][-2*2*16:]
print("INDEX 0.0", [int.from_bytes(index_bytes[i:i+8], byteorder="little") for i in range(0, len(index_bytes), 8)])
# INDEX 0.0 [0, 48, 48, 48, 96, 48, 144, 48]


a_reopened = zarrita.get_hierarchy("sharding_test.zr3").get_array("testarray")
assert a_reopened.store._shards == (2, 2)
assert a_reopened[15, 1] == 389
assert a_reopened[19, 2] == 1
assert a_reopened[0, 1] == -4.2
assert a_reopened[0, 0] == 42
