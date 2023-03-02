import zarrita
import numpy as np
import webknossos as wk


def test_sharding():
    # ds = wk.Dataset.open_remote(
    #     "l4_sample",
    #     organization_id="scalable_minds",
    #     webknossos_url="https://webknossos.org",
    # )
    ds = wk.Dataset.open(
        "/Users/norman/scalableminds/webknossos/binaryData/sample_organization/l4_sample"
    )

    def copy(data, path):
        s = zarrita.FileSystemStore("file://./testdata")
        a = zarrita.Array.create(
            s,
            path,
            shape=data.shape,
            chunk_shape=(32, 32, 32),
            dtype=data.dtype,
            fill_value=0,
            codecs=[
                zarrita.BloscCodecMetadata(
                    configuration=zarrita.BloscCodecConfigurationMetadata()
                ),
            ],
            storage_transformers=[
                zarrita.ShardingStorageTransformerMetadata(
                    configuration=zarrita.ShardingStorageTransformerConfigurationMetadata(
                        chunks_per_shard=(16, 16, 16)
                    )
                )
            ],
        )

        a[:, :, :] = data
        assert np.array_equal(data, a[:, :, :])

    copy(ds.get_layer("color").get_mag(1).read()[0], "l4_sample/color/1")
    copy(ds.get_layer("segmentation").get_mag(1).read()[0], "l4_sample/segmentation/1")


def test_order():
    data = np.arange(0, 256, dtype="uint8").reshape((16, 16), order="F")

    s = zarrita.FileSystemStore("file://./testdata")
    a = zarrita.Array.create(
        s,
        "order",
        shape=data.shape,
        chunk_shape=(16, 16),
        dtype=data.dtype,
        fill_value=0,
        codecs=[
            zarrita.TransposeCodecMetadata(
                configuration=zarrita.TransposeCodecConfigurationMetadata(order="F")
            ),
        ],
    )

    a[:, :] = data
    print(data, data.shape, data.flags)
    read_data = a[:, :]
    print(read_data, read_data.shape, read_data.flags)
    assert np.array_equal(data, read_data)
