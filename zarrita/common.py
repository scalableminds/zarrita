from typing import TYPE_CHECKING, Any, Dict, List, Literal, Tuple, Union

from cattr import Converter

if TYPE_CHECKING:
    from zarrita.codecs import CodecMetadata


ZARR_JSON = "zarr.json"


def is_total_slice(item, shape):
    """Determine whether `item` specifies a complete slice of array with the
    given `shape`. Used to optimize __setitem__ operations on the Chunk
    class."""

    # N.B., assume shape is normalized

    if item == Ellipsis:
        return True
    if item == slice(None):
        return True
    if isinstance(item, slice):
        item = (item,)
    if isinstance(item, tuple):
        return all(
            (
                isinstance(s, slice)
                and (
                    (s == slice(None))
                    or ((s.stop - s.start == l) and (s.step in [1, None]))
                )
            )
            for s, l in zip(item, shape)
        )
    else:
        raise TypeError("expected slice or tuple of slices, found %r" % item)


def get_order(codecs: List["CodecMetadata"]) -> Literal["C", "F"]:
    for codec in codecs:
        if codec.name == "transpose":
            order = codec.configuration.order
            if not isinstance(order, tuple):
                return order
    return "C"


def make_cattr():
    from zarrita.array import (
        ChunkKeyEncodingMetadata,
        DefaultChunkKeyEncodingMetadata,
        V2ChunkKeyEncodingMetadata,
    )
    from zarrita.codecs import (
        BloscCodecMetadata,
        CodecMetadata,
        EndianCodecMetadata,
        GzipCodecMetadata,
        ShardingCodecMetadata,
        TransposeCodecMetadata,
    )

    dataset_converter = Converter()

    def _structure_chunk_key_encoding_metadata(
        d: Dict[str, Any], _t
    ) -> ChunkKeyEncodingMetadata:
        if d["name"] == "default":
            return dataset_converter.structure(d, DefaultChunkKeyEncodingMetadata)
        if d["name"] == "v2":
            return dataset_converter.structure(d, V2ChunkKeyEncodingMetadata)
        raise KeyError

    dataset_converter.register_structure_hook(
        ChunkKeyEncodingMetadata, _structure_chunk_key_encoding_metadata
    )

    def _structure_codec_metadata(d: Dict[str, Any], _t=None) -> CodecMetadata:
        if d["name"] == "blosc":
            return dataset_converter.structure(d, BloscCodecMetadata)
        if d["name"] == "endian":
            return dataset_converter.structure(d, EndianCodecMetadata)
        if d["name"] == "transpose":
            return dataset_converter.structure(d, TransposeCodecMetadata)
        if d["name"] == "gzip":
            return dataset_converter.structure(d, GzipCodecMetadata)
        if d["name"] == "sharding_indexed":
            return dataset_converter.structure(d, ShardingCodecMetadata)
        raise KeyError

    dataset_converter.register_structure_hook(CodecMetadata, _structure_codec_metadata)

    dataset_converter.register_structure_hook_factory(
        lambda t: str(t) == "ForwardRef('CodecMetadata')",
        lambda t: _structure_codec_metadata,
    )

    def _structure_order(d: Any, _t=None) -> Union[Literal["C", "F"], Tuple[int, ...]]:
        if d == "C":
            return "C"
        if d == "F":
            return "F"
        if isinstance(d, list):
            return tuple(d)
        raise KeyError

    dataset_converter.register_structure_hook_factory(
        lambda t: str(t)
        == "typing.Union[typing.Literal['C', 'F'], typing.Tuple[int, ...]]",
        lambda t: _structure_order,
    )

    return dataset_converter
