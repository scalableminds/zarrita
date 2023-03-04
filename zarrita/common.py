from typing import TYPE_CHECKING, List, Literal

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
