import itertools
import math
import numbers
from typing import Any, NamedTuple


def _ensure_tuple(v):
    if not isinstance(v, tuple):
        v = (v,)
    return v


def _err_too_many_indices(selection, shape):
    raise IndexError(
        "too many indices for array; expected {}, got {}".format(
            len(shape), len(selection)
        )
    )


def _err_boundscheck(dim_len):
    raise IndexError("index out of bounds for dimension with length {}".format(dim_len))


def _err_negative_step():
    raise IndexError("only slices with step >= 1 are supported")


def _check_selection_length(selection, shape):
    if len(selection) > len(shape):
        _err_too_many_indices(selection, shape)


def _replace_ellipsis(selection, shape):
    selection = _ensure_tuple(selection)

    # count number of ellipsis present
    n_ellipsis = sum(1 for i in selection if i is Ellipsis)

    if n_ellipsis > 1:
        # more than 1 is an error
        raise IndexError("an index can only have a single ellipsis ('...')")

    elif n_ellipsis == 1:
        # locate the ellipsis, count how many items to left and right
        n_items_l = selection.index(Ellipsis)  # items to left of ellipsis
        n_items_r = len(selection) - (n_items_l + 1)  # items to right of ellipsis
        n_items = len(selection) - 1  # all non-ellipsis items

        if n_items >= len(shape):
            # ellipsis does nothing, just remove it
            selection = tuple(i for i in selection if i != Ellipsis)

        else:
            # replace ellipsis with as many slices are needed for number of dims
            new_item = selection[:n_items_l] + ((slice(None),) * (len(shape) - n_items))
            if n_items_r:
                new_item += selection[-n_items_r:]
            selection = new_item

    # fill out selection if not completely specified
    if len(selection) < len(shape):
        selection += (slice(None),) * (len(shape) - len(selection))

    # check selection not too long
    _check_selection_length(selection, shape)

    return selection


class _ChunkDimProjection(NamedTuple):
    dim_chunk_ix: Any
    dim_chunk_sel: Any
    dim_out_sel: Any


def _normalize_integer_selection(dim_sel, dim_len):
    # normalize type to int
    dim_sel = int(dim_sel)

    # handle wraparound
    if dim_sel < 0:
        dim_sel = dim_len + dim_sel

    # handle out of bounds
    if dim_sel >= dim_len or dim_sel < 0:
        _err_boundscheck(dim_len)

    return dim_sel


class _IntDimIndexer(object):
    def __init__(self, dim_sel, dim_len, dim_chunk_len):
        # normalize
        dim_sel = _normalize_integer_selection(dim_sel, dim_len)

        # store attributes
        self.dim_sel = dim_sel
        self.dim_len = dim_len
        self.dim_chunk_len = dim_chunk_len
        self.nitems = 1

    def __iter__(self):
        dim_chunk_ix = self.dim_sel // self.dim_chunk_len
        dim_offset = dim_chunk_ix * self.dim_chunk_len
        dim_chunk_sel = self.dim_sel - dim_offset
        dim_out_sel = None
        yield _ChunkDimProjection(dim_chunk_ix, dim_chunk_sel, dim_out_sel)


def _ceildiv(a, b):
    return math.ceil(a / b)


class _SliceDimIndexer(object):
    def __init__(self, dim_sel, dim_len, dim_chunk_len):
        # normalize
        self.start, self.stop, self.step = dim_sel.indices(dim_len)
        if self.step < 1:
            _err_negative_step()

        # store attributes
        self.dim_len = dim_len
        self.dim_chunk_len = dim_chunk_len
        self.nitems = max(0, _ceildiv((self.stop - self.start), self.step))
        self.nchunks = _ceildiv(self.dim_len, self.dim_chunk_len)

    def __iter__(self):
        # figure out the range of chunks we need to visit
        dim_chunk_ix_from = self.start // self.dim_chunk_len
        dim_chunk_ix_to = _ceildiv(self.stop, self.dim_chunk_len)

        # iterate over chunks in range
        for dim_chunk_ix in range(dim_chunk_ix_from, dim_chunk_ix_to):
            # compute offsets for chunk within overall array
            dim_offset = dim_chunk_ix * self.dim_chunk_len
            dim_limit = min(self.dim_len, (dim_chunk_ix + 1) * self.dim_chunk_len)

            # determine chunk length, accounting for trailing chunk
            dim_chunk_len = dim_limit - dim_offset

            if self.start < dim_offset:
                # selection starts before current chunk
                dim_chunk_sel_start = 0
                remainder = (dim_offset - self.start) % self.step
                if remainder:
                    dim_chunk_sel_start += self.step - remainder
                # compute number of previous items, provides offset into output array
                dim_out_offset = _ceildiv((dim_offset - self.start), self.step)

            else:
                # selection starts within current chunk
                dim_chunk_sel_start = self.start - dim_offset
                dim_out_offset = 0

            if self.stop > dim_limit:
                # selection ends after current chunk
                dim_chunk_sel_stop = dim_chunk_len

            else:
                # selection ends within current chunk
                dim_chunk_sel_stop = self.stop - dim_offset

            dim_chunk_sel = slice(dim_chunk_sel_start, dim_chunk_sel_stop, self.step)
            dim_chunk_nitems = _ceildiv(
                (dim_chunk_sel_stop - dim_chunk_sel_start), self.step
            )
            dim_out_sel = slice(dim_out_offset, dim_out_offset + dim_chunk_nitems)

            yield _ChunkDimProjection(dim_chunk_ix, dim_chunk_sel, dim_out_sel)


class _ChunkProjection(NamedTuple):
    chunk_coords: Any
    chunk_selection: Any
    out_selection: Any


class _BasicIndexer:
    def __init__(self, selection, shape, chunk_shape):
        # handle ellipsis
        selection = _replace_ellipsis(selection, shape)

        # setup per-dimension indexers
        dim_indexers = []
        for dim_sel, dim_len, dim_chunk_len in zip(selection, shape, chunk_shape):
            if isinstance(dim_sel, numbers.Integral):
                dim_indexer = _IntDimIndexer(dim_sel, dim_len, dim_chunk_len)

            elif isinstance(dim_sel, slice):
                dim_indexer = _SliceDimIndexer(dim_sel, dim_len, dim_chunk_len)

            else:
                raise IndexError(
                    "unsupported selection item for basic indexing; "
                    "expected integer or slice, got {!r}".format(type(dim_sel))
                )

            dim_indexers.append(dim_indexer)

        self.dim_indexers = dim_indexers
        self.shape = tuple(
            s.nitems for s in self.dim_indexers if not isinstance(s, _IntDimIndexer)
        )

    def __iter__(self):
        for dim_projections in itertools.product(*self.dim_indexers):
            chunk_coords = tuple(p.dim_chunk_ix for p in dim_projections)
            chunk_selection = tuple(p.dim_chunk_sel for p in dim_projections)
            out_selection = tuple(
                p.dim_out_sel for p in dim_projections if p.dim_out_sel is not None
            )

            yield _ChunkProjection(chunk_coords, chunk_selection, out_selection)
