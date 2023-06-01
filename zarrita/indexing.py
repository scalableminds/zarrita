import itertools
import math
from typing import Any, Iterator, List, NamedTuple, Optional, Tuple, Union

from zarrita.common import ChunkCoords, Selection, SliceSelection


def _ensure_tuple(v: Selection) -> SliceSelection:
    if not isinstance(v, tuple):
        v = (v,)
    return v


def _err_too_many_indices(selection: SliceSelection, shape: ChunkCoords):
    raise IndexError(
        "too many indices for array; expected {}, got {}".format(
            len(shape), len(selection)
        )
    )


def _err_negative_step():
    raise IndexError("only slices with step >= 1 are supported")


def _check_selection_length(selection: SliceSelection, shape: ChunkCoords):
    if len(selection) > len(shape):
        _err_too_many_indices(selection, shape)


def _ensure_selection(
    selection: Selection,
    shape: ChunkCoords,
) -> SliceSelection:
    selection = _ensure_tuple(selection)

    # fill out selection if not completely specified
    if len(selection) < len(shape):
        selection += (slice(None),) * (len(shape) - len(selection))

    # check selection not too long
    _check_selection_length(selection, shape)

    return selection


class _ChunkDimProjection(NamedTuple):
    dim_chunk_ix: int
    dim_chunk_sel: slice
    dim_out_sel: Optional[slice]


def _ceildiv(a, b):
    return math.ceil(a / b)


class _SliceDimIndexer:
    dim_sel: slice
    dim_len: int
    dim_chunk_len: int
    nitems: int

    def __init__(self, dim_sel: slice, dim_len: int, dim_chunk_len: int):
        self.start, self.stop, self.step = dim_sel.indices(dim_len)
        if self.step < 1:
            _err_negative_step()

        self.dim_len = dim_len
        self.dim_chunk_len = dim_chunk_len
        self.nitems = max(0, _ceildiv((self.stop - self.start), self.step))
        self.nchunks = _ceildiv(self.dim_len, self.dim_chunk_len)

    def __iter__(self) -> Iterator[_ChunkDimProjection]:
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
    chunk_coords: ChunkCoords
    chunk_selection: SliceSelection
    out_selection: SliceSelection


class BasicIndexer:
    dim_indexers: List[_SliceDimIndexer]
    shape: ChunkCoords

    def __init__(
        self,
        selection: Selection,
        shape: Tuple[int, ...],
        chunk_shape: Tuple[int, ...],
    ):
        # setup per-dimension indexers
        self.dim_indexers = [
            _SliceDimIndexer(dim_sel, dim_len, dim_chunk_len)
            for dim_sel, dim_len, dim_chunk_len in zip(
                _ensure_selection(selection, shape), shape, chunk_shape
            )
        ]
        self.shape = tuple(s.nitems for s in self.dim_indexers)

    def __iter__(self) -> Iterator[_ChunkProjection]:
        for dim_projections in itertools.product(*self.dim_indexers):
            chunk_coords = tuple(p.dim_chunk_ix for p in dim_projections)
            chunk_selection = tuple(p.dim_chunk_sel for p in dim_projections)
            out_selection = tuple(
                p.dim_out_sel for p in dim_projections if p.dim_out_sel is not None
            )

            yield _ChunkProjection(chunk_coords, chunk_selection, out_selection)
