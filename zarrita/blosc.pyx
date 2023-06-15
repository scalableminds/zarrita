# cython: embedsignature=True
# cython: profile=False
# cython: linetrace=False
# cython: binding=False
# cython: language_level=3
import multiprocessing
import os

from cpython.buffer cimport (
    PyBUF_ANY_CONTIGUOUS,
    PyBUF_WRITEABLE,
    PyBuffer_Release,
    PyObject_GetBuffer,
)
from cpython.bytes cimport PyBytes_AS_STRING, PyBytes_FromStringAndSize

from numcodecs.abc import Codec
from numcodecs.compat import ensure_contiguous_ndarray


cdef class Buffer:
    cdef:
        char *ptr
        Py_buffer buffer
        size_t nbytes
        size_t itemsize
        bint acquired

    def __cinit__(self, obj, flags):
        PyObject_GetBuffer(obj, &(self.buffer), flags)
        self.acquired = True
        self.ptr = <char *> self.buffer.buf
        self.itemsize = self.buffer.itemsize
        self.nbytes = self.buffer.len

    cpdef release(self):
        if self.acquired:
            PyBuffer_Release(&(self.buffer))
            self.acquired = False

    def __dealloc__(self):
        self.release()

cdef extern from "blosc.h":
    cdef enum:
        BLOSC_MAX_OVERHEAD,
        BLOSC_VERSION_STRING,
        BLOSC_VERSION_DATE,
        BLOSC_NOSHUFFLE,
        BLOSC_SHUFFLE,
        BLOSC_BITSHUFFLE,
        BLOSC_MAX_BUFFERSIZE,
        BLOSC_MAX_THREADS,
        BLOSC_MAX_TYPESIZE,
        BLOSC_DOSHUFFLE,
        BLOSC_DOBITSHUFFLE,
        BLOSC_MEMCPYED

    char* blosc_list_compressors()
    int blosc_compress_ctx(int clevel, int doshuffle, size_t typesize, size_t nbytes,
                           const void* src, void* dest, size_t destsize,
                           const char* compressor, size_t blocksize,
                           int numinternalthreads) nogil
    int blosc_decompress_ctx(const void* src, void* dest, size_t destsize,
                             int numinternalthreads) nogil
    void blosc_cbuffer_sizes(const void* cbuffer, size_t* nbytes, size_t* cbytes,
                             size_t* blocksize)


MAX_OVERHEAD = BLOSC_MAX_OVERHEAD
MAX_BUFFERSIZE = BLOSC_MAX_BUFFERSIZE
MAX_THREADS = BLOSC_MAX_THREADS
MAX_TYPESIZE = BLOSC_MAX_TYPESIZE
VERSION_STRING = <char *> BLOSC_VERSION_STRING
VERSION_DATE = <char *> BLOSC_VERSION_DATE
VERSION_STRING = VERSION_STRING.decode()
VERSION_DATE = VERSION_DATE.decode()
__version__ = VERSION_STRING
NOSHUFFLE = BLOSC_NOSHUFFLE
SHUFFLE = BLOSC_SHUFFLE
BITSHUFFLE = BLOSC_BITSHUFFLE
# automatic shuffle
AUTOSHUFFLE = -1
# automatic block size - let blosc decide
AUTOBLOCKS = 0

def list_compressors():
    """Get a list of compressors supported in the current build."""
    s = blosc_list_compressors()
    s = s.decode('ascii')
    return s.split(',')

def err_bad_cname(cname):
    raise ValueError('bad compressor or compressor not supported: %r; expected one of '
                     '%s' % (cname, list_compressors()))


def compress(source, char* cname, int clevel, int shuffle=SHUFFLE,
             int blocksize=AUTOBLOCKS):
    """Compress data.

    Parameters
    ----------
    source : bytes-like
        Data to be compressed. Can be any object supporting the buffer
        protocol.
    cname : bytes
        Name of compression library to use.
    clevel : int
        Compression level.
    shuffle : int
        Either NOSHUFFLE (0), SHUFFLE (1), BITSHUFFLE (2) or AUTOSHUFFLE (-1). If AUTOSHUFFLE,
        bit-shuffle will be used for buffers with itemsize 1, and byte-shuffle will
        be used otherwise. The default is `SHUFFLE`.
    blocksize : int
        The requested size of the compressed blocks.  If 0, an automatic blocksize will
        be used.

    Returns
    -------
    dest : bytes
        Compressed data.

    """

    cdef:
        char *source_ptr
        char *dest_ptr
        Buffer source_buffer
        size_t nbytes, itemsize
        int cbytes
        bytes dest

    # check valid cname early
    cname_str = cname.decode('ascii')
    if cname_str not in list_compressors():
        err_bad_cname(cname_str)

    # setup source buffer
    source_buffer = Buffer(source, PyBUF_ANY_CONTIGUOUS)
    source_ptr = source_buffer.ptr
    nbytes = source_buffer.nbytes
    itemsize = source_buffer.itemsize

    # determine shuffle
    if shuffle == AUTOSHUFFLE:
        if itemsize == 1:
            shuffle = BITSHUFFLE
        else:
            shuffle = SHUFFLE
    elif shuffle not in [NOSHUFFLE, SHUFFLE, BITSHUFFLE]:
        raise ValueError('invalid shuffle argument; expected -1, 0, 1 or 2, found %r' %
                         shuffle)

    try:

        # setup destination
        dest = PyBytes_FromStringAndSize(NULL, nbytes + BLOSC_MAX_OVERHEAD)
        dest_ptr = PyBytes_AS_STRING(dest)

        with nogil:
            cbytes = blosc_compress_ctx(clevel, shuffle, itemsize, nbytes, source_ptr,
                                        dest_ptr, nbytes + BLOSC_MAX_OVERHEAD,
                                        cname, blocksize, 1)

    finally:

        # release buffers
        source_buffer.release()

    # check compression was successful
    if cbytes <= 0:
        raise RuntimeError('error during blosc compression: %d' % cbytes)

    # resize after compression
    dest = dest[:cbytes]

    return dest


def decompress(source, dest=None):
    """Decompress data.

    Parameters
    ----------
    source : bytes-like
        Compressed data, including blosc header. Can be any object supporting the buffer
        protocol.
    dest : array-like, optional
        Object to decompress into.

    Returns
    -------
    dest : bytes
        Object containing decompressed data.

    """
    cdef:
        int ret
        char *source_ptr
        char *dest_ptr
        Buffer source_buffer
        Buffer dest_buffer = None
        size_t nbytes, cbytes, blocksize

    # setup source buffer
    source_buffer = Buffer(source, PyBUF_ANY_CONTIGUOUS)
    source_ptr = source_buffer.ptr

    # determine buffer size
    blosc_cbuffer_sizes(source_ptr, &nbytes, &cbytes, &blocksize)

    # setup destination buffer
    if dest is None:
        # allocate memory
        dest = PyBytes_FromStringAndSize(NULL, nbytes)
        dest_ptr = PyBytes_AS_STRING(dest)
        dest_nbytes = nbytes
    else:
        arr = ensure_contiguous_ndarray(dest)
        dest_buffer = Buffer(arr, PyBUF_ANY_CONTIGUOUS | PyBUF_WRITEABLE)
        dest_ptr = dest_buffer.ptr
        dest_nbytes = dest_buffer.nbytes

    try:

        # guard condition
        if dest_nbytes < nbytes:
            raise ValueError('destination buffer too small; expected at least %s, '
                             'got %s' % (nbytes, dest_nbytes))

        # perform decompression
        with nogil:
            ret = blosc_decompress_ctx(source_ptr, dest_ptr, nbytes, 1)

    finally:

        # release buffers
        source_buffer.release()
        if dest_buffer is not None:
            dest_buffer.release()

    # handle errors
    if ret <= 0:
        raise RuntimeError('error during blosc decompression: %d' % ret)

    return dest



_shuffle_repr = ['AUTOSHUFFLE', 'NOSHUFFLE', 'SHUFFLE', 'BITSHUFFLE']


class Blosc(Codec):
    """Codec providing compression using the Blosc meta-compressor.

    Parameters
    ----------
    cname : string, optional
        A string naming one of the compression algorithms available within blosc, e.g.,
        'zstd', 'blosclz', 'lz4', 'lz4hc', 'zlib' or 'snappy'.
    clevel : integer, optional
        An integer between 0 and 9 specifying the compression level.
    shuffle : integer, optional
        Either NOSHUFFLE (0), SHUFFLE (1), BITSHUFFLE (2) or AUTOSHUFFLE (-1). If AUTOSHUFFLE,
        bit-shuffle will be used for buffers with itemsize 1, and byte-shuffle will
        be used otherwise. The default is `SHUFFLE`.
    blocksize : int
        The requested size of the compressed blocks.  If 0 (default), an automatic
        blocksize will be used.

    See Also
    --------
    numcodecs.zstd.Zstd, numcodecs.lz4.LZ4

    """

    codec_id = 'blosc'
    NOSHUFFLE = NOSHUFFLE
    SHUFFLE = SHUFFLE
    BITSHUFFLE = BITSHUFFLE
    AUTOSHUFFLE = AUTOSHUFFLE
    max_buffer_size = 2**31 - 1

    def __init__(self, cname='lz4', clevel=5, shuffle=SHUFFLE, blocksize=AUTOBLOCKS):
        self.cname = cname
        if isinstance(cname, str):
            self._cname_bytes = cname.encode('ascii')
        else:
            self._cname_bytes = cname
        self.clevel = clevel
        self.shuffle = shuffle
        self.blocksize = blocksize

    def encode(self, buf):
        buf = ensure_contiguous_ndarray(buf, self.max_buffer_size)
        return compress(buf, self._cname_bytes, self.clevel, self.shuffle, self.blocksize)

    def decode(self, buf, out=None):
        buf = ensure_contiguous_ndarray(buf, self.max_buffer_size)
        return decompress(buf, out)

    def __repr__(self):
        r = '%s(cname=%r, clevel=%r, shuffle=%s, blocksize=%s)' % \
            (type(self).__name__,
             self.cname,
             self.clevel,
             _shuffle_repr[self.shuffle + 1],
             self.blocksize)
        return r