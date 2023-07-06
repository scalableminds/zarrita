# Benchmark

This is a very basic benchmark to show the performance characteristics of zarrita in comparison with [wkw](https://github.com/scalableminds/webknossos-wrap) and [zarr-python](https://github.com/zarr-developers/zarr-python).

Run on June 22nd, 2023 on a MacBook with M1 Pro and 32GB of RAM. Python has been emulated with Rosetta2.

Run it for yourself with `pytest tests/test_perf.py -vv -s`.

## Experiment design

- Reading and writing a dataset of shape (1024, 1024, 1024)
- Additionally, reading a cutout of shape (32, 32, 32)
- Data sampled from an EM dataset ("color", uint8) by Motta et al., Science 2019 with an automated instance segmentation ("segmentation", uint32) by scalable minds
- Chunk size (512, 512, 512)
- Inner chunk sizes (32, 32, 32), (64, 64, 64) and (128, 128, 128) where applicable
- Writing aligned shards

### Variants

- Zarr v3 datasets with zarrita
  - Sharding with nested blosc and transpose (F order) codecs
  - Default chunk key encoding with `/` separator
- [WKW](https://github.com/scalableminds/webknossos-wrap) datasets, implemented in Rust with Python bindings
  - LZ4 compression
- Zarr v3 datasets with [zarr-python](https://github.com/zarr-developers/zarr-python)
  - Blosc codec
  - F order

### Metrics

- Write and read time
- File size and inodes

## Results

Inner chunk size (32, 32, 32)

```
  zarrita WRITE color - 8.91s
  zarrita STORAGE 845.80 MB - 16 inodes
  zarrita READ color - 7.56s
  zarrita PARTIAL READ color - 0.00294s

  zarrita WRITE segmentation - 11.72s
  zarrita STORAGE 49.83 MB - 16 inodes
  zarrita READ segmentation - 14.02s
  zarrita PARTIAL READ segmentation - 0.00527s

  wkw WRITE color - 1.06s
  wkw STORAGE 1,072.11 MB - 15 inodes
  wkw READ color - 0.63s
  wkw PARTIAL READ color - 0.00024s

  wkw WRITE segmentation - 2.16s
  wkw STORAGE 227.50 MB - 15 inodes
  wkw READ segmentation - 3.61s
  wkw PARTIAL READ segmentation - 0.00108s

  zarr WRITE color - 13.18s
  zarr STORAGE 864.23 MB - 9 inodes
  zarr READ color - 2.02s
  zarr PARTIAL READ color - 0.18277s

  zarr WRITE segmentation - 10.87s
  zarr STORAGE 56.17 MB - 9 inodes
  zarr READ segmentation - 6.86s
  zarr PARTIAL READ segmentation - 0.26412s
```

Inner chunk size (64, 64, 64)

```
  zarrita WRITE color - 5.17s
  zarrita STORAGE 850.06 MB - 16 inodes
  zarrita READ color - 2.18s
  zarrita PARTIAL READ color - 0.00167s

  zarrita WRITE segmentation - 6.66s
  zarrita STORAGE 50.76 MB - 16 inodes
  zarrita READ segmentation - 8.12s
  zarrita PARTIAL READ segmentation - 0.00538s

  wkw WRITE color - 0.90s
  wkw STORAGE 1,073.19 MB - 15 inodes
  wkw READ color - 0.71s
  wkw PARTIAL READ color - 0.00030s

  wkw WRITE segmentation - 1.91s
  wkw STORAGE 221.61 MB - 15 inodes
  wkw READ segmentation - 3.49s
  wkw PARTIAL READ segmentation - 0.00183s

  zarr WRITE color - 13.36s
  zarr STORAGE 864.23 MB - 9 inodes
  zarr READ color - 1.97s
  zarr PARTIAL READ color - 0.17768s

  zarr WRITE segmentation - 10.39s
  zarr STORAGE 56.17 MB - 9 inodes
  zarr READ segmentation - 6.52s
  zarr PARTIAL READ segmentation - 0.26512s
```

Inner chunk size (128, 128, 128)

```
  zarrita WRITE color - 3.88s
  zarrita STORAGE 852.53 MB - 16 inodes
  zarrita READ color - 2.06s
  zarrita PARTIAL READ color - 0.00494s

  zarrita WRITE segmentation - 9.35s
  zarrita STORAGE 53.08 MB - 16 inodes
  zarrita READ segmentation - 7.14s
  zarrita PARTIAL READ segmentation - 0.00776s

  wkw WRITE color - 0.78s
  wkw STORAGE 1,075.84 MB - 15 inodes
  wkw READ color - 0.69s
  wkw PARTIAL READ color - 0.00135s

  wkw WRITE segmentation - 1.83s
  wkw STORAGE 204.30 MB - 15 inodes
  wkw READ segmentation - 3.02s
  wkw PARTIAL READ segmentation - 0.00627s

  zarr WRITE color - 14.07s
  zarr STORAGE 864.23 MB - 9 inodes
  zarr READ color - 1.84s
  zarr PARTIAL READ color - 0.17864s

  zarr WRITE segmentation - 10.67s
  zarr STORAGE 56.17 MB - 9 inodes
  zarr READ segmentation - 6.64s
  zarr PARTIAL READ segmentation - 0.26791s
```

Inner chunk size (32, 32, 32), chunk size (32, 32, 32) for zarr

```
tests/test_perf.py::test_zarrita_sharding[color-testdata0]
  zarrita WRITE color - 8.79s
  zarrita STORAGE 845.80 MB - 16 inodes
  zarrita READ color - 7.34s
  zarrita PARTIAL READ color - 0.00224s

  zarrita WRITE segmentation - 10.51s
  zarrita STORAGE 49.83 MB - 16 inodes
  zarrita READ segmentation - 14.99s
  zarrita PARTIAL READ segmentation - 0.00450s

  wkw WRITE color - 1.10s
  wkw STORAGE 1,072.11 MB - 15 inodes
  wkw READ color - 0.67s
  wkw PARTIAL READ color - 0.00026s

  wkw WRITE segmentation - 2.24s
  wkw STORAGE 227.50 MB - 15 inodes
  wkw READ segmentation - 3.56s
  wkw PARTIAL READ segmentation - 0.00107s

  zarr WRITE color - 12.89s
  zarr STORAGE 845.28 MB - 32769 inodes
  zarr READ color - 4.60s
  zarr PARTIAL READ color - 0.00039s

  zarr WRITE segmentation - 18.87s
  zarr STORAGE 49.30 MB - 32769 inodes
  zarr READ segmentation - 9.12s
  zarr PARTIAL READ segmentation - 0.00251s
```

## Discussion

wkw with its Rust-based implementations performs best in all scenarios. zarrita and zarr perform similarly in reading and writing full datasets. zarrita even outperforms zarr slightly, which is probably due to the use of multithreading for (de)compression.

zarrita with sharding dramatically outperforms unsharded zarr for partial reads (with large a chunk size). This is unsurprising because sharding has been designed to speed up partial reads. With sharding, smaller pieces of the data (i.e. inner chunks) need to be decompressed instead of the full chunk. This means less IO and less processing.

When using unsharded zarr with small chunk sizes, the read speeds, become similar to sharded zarrita. In fact, zarr outperforms zarrita on this benchmark machine. However, it comes at the cost of a large number of consumed inodes which is prohibitive for large arrays on many file systems.
