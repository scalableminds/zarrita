# Benchmark

This is a very basic benchmark to show the performance characteristics of zarrita in comparison with [wkw](https://github.com/scalableminds/webknossos-wrap) and [zarr-python](https://github.com/zarr-developers/zarr-python).

Run on June 22nd, 2023 on a MacBook with M1 Pro and 32GB of RAM. Python has been emulated with Rosetta2.

Run it for yourself with `pytest tests/test_perf.py -vv -s`.

## Experiment design

- Reading and writing a dataset of shape (1024, 1024, 1024)
- Data sampled from an EM dataset ("color", uint8) by Motta et al., Science 2019 with an automated instance segmentation ("segmentation", uint32) by scalable minds
- Chunk sizes (32, 32, 32), (64, 64, 64) and (128, 128, 128)
- Shard size (1024, 1024, 1024) where applicable
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

Chunk size (32, 32, 32)

```
zarrita WRITE color - 8.62s
zarrita READ color - 6.93s
zarrita STORAGE 845.80 MB - 5 inodes

zarrita WRITE segmentation - 8.71s
zarrita READ segmentation - 12.06s
zarrita STORAGE 49.83 MB - 5 inodes

wkw WRITE color - 1.01s
wkw READ color - 0.64s
wkw STORAGE 1,072.11 MB - 4 inodes

wkw WRITE segmentation - 1.90s
wkw READ segmentation - 2.98s
wkw STORAGE 227.50 MB - 4 inodes

zarr WRITE color - 11.46s
zarr READ color - 4.32s
zarr STORAGE 845.28 MB - 32769 inodes

zarr WRITE segmentation - 16.58s
zarr READ segmentation - 5.98s
zarr STORAGE 49.30 MB - 32769 inodes
```

Chunk size (64, 64, 64)

```
zarrita WRITE color - 5.12s
zarrita READ color - 2.03s
zarrita STORAGE 850.06 MB - 5 inodes

zarrita WRITE segmentation - 6.21s
zarrita READ segmentation - 7.54s
zarrita STORAGE 50.76 MB - 5 inodes

wkw WRITE color - 0.81s
wkw READ color - 0.68s
wkw STORAGE 1,073.19 MB - 4 inodes

wkw WRITE segmentation - 1.62s
wkw READ segmentation - 2.90s
wkw STORAGE 221.61 MB - 4 inodes

zarr WRITE color - 13.20s
zarr READ color - 2.78s
zarr STORAGE 849.99 MB - 4097 inodes

zarr WRITE segmentation - 10.83s
zarr READ segmentation - 3.90s
zarr STORAGE 50.69 MB - 4097 inodes
```

Chunk size (128, 128, 128)

```
zarrita WRITE color - 3.84s
zarrita READ color - 1.87s
zarrita STORAGE 852.53 MB - 5 inodes

zarrita WRITE segmentation - 6.38s
zarrita READ segmentation - 6.57s
zarrita STORAGE 53.08 MB - 5 inodes

wkw WRITE color - 0.70s
wkw READ color - 0.62s
wkw STORAGE 1,075.84 MB - 4 inodes

wkw WRITE segmentation - 1.57s
wkw READ segmentation - 2.68s
wkw STORAGE 204.30 MB - 4 inodes

zarr WRITE color - 12.37s
zarr READ color - 2.34s
zarr STORAGE 852.52 MB - 513 inodes

zarr WRITE segmentation - 10.83s
zarr READ segmentation - 3.62s
zarr STORAGE 53.07 MB - 513 inodes
```
