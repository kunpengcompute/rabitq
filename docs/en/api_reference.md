# API Reference

This document details the API modifications in the equivalence and non-equivalence index optimizations compared to the baseline open-source RaBitQ codebase.

## Equivalence Index Optimization API Modifications

### C++ Command Line Parameters

**index program**

| Parameter| Original| After Optimization|
|------|------|--------|
| `-d` (dataset) | Yes| Yes|
| `-s` (source) | Yes| Yes|
| `-p` (data\_path) | — | Added. Path to the HDF5 data file.|
| `-m` (metric\_type) | — | Added. Metric type (squared_l2/dot_product).|
| `-a` (soar\_lambda) | — | Added. SOAR parameter (set to `0` to disable).|

The data loading method has been changed from directly reading `fvecs` files to loading the HDF5 format via `loadHDFBase()`.

**search program**

| Parameter| Original| After Optimization|
|------|------|--------|
| `-d` (dataset) | Yes| Yes|
| `-s` (source) | Yes| Yes|
| `-r` (result\_path) | Yes| Yes|
| `-k` (topK) | Yes| Yes|
| `-n` (nprobe) | — | Added. Number of clusters to probe.|
| `-p` (data\_path) | — | Added. Path to the HDF5 data file.|
| `-m` (metric\_type) | — | Added. Metric type.|

The data loading method has been changed from reading `fvecs`/`ivecs` files to loading the HDF5 format via `loadHDF()`. The search framework has been upgraded from single-threaded to multi-threaded execution (via pthread CPU affinity and OpenMP), enabling highly concurrent operations across multiple NUMA nodes.

### C++ IVFRN Class

The public interface signature of the IVFRN class remains unchanged. On the AArch64 platform, the following private members and methods have been introduced internally:

| Change| Description|
|------|------|
| `Factor` structure (AArch64)| Batch layout: `float sqr_x[4]`, etc., replacing the original element-wise `float sqr_x`|
| `Factor_f16` structure| Newly added FP16 vectorized factor structure (grouped by 16 elements).|
| `search_fast_scan()` | Added private method, main search loop optimized using AArch64 NEON.|
| `compute_factor()` | Added private method to calculate FP16 factors.|
| `pack_codes_from_file()` | Added private method for packing quantized codes on AArch64.|
| `fast_scan()` (AArch64 overloaded)| Parameter types changed to `float16_t` (`query` and `data`), with a newly added `low_dist_scale` parameter.|
| `fast_scan_mask()` | Added static method for fast scanning using NEON mask filtering.|

### Shell Script

The original `script`/`index.sh` and `script`/`search.sh` use hardcoded parameters and are compiled using G++. After the optimization, `run.sh` is added as the unified entry.

Original:

```bash
./script/index.sh     # Hardcoded sift, K=4096, and compiled using g++
./script/search.sh    # Hardcoded sift, compiled using g++, and single-process
```

After optimization:

* The sift/deep/glove/fashion/gist datasets are supported.
* The architecture (x86_64/aarch64) is automatically identified, based on which compilation parameters are selected.
* clang++ is used and linked with jemalloc under AArch64.
* Concurrent search is performed on multiple NUMA nodes.

```bash
./run.sh <dataset> [fastscan|scan] [generate|index|search|all]
```

## Non-Equivalence Index Optimization API Modifications

The non-equivalence optimization contains all API modifications of the equivalence optimization, with the following additional modifications:

### Python Script

| Script| Description|Usage|
|------|------|-------|
| `data/eval.py` | New script: Responsible for ML model training and exporting.| `python eval.py <hdf5_path> <dataset> <K> <metric_type> <data_path> <BB>` |
| `data/test.py` | New script: Test tool for vector normalization.|-  |

### C++ Command Line Parameters

In addition to the changes in the equivalence optimization, the following parameters are added to the `search` program:

| Parameter| Description|
|------|------|
| `-t` (threshold) | ML model prediction threshold; enables adaptive nprobe when it it greater than 0.|
| `-e` (pred\_nprobe) | Predicted lower nprobe value.|
| `-a` (soar\_lambda) | SOAR parameter, which enables spilling cluster search when it is greater than 0.|

### C++ IVFRN Class

| Change| Description|
|------|------|
| `IVFRN()` constructor (AArch64)| Extended to accept SOAR parameters: `IVFRN(..., dist_to_spilled_labels*, x0_spilled*, spilled_labels*, binary_spilled*)`; SOAR is automatically enabled when spilling data is passed in.|
| `search()` (AArch64) | Signature extended to `search(query, rd_query, k, nprobe, soar_lambda, threshold, pred_nprobe, pred_func, distK)`, supporting SOAR search and ML-based adaptive nprobe.|
| `bool use_soar` | New member variable added for controlling whether to enable SOAR dual-cluster search.|
| `fast_scan_soar()` | New static method added for SOAR fast scanning with deduplication; parameters include `unordered_set<uint32_t> &seen`.|
| `search_fast_scan<bool soar>()` | New private template method added for controlling whether to scan spilling clusters via template parameters.|
| `PredictFunc` | New type alias `void ()(Entry data, int pred_margin, double* result)` added for ML model inference callbacks.|
| Spilled data member| New spilled data members added, including `start_spilled`, `len_spilled`, `id_spilled`, `binary_code_spilled`, `fac_f16_spilled`, `data_f16_spilled`, etc.|

### Shell Script

In addition to the changes in the equivalence optimization, the `run.sh` script adds two new phases:

* `train`: compiles and runs the `search_model` program to generate the optimal nprobe data per query (`approximateGT.bin` and `expectedNprobe1.bin`).
* `eval`: runs `eval.py` to train the LightGBM model and export it as a shared library (`libadaptivemodel_less.so`).

```bash
./run.sh <dataset> [fastscan|scan] [generate|index|search|train|eval|all]
```
