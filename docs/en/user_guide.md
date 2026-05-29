# User Guide

## Overview

`run.sh` is an optimized, unified test script that integrates the entire workflow of data generation, index construction, model training, and search benchmarking. The script automatically detects the system architecture (x86_64/AArch64) to select and apply the corresponding compilation parameters and execution strategies.

A `run.sh` script is provided for both equivalence and non-equivalence optimizations. While they share the same basic usage, the non-equivalence optimization provides additional support for the `train` and `eval` phases.

## Basic Usage

```bash
./run.sh <dataset> [mode] [stage]
```

| Parameter| Description| Allowed Values| Default Value|
|------|------|--------|--------|
| `dataset` | (Mandatory) Dataset name| `sift`, `deep`, `glove`, `fashion`, or `gist`| — |
| `mode` | Scan mode| `fastscan` or `scan`| `fastscan` |
| `stage` | Execution phase| See [Execution phase](#execution-phase).| `all` |

### Execution phase

**Phases supported by the equivalence optimization**

| Phase| Description|
|------|------|
| `generate` | Runs `ivf.py` and `rabitq.py` to perform IVF clustering and RaBitQ quantization, generating the intermediate data required for index construction.|
| `index` | Compiles and runs the C++ `index` program to generate a binary index file from the intermediate data.|
| `search` | Compiles and runs the C++ `search` program to perform search benchmarking and output the Query Per Second (QPS) and the recall rate.|
| `all` | Executes the phases in sequence: <code>generate</code> → <code>index</code> → <code>search</code>|

**Additional phases supported by the non-equivalence optimization**

| Phase| Description|
|------|------|
| `train` | Compiles and runs the `search_model` program to search for the optimal nprobe for each base vector, generating `approximateGT.bin` and `expectedNprobe1.bin` (AArch64 only).|
| `eval` | Runs `eval.py` to train the LightGBM classification model and export it as the `libadaptivemodel_less.so` shared library via Treelite (AArch64 only).|
| `all` | Executes the phases in sequence: `generate` → `index` → `train` → `eval` → `search`.|

## Dataset Configuration

The script pre-configures the dimension, quantization bit width, HDF5 filename, and metric type for each dataset.

| Dataset| HDF5 File| Metric Type| D (Dimension)| B (Quantization Bit Width)|
|--------|----------|---------|-----------|--------------|
| `sift` | `sift-128-euclidean.hdf5` | squared\_l2 | 128 | 128 |
| `deep` | `deep-image-96-angular.hdf5` | dot\_product | 96 | 128 |
| `glove` | `glove-100-angular.hdf5` | dot\_product | 100 | 128 |
| `fashion` | `fashion-mnist-784-euclidean.hdf5` | squared\_l2 | 784 | 832 |
| `gist` | `gist-960-euclidean.hdf5` | squared\_l2 | 960 | 960 |

The HDF5 data file must be stored in the `./datasets/` directory.

## Search Parameters

The script automatically configures search parameters based on the dataset and architecture. The format is `K_VALUE NPROBE THRESHOLD PRED_NPROBE SOAR_LAMBDA`.

### Equivalence optimization parameters (applicable to AArch64)

| Dataset| K\_VALUE | NPROBE | THRESHOLD | PRED\_NPROBE | SOAR\_LAMBDA |
|--------|---------|--------|-----------|-------------|-------------|
| sift | 2048 | 77 | 0 | 0 | 0 |
| deep | 4096 | 95 | 0 | 0 | 0 |
| glove | 2048 | 765 | 0 | 0 | 0 |
| fashion | 128 | 6 | 0 | 0 | 0 |
| gist | 2048 | 100 | 0 | 0 | 0 |

### Non-equivalence optimization parameters (AArch64)

| Dataset| K\_VALUE | NPROBE | THRESHOLD | PRED\_NPROBE | SOAR\_LAMBDA |
|--------|---------|--------|-----------|-------------|-------------|
| sift | 2048 | 77 | 0.4 | 40 | 0 |
| deep | 4096 | 90 | 0 | 0 | 1.2 |
| glove | 2048 | 765 | 0 | 0 | 0 |
| fashion | 128 | 6 | 0 | 0 | 0 |
| gist | 2048 | 100 | 0.34 | 43 | 1.2 |

Description

- `THRESHOLD > 0`: Enables ML-based adaptive nprobe (enabled for `sift` and `gist`).
- `PRED_NPROBE > 0`: The lower, dedicated nprobe value applied when a query is predicted as "easy" by the ML model.
- `SOAR_LAMBDA > 0`: Enables SOAR vector search (enabled for `deep` and `gist`).

## Environment Configuration

### HDF5 library path

- For equivalence index optimization, the `run.sh` script requires manual configuration of the HDF5 library path.

   ```bash
   HDF5_LIB_ROOT_DIR="/path/to/HDF5"
   ```

- The non-equivalence index optimization defaults to the system paths: `/usr/include/hdf5/serial/` and `/usr/lib/aarch64-linux-gnu/hdf5/serial/`.

### Compiler selection

| Architecture| Equivalence Optimization| Non-Equivalent Optimization|
|------|---------|-----------|
| x86\_64 | clang++, `-march=native`| index: g++; search: clang++, `-march=native`|
| aarch64 | clang++, `-march=armv8-a+fp16fml`| index: g++, `-march=armv8.2-a+fp16fml+dotprod`; search: clang++, `-march=armv8-a+fp16fml`|

For Am64 search compilation, the assembly file `krl_table_lookup_fast_scan.s` and `jemalloc` must be additionally linked.

## Example

### Complete process of equivalence index optimization

Apply the equivalence index optimization patch.

```bash
./run.sh sift fastscan all
```

The command output contains:

- Configuration information (dataset, mode, and parameters)
- Data generation progress
- Index build status
- Search result (QPS per round, thread statistics, and final recall rate)

### Complete process of non-equivalence optimization

Apply the non-equivalence index optimization patch.

```bash
./run.sh sift fastscan all
```

The complete `all` process is executed in the following sequence:

1. `generate`: generates IVF clustering, RaBitQ quantization, and SOAR data.
2. `index`: builds indexes (including spilling data).
3. `train`: runs `search_model` to generate the optimal nprobe label data (only for AArch64).
4. `eval`: trains the LightGBM model and exports the `.so` file (only for AArch64).
5. `search`: performs search (with SOAR and ML-based adaptive nprobe enabled).

### Step-by-step debugging

- Rebuild the index only.

  ```bash
  ./run.sh deep fastscan index
  ```

- Perform search only (the `generate` and `index` phases must be completed first).

  ```bash
  ./run.sh deep fastscan search
  ```

- Train the ML model only (non-equivalence optimization; the `generate` and `index` phases must be completed first).

  ```bash
  ./run.sh sift fastscan train
  ./run.sh sift fastscan eval
  ```

### Switching the scanning mode

- Use SIMD for fast scanning (recommended for better performance).

  ```bash
  ./run.sh sift fastscan all
  ```

- Use bitwise scanning (used for correctness verification).

  ```bash
  ./run.sh sift scan all
  ```
