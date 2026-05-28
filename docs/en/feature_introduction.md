# Feature Guide

## Technical Structure

```
Original RaBitQ (x86_64 AVX2)
│
├── 0002-rabitq-optimize-eqv.patch ──→ Equivalence optimization version (preserving algorithmic semantics)
│   ├── AArch64 NEON vectorization
│   ├── FP16 precision + batched factor structure
│   ├── Assembly-level LUT acceleration + SIMD mask filtering
│   ├── Adaptive batch processing (64/96)
│   ├── HDF5 data format + support for multiple metric types
│   └── Multi-threaded search framework (pthread + NUMA)
│
└── 0001-rabitq-optimize-neq.patch ──→ Non-equivalence optimization version (changing search behavior)
    ├── All contents of the equivalence optimization +
    ├── SOAR spilled vector assignment (index phase, extending the index structure)
    ├── Dual-cluster search and deduplication (search phase)
    └── ML-based adaptive nprobe (search phase, without changing the index structure)
        ├── Offline: LightGBM training → treelite export .so
        └── Online: Query feature extraction → model inference → dynamic nprobe adjustment
```

## Equivalence Index Optimization (0002-rabitq-optimize-eqv.patch)

Performance optimization is tailored for AArch64 while preserving algorithmic semantics; search results remain identical to the original implementation.

### Main Optimizations

- **FP16 precision optimization**: Introduces `float16_t` storage for centroids, data vectors, and factor structures, reducing memory footprint and bandwidth consumption while boosting SIMD throughput.
- **NEON SIMD vectorization**: Replaces x86 AVX2 implementations with Arm NEON instructions, covering critical paths such as distance computation, bit transposition, and scalar quantization.
- **Assembly-level LUT optimization**: Adds `krl_table_lookup_fast_scan.s`, leveraging AArch64 assembly for batched LUT lookups to optimize instruction scheduling and prefetching strategies.
- **Adaptive batch processing**: Automatically selects batch sizes (64/96) based on vector dimensions, enhancing cache utilization compared to the original fixed batch size `32`.
- **SIMD mask filtering**: Generates masks via NEON vector comparisons to batch skip candidate vectors that fail to meet the distance threshold, reducing branch prediction overhead.
- **Memory alignment optimization**: Enforces 64/256-byte alignment on critical data structures to improve NEON memory access efficiency.
- **Batched factor structure**: Restructures the factor structure into batched layout (grouped by 4 or 16 vectors) to facilitate SIMD parallel processing.

### File Changes

| File| Change Type| Description|
|------|---------|------|
| `src/ivf_rabitq.h` | Modified| Added the AArch64 data structure and search process.|
| `src/ivf_rabitq_search.h` | New| Search implementation optimized for AArch64.|
| `src/index_io.h` | New| AArch64 index I/O (including FP16 conversion).|
| `src/space.h` | Modified| NEON SIMD implementation (bit operation and distance computation).|
| `src/fast_scan.h` | Modified| Adaptive batch packing and NEON fast scanning.|
| `src/matrix.h` | Modified| Matrix operation adaptation.|
| `src/index.cpp` | Modified| Index construction process adaptation.|
| `src/search.cpp` | Modified| Search process adaptation.|
| `src/test_result.cpp` | Modified| Test result processing.|
| `data/ivf.py` | Modified| Optimized IVF clustering workflow; added SOAR assignment and dot product metric support.|
| `data/rabitq.py` | Modified| Quantized index construction adaptation.|
| `run.sh` | New| One-click execution script.|

## Non-Equivalence Index Optimization (0001-rabitq-optimize-neq.patch)

The Spilling with Orthogonality-Amplified Residuals (SOAR) algorithm is introduced to break the single-cluster assignment constraint of traditional IVF. It allows vectors to concurrently belong to both its original cluster and a spilled cluster, trading a minor memory and computational overhead for a significantly higher recall rate.

### Main Optimizations

- **SOAR spilled vector assignment**: Computes the optimal spilled cluster assignment for each vector based on a joint loss function of the distance term and projection energy.
  - Loss function: `loss = ||x - c'||² + λ * (r·r')² / ||r||²`, where `λ` (`soar_lambda`) controls the weight of the projection term.
  - Batch processing is supported (with the default batch size `8000`), improving assignment efficiency.
- **Spilling data structure**: Extends the index structure to store the binary codes, FP16 data, and factors of spilled vectors.
- **SOAR search**: Scans both the original and spilled clusters simultaneously during execution, utilizing `unordered_set` for deduplication to prevent redundant computations.
- **AArch64 optimization**: Shares low-level optimizations (FP16, NEON SIMD, assembly LUT) with the equivalence optimization codebase.
- **ML-based adaptive nprobe**: Dynamically adjusts nprobe during the search phase using a LightGBM model (detailed below).

### File Changes

| File| Change Type| Description|
|------|---------|------|
| `src/ivf_rabitq.h` | Modified| Added the SOAR data structure and search entry points.|
| `src/ivf_rabitq_search.h` | New| Implemented SOAR search (including deduplication and dual-cluster scanning).|
| `src/index_io.h` | New| Persistent storage for SOAR indexes (including spilled data)|
| `src/space.h` | Modified| NEON SIMD implementation.|
| `src/fast_scan.h` | Modified| Batch scanning optimization.|
| `src/index.cpp` | Modified| Index construction (including spilled data)|
| `src/krl_table_lookup_fast_scan.s` | New| Optimized AArch64 assembly.|
| `data/ivf.py` | Modified| IVF clustering + SOAR spilled assignment.|
| `data/rabitq.py` | Modified| Quantized index construction (including spilling data)|
| `data/eval.py` | New| Feature extraction and LightGBM evaluation.|
| `data/test.py` | New| Test script.|
| `run.sh` | New| One-click execution script.|
| `script/index.sh` | Modified| Index construction script (including SOAR parameters).|
| `script/search.sh` | Modified| Search script (including SOAR parameters).|

## ML-based Adaptive nprobe

In addition to the SOAR spilled vector assignment, the non-equivalence optimization introduces a **machine learning-based query-adaptive nprobe prediction** mechanism. This mechanism leaves the **index structure untouched**. It dynamically scales the number of probed clusters (nprobe) based on the features of the query vector during execution, shaving off immense computational workloads for "simple" queries to drive up overall throughput.

### Offline training phase (`data/eval.py`)

1. **Data preparation**: Loads the base vector data and the precomputed optimal nprobe per query (`expectedNprobe1.bin`).
2. **Feature extraction**: Extracts four-dimensional features from each vector, including the mean, standard deviation, sparsity, and norm.
3. **Label construction**: Leverages the median nprobe as the threshold for binary classification. Queries with an nprobe greater than the median are labeled as `1` ("difficult" queries), while the rest are labeled as `0` ("simple" queries).
4. **Model training**: Trains a classification model via LightGBM (GBDT, binary objective, AUC metric), establishing the optimal classification threshold via Youden's Index.
5. **Model export**: Compiles the trained LightGBM model into a C shared library (`libadaptivemodel_less.so`) via treelite and tl2cgen, allowing direct invocation by the C++ search program.

### Online search phase (`src/ivf_rabitq_search.h`)

Adaptive nprobe is automatically triggered during execution when `threshold > 0` and `pred_nprobe > 0`.

1. Loads `libadaptivemodel_less.so` using `dlopen()` to retrieve the `predict` function pointer.
2. Extracts the identical 4-dimensional features (mean, std, sparsity, and norm) from each query vector in real time.
3. Call the model inference: `pred(fdata, 0, &result)`.
4. If `result <= threshold` (the query is predicted as a "simple" query), reduce the nprobe value from the default value to `pred_nprobe`.
5. Otherwise, retain the default nprobe value.

### Impact on Index Structure

The ML-based adaptive nprobe mechanism enforces **zero modifications on the index structure**. The index construction, storage, and loading pipelines remain entirely identical to those without ML. The ML model acts strictly as a bypass predictor during the search phase, drastically scaling up QPS by truncating nprobe for simple queries and minimizing average search overhead without sacrificing recall. This mechanism can be completely bypassed by applying `threshold=0`.
