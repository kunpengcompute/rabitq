
# Quick Start

## Overview

This repository provides patches optimized for the AArch64 platform based on the [RaBitQ](https://github.com/gaoj0017/RaBitQ) (SIGMOD 2024) open-source project, covering both equivalence index optimization and non-equivalence index optimization.

RaBitQ is a randomized binary quantization method for high-dimensional Approximate Nearest Neighbor (ANN) search. It quantizes a D-dimensional vector into a D-bit binary string while providing a theoretical error bound. The original implementation relies on the x86_64 AVX2 instruction set; these patches extend it to the AArch64 architecture and introduces multiple performance optimizations.

## Patch Overview

| Patch File| Type| Description|
|---------|------|------|
| `0001-rabitq-optimize-neq.patch` | Non-equivalence index optimization| Introduces the Spilling with Orthogonality-Amplified Residuals (SOAR) algorithm, allows spilled vector assignment across multiple clusters to improve the recall rate.|
| `0002-rabitq-optimize-eqv.patch` | Equivalence index optimization| Deeply optimizes search performance for the AArch64 platform while preserving algorithmic semantics.|

## Prerequisites

- Linux environment on the AArch64 architecture (the code can also be compiled and executed on x86_64, but Arm-specific optimizations cannot be enabled).
- Eigen 3.4.0 (place the `Eigen` folder under `./src/`). Install it by following the instructions in [Installing Eigen 3.4.0](#installing-eigen-340).
- HDF5 development library (`libhdf5-dev`).
- Python 3.8+, NumPy, Faiss, tqdm, h5py, scikit-learn, pandas, and lightgbm.
- treelite, tl2cgen (used for ML model export; required only for non-equivalence optimization).
- Compiler with NEON instruction set support (LLVM 16.0.6 is recommended). Install it by following the instructions in [Installing LLVM 16.0.6](#installing-llvm-1606).
- AArch64 optimization dependency: jemalloc (memory allocation optimization). Install it by following the instructions in [Installing jemalloc](#installing-jemalloc).

## Compiling Dependencies

### Installing LLVM 16.0.6

1. Download and decompress the LLVM 16.0.6 source package.

   ```bash
   wget -O llvm-project-16.0.6.src.tar.xz https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.6/llvm-project-16.0.6.src.tar.xz --no-check-certificate
   tar xf llvm-project-16.0.6.src.tar.xz
   ```

2. Compile LLVM (this process takes a long time;running in the background is recommended).

   ```bash
   cd llvm-project-16.0.6.src
   mkdir -p build && cd build
   cmake -G Ninja ../llvm \
     -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_INSTALL_PREFIX=/opt/llvm-16.0.6 \
     -DLLVM_TARGETS_TO_BUILD="AArch64" \
     -DLLVM_ENABLE_PROJECTS="clang;lld;clang-tools-extra;openmp" \
     -DLLVM_ENABLE_TERMINFO=ON \
     -DLLVM_ENABLE_ZLIB=ON
   ```

   Perform the compilation (the `-j` option specifies the number of CPU cores, for example, `-j8` for 8 cores.)

   `ninja -j$(nproc)`
3. Install it to a specified directory.

   ```bash
   ninja install 
   ```  

4. Configure the LLVM environment variables (temporarily effective).

   ```bash
   export CXX=/opt/llvm-16.0.6/bin/clang++
   export CC=/opt/llvm-16.0.6/bin/clang
   export PATH=/opt/llvm-16.0.6/bin:$PATH
   ```

5. Check whether the installation is successful.

   ```bash
   clang --version
   ```

   If the installation is successful, the following information is displayed:
  
   `clang version 16.0.6`

### Installing jemalloc

1. Download jemalloc 5.3.0 and decompress it.

   ```bash
   wget --no-check-certificate https://github.com/jemalloc/jemalloc/releases/download/5.3.0/jemalloc-5.3.0.tar.bz2
   tar -jxvf jemalloc-5.3.0.tar.bz2
   cd jemalloc-5.3.0
   ```

2. Perform compilation and installation.

   ```bash
   ./configure --prefix=/usr/local
   make -j$(nproc)
   make install
   ```

3. Check whether the installation is successful.

   ```bash
   jemalloc-config --version
   ```

   If the installation is successful, the version number 5.3.0 is displayed.

4. Configure the jemalloc library path (generic for Debian).

   ```bash
   echo "/usr/local/lib" >> /etc/ld.so.conf
   ldconfig
   ```

### Installing Eigen 3.4.0

1. Download the Eigen source code and decompress it.

   ```bash
   wget --no-check-certificate https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
   tar -zxvf eigen-3.4.0.tar.gz
   ```

2. Move the core header file to the root directory `src1 (matching the #include <Eigen/Dense> in the code).

   ```bash
   mv eigen-3.4.0/Eigen/ ./
   ```

3. Check whether the file exists.

   ```bash
   ls ./Eigen/Dense
   ```

   If the operation is successful, the following information is displayed: ./Eigen/Dense

## Applying the Patch

1. Obtain the open-source code of RaBitQ.

   ```bash
   git clone https://github.com/gaoj0017/RaBitQ.git
   ```

2. Obtain patches optimized for the AArch64 platform based on the open-source RaBitQ.

   ```bash
   git clone https://gitcode.com/boostkit/rabitq.git -b v 1.0.0
   ```

3. Copy the patch files to the current directory.

   ```bash
   cp rabitq/0001-rabitq-optimize-neq.patch ./
   cp rabitq/0002-rabitq-optimize-eqv.patch ./
   ```

4. Apply a patch. (Select either of the patches; do not apply both.)
   - Apply the non-equivalence index optimization patch.

     ```bash
     patch -p1 < ../0001-rabitq-optimize-neq.patch
     ```

   - Apply the equivalence index optimization patch.

     ```bash
     patch -p1 < ../0002-rabitq-optimize-eqv.patch
     ```

The full directory structure of RaBitQ after applying the patch is as follows:

```text
RaBitQ/
├─ src/                                   // C++ source code
│   ├─ ivf_rabitq.h                       // IVF-RaBitQ main class (including AArch64 SOAR data structure)
│   ├─ ivf_rabitq_search.h                // AArch64 optimized search implementation (including SOAR/mask scanning)
│   ├─ index_io.h                         // Arm 64 index I/O (including FP16 conversion)
│   ├─ space.h                            // Bitwise operations and distance computation (Arm NEON implementation)
│   ├─ fast_scan.h                        // SIMD fast scanning (adaptive batch 64/96)
│   ├─ krl_table_lookup_fast_scan.s       // Arm 64 assembly LUT search optimization
│   ├─ matrix.h                           // Matrix data structure
│   ├─ utils.h                            // Tool functions (HDF5 loading, time measurement, etc.)
│   ├─ test_result.h                      // Test result statistics
│   ├─ test_result.cpp                    // Test result implementation
│   ├─ index.cpp                          // Main program for index construction
│   ├─ search.cpp                         // Main program for search (multi-threading, NUMA core pinning)
│   └─ search_model.cpp                   // ML training data generation program (non-equivalence only)
├─ data/                                  // Python data processing
│   ├─ ivf.py                             // IVF clustering + SOAR spilled assignment
│   ├─ rabitq.py                          // RaBitQ quantized index construction
│   ├─ eval.py                            // LightGBM model training and export (non-equivalence only)
│   ├─ test.py                            // Vector normalization test (non-equivalence only)
│   └─ utils/
│       └─ io.py                          // Data I/O tool (fvecs/ivecs/HDF5)
├─ script/                                // Scripts
│   ├─ index.sh                           // Index construction script
│   └─ search.sh                          // Search script
├─ bin/                                   // Compilation output directory
├─ results/                               // Output directory for search results
├─ datasets/                              // HDF5 dataset directory
├─ run.sh                                 // One-click execution script (unified entry)
├─ LICENSE
└─ README.md
```

## Running

### Running with Script (Recommended)

After the patch is installed, it is recommended to use the `run.sh` script to manage the entire process.

- Equivalence index optimization patch: Data generation + Index building + Search

  ```bash
  ./run.sh sift fastscan all
  ```

- Non-equivalence index optimization patch: Data generation + Index building + ML training + Search

  ```bash
  ./run.sh sift fastscan all
  ```

- (Optional) Perform the operations step by step.

  ```bash
  ./run.sh sift fastscan generate   # Generate IVF clustering and RaBitQ quantization data.
  ./run.sh sift fastscan index      # Build indexes.
  ./run.sh sift fastscan train      # [Non-equivalence only] Generate ML training data.
  ./run.sh sift fastscan eval       # [Non-equivalence only] Train the LightGBM model.
  ./run.sh sift fastscan search     # Perform search.
  ```

Supported datasets: `sift`, `deep`, `glove`, `fashion`, and `gist`. For details about how to use the `run.sh` script, see the [User Guide](user_guide.md).

### Manual running

1. Python indexing phase

   ```bash
   cd data
   python ivf.py <hdf5_path> <dataset> <K> <metric_type> <soar_lambda>
   python rabitq.py <hdf5_path> <dataset> <K> <metric_type> <soar_lambda>
   cd ..
   ```

2. C++ index building

   ```bash
   ./bin/index_<dataset> -d <dataset> -s <source> -p <hdf5_path> -m <metric_type> -a <soar_lambda>
   ```

3. C++ search

   ```bash
   ./bin/search_<dataset> -d <dataset> -s <source> -r <result_path> -k <topk> -n <nprobe> \
    -p <hdf5_path> -m <metric_type> -t <threshold> -e <pred_nprobe> -a <soar_lambda>
   ```
