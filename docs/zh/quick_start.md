# 快速入门

## 简介

RaBitQ是一种面向高维向量近似最近邻搜索（ANN）的随机二值量化方法，能将D维向量量化为D位二进制字符串，并提供理论误差界。原始实现基于x86\_64 AVX2指令集，本补丁将其扩展至ARM64（AArch64）架构，并引入多项性能优化。

本仓库基于[RaBitQ](https://github.com/gaoj0017/RaBitQ)（SIGMOD 2024）开源项目提供ARM64平台优化的补丁，涵盖**等价索引优化**和**非等价索引优化**两个方向。补丁文件说明如[**表 1** 补丁优化说明](#补丁优化说明)所示。

**表 1** 补丁优化说明<a id="补丁优化说明"></a>

| 补丁文件 | 类型 | 说明 |
| --------- | ------ | ------ |
| `0001-rabitq-optimize-neq.patch` | 非等价索引优化 | 引入SOAR算法，允许向量被分配到多个聚类簇，提升召回率 |
| `0002-rabitq-optimize-eqv.patch` | 等价索引优化 | 在保持算法等价性的前提下，针对ARM64平台深度优化搜索性能 |

## 前提条件

- ARM64（AArch64）架构的Linux环境（x86\_64也可编译运行，但无法启用ARM优化）。
- HDF5开发库（`libhdf5-dev`）。
- 依赖库：Python 3.8+，NumPy，Faiss，tqdm，h5py，scikit-learn, pandas, lightgbm。
- treelite, tl2cgen（ML模型导出，仅非等价优化需要）。
- 支持NEON指令集的编译器（推荐llvm 16.0.6），具体安装见[安装 LLVM 16.0.6](#安装llvm-1606)
- ARM64优化依赖：jemalloc（内存分配优化），具体安装见[安装 jemalloc](#安装jemalloc)
- Eigen 3.4.0（将`Eigen`文件夹放置于`./src/`下），具体安装见[安装 Eigen 3.4.0](#安装eigen-340)

## 编译依赖

### 安装LLVM 16.0.6

1. 下载并解压LLVM 16.0.6源码。

   ```bash
   wget -O llvm-project-16.0.6.src.tar.xz https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.6/llvm-project-16.0.6.src.tar.xz --no-check-certificate
   tar xf llvm-project-16.0.6.src.tar.xz
   ```

2. 编译LLVM（耗时较长，建议后台执行）。

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

   编译（-j后接CPU核心数，如8核写-j8）。

   ```bash
   ninja -j$(nproc)
   ```

3. 安装到指定目录。

   ```text
   ninja install 
   ```  

4. 配置 LLVM 环境变量（临时生效）。

   ```bash
   export CXX=/opt/llvm-16.0.6/bin/clang++
   export CC=/opt/llvm-16.0.6/bin/clang
   export PATH=/opt/llvm-16.0.6/bin:$PATH
   ```

5. 验证是否安装成功。

   ```bash
   clang --version
   ```

   成功应输出如下显示：
  
   ```text
   clang version 16.0.6
   ```

### 安装jemalloc

1. 下载jemalloc 5.3.0并解压。

   ```bash
   wget --no-check-certificate https://github.com/jemalloc/jemalloc/releases/download/5.3.0/jemalloc-5.3.0.tar.bz2
   tar -jxvf jemalloc-5.3.0.tar.bz2
   cd jemalloc-5.3.0
   ```

2. 编译安装。

   ```bash
   ./configure --prefix=/usr/local
   make -j$(nproc)
   make install
   ```

3. 验证是否安装成功。

   ```bash
   jemalloc-config --version
   ```

   成功应输出版本号：5.3.0。

4. 配置jemalloc库路径（Debian通用）。

   ```bash
   echo "/usr/local/lib" >> /etc/ld.so.conf
   ldconfig
   ```

### 安装Eigen 3.4.0

1. 下载Eigen源码并解压。

   ```bash
   wget --no-check-certificate https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
   tar -zxvf eigen-3.4.0.tar.gz
   ```

2. 移动核心头文件到src根目录（匹配代码的 #include <Eigen/Dense>）。

   ```bash
   mv eigen-3.4.0/Eigen/ ./
   ```

3. 验证文件是否存在。

   ```bash
   ls ./Eigen/Dense
   ```

   成功应输出：./Eigen/Dense

## 合入补丁

1. 获取RaBitQ开源代码。

   ```bash
   git clone https://github.com/gaoj0017/RaBitQ.git
   ```

2. 获取基于RaBitQ开源项目的ARM64平台优化补丁。

   ```bash
   git clone https://gitcode.com/boostkit/rabitq.git -b v 1.0.0
   ```

3. 将补丁文件复制到当前目录。

   ```bash
   cp rabitq/0001-rabitq-optimize-neq.patch ./
   cp rabitq/0002-rabitq-optimize-eqv.patch ./
   ```

4. 合入补丁。（二选一，不可同时合入。）

   - 合入非等价索引优化补丁

     ```bash
     patch -p1 < ../0001-rabitq-optimize-neq.patch
     ```

   - 合入等价索引优化补丁

     ```bash
     patch -p1 < ../0002-rabitq-optimize-eqv.patch
     ```

   合入补丁后RaBitQ完整的目录结构如下所示：

   ```text
   RaBitQ/
   ├─ src/                                    // C++源代码
   │   ├─ ivf_rabitq.h                       // IVF-RaBitQ主类（含ARM64 SOAR数据结构）
   │   ├─ ivf_rabitq_search.h                // ARM64优化的搜索实现（含SOAR/掩码扫描）
   │   ├─ index_io.h                         // ARM64索引I/O（含FP16转换）
   │   ├─ space.h                            // 位操作与距离计算（ARM NEON实现）
   │   ├─ fast_scan.h                        // SIMD快速扫描（自适应批量64/96）
   │   ├─ krl_table_lookup_fast_scan.s       // ARM64汇编LUT查找优化
   │   ├─ matrix.h                           // 矩阵数据结构
   │   ├─ utils.h                            // 工具函数（HDF5加载、时间测量等）
   │   ├─ test_result.h                      // 测试结果统计
   │   ├─ test_result.cpp                    // 测试结果实现
   │   ├─ index.cpp                          // 索引构建主程序
   │   ├─ search.cpp                         // 搜索主程序（多线程、NUMA绑核）
   │   └─ search_model.cpp                   // ML训练数据生成程序（仅非等价）
   ├─ data/                                   // Python数据处理
   │   ├─ ivf.py                             // IVF聚类 + SOAR溢出分配
   │   ├─ rabitq.py                          // RaBitQ量化索引构建
   │   ├─ eval.py                            // LightGBM模型训练与导出（仅非等价）
   │   ├─ test.py                            // 向量归一化测试（仅非等价）
   │   └─ utils/
   │       └─ io.py                          // 数据I/O工具（fvecs/ivecs/HDF5）
   ├─ script/                                 // 脚本
   │   ├─ index.sh                           // 索引构建脚本
   │   └─ search.sh                          // 搜索脚本
   ├─ bin/                                    // 编译输出目录
   ├─ results/                                // 搜索结果输出目录
   ├─ datasets/                               // HDF5数据集目录
   ├─ run.sh                                  // 一键运行脚本（统一入口）
   ├─ LICENSE
   └─ README.md
   ```

## 运行

### 使用脚本运行（推荐）

应用补丁后，推荐使用 `run.sh` 脚本统一管理全流程。

- 等价索引优化补丁：数据生成 + 索引构建 + 搜索

  ```bash
  ./run.sh sift fastscan all
  ```

- 非等价索引优化补丁：数据生成 + 索引构建 + ML 训练 + 搜索

  ```bash
  ./run.sh sift fastscan all
  ```

- （可选）分步执行。

  ```bash
  ./run.sh sift fastscan generate   # 生成 IVF 聚类 + RaBitQ 量化数据
  ./run.sh sift fastscan index      # 构建索引
  ./run.sh sift fastscan train      # [仅非等价] 生成 ML 训练数据
  ./run.sh sift fastscan eval       # [仅非等价] 训练 LightGBM 模型
  ./run.sh sift fastscan search     # 执行搜索
  ```

支持的数据集：`sift`、`deep`、`glove`、`fashion`、`gist`。更多 `run.sh` 的详细使用方法请参见《[用户指南](user_guide.md)》。

### 手动运行

1. Python索引阶段。

   ```bash
   cd data
   python ivf.py <hdf5_path> <dataset> <K> <metric_type> <soar_lambda>
   python rabitq.py <hdf5_path> <dataset> <K> <metric_type> <soar_lambda>
   cd ..
   ```

2. C++索引构建。

   ```bash
   ./bin/index_<dataset> -d <dataset> -s <source> -p <hdf5_path> -m <metric_type> -a <soar_lambda>
   ```

3. C++搜索。

   ```bash
   ./bin/search_<dataset> -d <dataset> -s <source> -r <result_path> -k <topk> -n <nprobe> \
    -p <hdf5_path> -m <metric_type> -t <threshold> -e <pred_nprobe> -a <soar_lambda>
   ```
