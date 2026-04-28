# 用户指南

## 概述

`run.sh`是优化后的统一测试脚本，整合了数据生成、索引构建、模型训练和搜索测试的完整流程。脚本自动检测系统架构（x86\_64 / aarch64），选择对应的编译参数和运行策略。

等价优化和非等价优化各自提供了一个`run.sh`脚本，两者共享相同的基础用法，非等价优化额外支持`train`和`eval`两个阶段。

## 基本用法

`run.sh`基本用法如下命令所示，命令中的参数值与说明如[**表 1** run.sh脚本使用参数说明](#run.sh脚本使用参数说明)所示。

```bash
./run.sh <dataset> [mode] [stage]
```

**表 1** run.sh脚本使用参数说明<a id="run.sh脚本使用参数说明"></a>

| 参数 | 说明 | 可选值 | 默认值 |
| ------ | ------ | -------- | -------- |
| `dataset` | 数据集名称（必填） | `sift`，`deep`，`glove`，`fashion`，`gist` | - |
| `mode` | 扫描模式 | `fastscan`，`scan` | `fastscan` |
| `stage` | 执行阶段 | 请参见[执行阶段](#执行阶段)。 | `all` |

### 执行阶段

**等价优化支持的阶段**

等价优化支持的阶段说明如[**表 2** 等价优化支持的阶段说明](#等价优化支持的阶段说明)所示。

**表 2** 等价优化支持的阶段说明<a id="等价优化支持的阶段说明"></a>

| 阶段 | 说明 |
| ------ | ------ |
| `generate` | 运行`ivf.py`和`rabitq.py`文件，执行IVF聚类和RaBitQ量化，生成索引所需的中间数据。 |
| `index` | 编译并运行C++ `index`程序，从中间数据构建二进制索引文件。 |
| `search` | 编译并运行C++ `search`程序，执行搜索测试并输出QPS和召回率。 |
| `all` | 按顺序执行`generate`，`index`，`search`阶段。 |

**非等价优化额外支持的阶段**

非等价优化除支持等价优化的阶段外，还额外支持以下阶段，额外支持的阶段说明如[**表 3** 非等价优化额外支持的阶段说明](#非等价优化额外支持的阶段说明)所示。

**表 3** 非等价优化额外支持的阶段说明<a id="非等价优化额外支持的阶段说明"></a>

| 阶段 | 说明 |
| ------ | ------ |
| `train` | 编译并运行`search_model`程序，对每个基向量搜索最优nprobe，生成`approximateGT.bin`和`expectedNprobe1.bin`（仅ARM64）。 |
| `eval` | 运行`eval.py`文件，训练LightGBM分类模型并通过treelite导出为`libadaptivemodel_less.so`共享库（仅ARM64）。 |
| `all` | 按顺序执行`generate`，`index`，`train`，`eval`，`search`阶段。 |

## 数据集配置

`run.sh`脚本内置了每个数据集的维度、量化位宽、HDF5文件名和度量类型，如[**表 4** 数据集配置说明](#数据集配置说明)所示。其中HDF5数据文件需放置在`./datasets/`目录下。

**表 4** 数据集配置说明<a id="数据集配置说明"></a>

| 数据集 | HDF5 文件 | 度量类型 | D（维度） | B（量化位宽） |
| -------- | ---------- | --------- | ----------- | -------------- |
| `sift` | `sift-128-euclidean.hdf5` | squared\_l2 | 128 | 128 |
| `deep` | `deep-image-96-angular.hdf5` | dot\_product | 96 | 128 |
| `glove` | `glove-100-angular.hdf5` | dot\_product | 100 | 128 |
| `fashion` | `fashion-mnist-784-euclidean.hdf5` | squared\_l2 | 784 | 832 |
| `gist` | `gist-960-euclidean.hdf5` | squared\_l2 | 960 | 960 |

## 搜索参数

脚本根据数据集和架构自动配置搜索参数，格式为`K_VALUE NPROBE THRESHOLD PRED_NPROBE SOAR_LAMBDA`。

### 等价优化参数（适用于aarch64）

| 数据集 | K\_VALUE | NPROBE | THRESHOLD | PRED\_NPROBE | SOAR\_LAMBDA |
| -------- | --------- | -------- | ----------- | ------------- | ------------- |
| sift | 2048 | 77 | 0 | 0 | 0 |
| deep | 4096 | 95 | 0 | 0 | 0 |
| glove | 2048 | 765 | 0 | 0 | 0 |
| fashion | 128 | 6 | 0 | 0 | 0 |
| gist | 2048 | 100 | 0 | 0 | 0 |

### 非等价优化参数（aarch64）

| 数据集 | K\_VALUE | NPROBE | THRESHOLD | PRED\_NPROBE | SOAR\_LAMBDA |
| -------- | --------- | -------- | ----------- | ------------- | ------------- |
| sift | 2048 | 77 | 0.4 | 40 | 0 |
| deep | 4096 | 90 | 0 | 0 | 1.2 |
| glove | 2048 | 765 | 0 | 0 | 0 |
| fashion | 128 | 6 | 0 | 0 | 0 |
| gist | 2048 | 100 | 0.34 | 43 | 1.2 |

参数含义：

- **THRESHOLD > 0**：启用ML自适应nprobe（`sift`和`gist`启用）。
- **PRED\_NPROBE > 0**：ML预测为“简单”查询时使用的缩减nprobe值。
- **SOAR\_LAMBDA > 0**：启用SOAR溢出向量搜索（`deep`和`gist`启用）。

## 环境配置

### HDF5库路径

- 等价索引优化的`run.sh` 需要手动配置HDF5库路径。

   ```bash
   HDF5_LIB_ROOT_DIR="/path/to/HDF5"
   ```

- 非等价索引优化默认使用系统路径`/usr/include/hdf5/serial/`和`/usr/lib/aarch64-linux-gnu/hdf5/serial/`。

### 编译器选择

| 架构 | 等价优化 | 非等价优化 |
| ------ | --------- | ----------- |
| x86\_64 | clang++，`-march=native` | index: g++；search: clang++，`-march=native` |
| aarch64 | clang++，`-march=armv8-a+fp16fml` | index: g++，`-march=armv8.2-a+fp16fml+dotprod`；search: clang++，`-march=armv8-a+fp16fml` |

ARM64搜索编译额外链接汇编文件`krl_table_lookup_fast_scan.s`和jemalloc。

## 使用示例

### 等价索引优化完整流程

应用等价索引优化补丁。

```bash
./run.sh sift fastscan all
```

输出包含：

- 配置信息（数据集、模式、参数）
- 数据生成进度
- 索引构建状态
- 搜索结果（每轮QPS、线程统计、最终召回率）

### 非等价优化完整流程

应用非等价索引优化补丁。

```bash
./run.sh sift fastscan all
```

完整`all`流程依次执行：

1. `generate`：生成IVF聚类 + RaBitQ量化 + SOAR溢出数据。
2. `index`：构建索引（含溢出数据）。
3. `train`：运行`search_model`生成最优nprobe标签数据（仅ARM64）。
4. `eval`：训练LightGBM模型并导出`.so`（仅ARM64）。
5. `search`：搜索（启用SOAR + ML自适应nprobe）。

### 分步调试

- 仅重新构建索。

  ```bash
  ./run.sh deep fastscan index
  ```
  
- 仅执行搜索（需先完成generate + index）。

  ```bash
  ./run.sh deep fastscan search
  ```

- 仅训练ML模型（非等价，需先完成generate + index）。

  ```bash
  ./run.sh sift fastscan train
  ./run.sh sift fastscan eval
  ```

### 切换扫描模式

- 使用SIMD快速扫描（推荐，性能更优）。

  ```bash
  ./run.sh sift fastscan all
  ```

- 使用逐位操作扫描（用于验证正确性）。

  ```bash
  ./run.sh sift scan all
  ```
