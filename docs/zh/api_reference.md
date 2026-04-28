# API参考

本文档详细说明等价索引优化和非等价索引优化相对于原始RaBitQ开源代码的接口变动。

## 等价索引优化接口变动

### C++命令行参数

**index程序**

| 参数 | 原始 | 优化后 |
| ------ | ------ | -------- |
| `-d` (dataset) | 有 | 有 |
| `-s` (source) | 有 | 有 |
| `-p` (data\_path) | - | 新增，HDF5数据文件路径 |
| `-m` (metric\_type) | - | 新增，度量类型（squared\_l2 / dot\_product） |
| `-a` (soar\_lambda) | - | 新增，SOAR参数（设为0禁用） |

数据加载方式从直接读取fvecs改为通过loadHDFBase()读取HDF5格式。

**search程序**

| 参数 | 原始 | 优化后 |
| ------ | ------ | -------- |
| `-d` (dataset) | 有 | 有 |
| `-s` (source) | 有 | 有 |
| `-r` (result\_path) | 有 | 有 |
| `-k` (topK) | 有 | 有 |
| `-n` (nprobe) | - | 新增，探查簇数量 |
| `-p` (data\_path) | - | 新增，HDF5数据文件路径 |
| `-m` (metric\_type) | - | 新增，度量类型 |

数据加载方式从 fvecs/ivecs 改为通过loadHDF()读取HDF5格式。搜索框架从单线程改为多线程（pthread绑核 + OpenMP），支持多NUMA节点并发。

### C++ IVFRN类

IVFRN类的公有接口签名保持不变，ARM64平台下内部新增以下私有成员和方法：

| 变更 | 说明 |
| ------ | ------ |
| `Factor` 结构 (ARM64) | 批量布局：`float sqr_x[4]` 等，替代原始逐元素 `float sqr_x` |
| `Factor_f16` 结构 | 新增FP16批量因子结构（16个元素一组） |
| `search_fast_scan()` | 新增私有方法，ARM64NEON 优化的搜索主循环 |
| `compute_factor()` | 新增私有方法，计算FP16因子 |
| `pack_codes_from_file()` | 新增私有方法，ARM64打包量化码 |
| `fast_scan()` (ARM64重载) | 参数类型改为`float16_t`（query、data），新增 `low_dist_scale` 参数 |
| `fast_scan_mask()` | 新增静态方法，使用NEON掩码过滤的快速扫描 |

### Shell脚本

* 原始的script/index.sh和script/search.sh使用硬编码参数和G++编译。

  ```bash
  ./script/index.sh     # 硬编码sift, K=4096, G++编译
  ./script/search.sh    # 硬编码sift, G++编译, 单进程
  ```

* 优化后新增run.sh作为统一入口。

  * 支持sift/deep/glove/fashion/gist数据集。
  * 自动检测架构（x86_64/aarch64）选择编译参数。
  * ARM64 下使用clang++并链接jemalloc。
  * 多NUMA节点并发搜索。

  ```bash
  ./run.sh <dataset> [fastscan|scan] [generate|index|search|all]
  ```

## 非等价索引优化接口变动

非等价优化包含等价优化的全部接口变动，并在此基础上新增以下变更。

### Python脚本

| 脚本名称 | 脚本说明 | 使用方法 |
| ------ | ------ | ------- |
| `data/eval.py` | 全新脚本：负责ML模型训练与导出。 | `python eval.py <hdf5_path> <dataset> <K> <metric_type> <data_path> <BB>` |
| `data/test.py` | 全新脚本：向量归一化测试工具。 |- |

### C++命令行参数

在等价优化基础上，search程序新增：

| 参数 | 说明 |
| ------ | ------ |
| `-t` (threshold) | ML模型预测阈值，大于0时启用自适应nprobe。 |
| `-e` (pred\_nprobe) | 预测的缩减nprobe值。 |
| `-a` (soar\_lambda) | SOAR 参数，大于0时启用溢出簇搜索。 |

### C++ IVFRN类

| 变更 | 说明 |
| ------ | ------ |
| `IVFRN()` 构造函数 (ARM64) | 扩展为接受SOAR参数：`IVFRN(..., dist_to_spilled_labels*, x0_spilled*, spilled_labels*, binary_spilled*)`，传入溢出数据时自动启用SOAR。 |
| `search()` (ARM64) | 签名扩展为 `search(query, rd_query, k, nprobe, soar_lambda, threshold, pred_nprobe, pred_func, distK)`，支持SOAR搜索和ML自适应nprobe。 |
| `bool use_soar` | 新增成员变量，控制是否启用SOAR双簇搜索。 |
| `fast_scan_soar()` | 新增静态方法，带去重的SOAR快速扫描，参数包含 `unordered_set<uint32_t> &seen` 。 |
| `search_fast_scan<bool soar>()` | 新增模板私有方法，通过模板参数控制是否扫描溢出簇。 |
| `PredictFunc` | 新增类型别名 `void (*)(Entry* data, int pred_margin, double* result)`，用于ML模型推理回调。 |
| 溢出数据成员 | 新增`start_spilled`, `len_spilled`, `id_spilled`, `binary_code_spilled`, `fac_f16_spilled`, `data_f16_spilled`等成员。 |

### Shell脚本

`run.sh`在等价优化的基础上新增train和eval两个阶段：

- `train`：编译并运行`search_model`程序，生成每个查询的最优nprobe数据（`approximateGT.bin`、`expectedNprobe1.bin`）。
- `eval`：运行`eval.py`，训练 LightGBM 模型并导出为共享库（`libadaptivemodel_less.so`）。

```text
./run.sh <dataset> [fastscan|scan] [generate|index|search|train|eval|all]
```
