# API参考

本文档详细说明等价索引优化和非等价索引优化相对于原始 RaBitQ 开源代码的接口变动。

---

## 等价索引优化接口变动

### Python 脚本

原始 RaBitQ 的 Python 脚本使用硬编码参数，优化后改为命令行参数驱动，并支持 HDF5 数据格式和多种度量类型。

| 脚本 | 原始接口 | 优化后接口 |
|------|---------|-----------|
| `data/ivf.py` | 无命令行参数，硬编码 `dataset='sift'`, `K=4096`，从 fvecs 读取数据 | `python ivf.py <hdf5_path> <dataset> <K> <metric_type> <soar_lambda>`，支持 HDF5 输入、dot\_product 度量和向量归一化 |
| `data/rabitq.py` | 无命令行参数，硬编码 `datasets=['sift']`, `C=4096`，从 fvecs 读取数据 | `python rabitq.py <hdf5_path> <dataset> <K> <metric_type> <soar_lambda>`，支持 HDF5 输入；当 `soar_lambda > 0` 时额外生成溢出向量的量化数据 |

新增函数：
- `ivf.py::assign_spilled_vec_fast(X, C, labels, lam, batch)` — SOAR 溢出向量分配
- `ivf.py::test_ip(hdf5_path, dataset_name, K_value, metric_type, soar_lambda)` — 主入口函数

### C++ 命令行参数

**`index` 程序：**

| 参数 | 原始 | 优化后 |
|------|------|--------|
| `-d` (dataset) | 有 | 有 |
| `-s` (source) | 有 | 有 |
| `-p` (data\_path) | — | 新增，HDF5 数据文件路径 |
| `-m` (metric\_type) | — | 新增，度量类型（squared\_l2 / dot\_product） |
| `-a` (soar\_lambda) | — | 新增，SOAR 参数（设为 0 禁用） |

数据加载方式从直接读取 fvecs 改为通过 `loadHDFBase()` 读取 HDF5 格式。

**`search` 程序：**

| 参数 | 原始 | 优化后 |
|------|------|--------|
| `-d` (dataset) | 有 | 有 |
| `-s` (source) | 有 | 有 |
| `-r` (result\_path) | 有 | 有 |
| `-k` (topK) | 有 | 有 |
| `-n` (nprobe) | — | 新增，探查簇数量 |
| `-p` (data\_path) | — | 新增，HDF5 数据文件路径 |
| `-m` (metric\_type) | — | 新增，度量类型 |

数据加载方式从 fvecs/ivecs 改为通过 `loadHDF()` 读取 HDF5 格式。搜索框架从单线程改为多线程（pthread 绑核 + OpenMP），支持多 NUMA 节点并发。

### C++ IVFRN 类

`IVFRN` 类的公有接口签名保持不变，ARM64 平台下内部新增以下私有成员和方法：

| 变更 | 说明 |
|------|------|
| `Factor` 结构 (ARM64) | 批量布局：`float sqr_x[4]` 等，替代原始逐元素 `float sqr_x` |
| `Factor_f16` 结构 | 新增 FP16 批量因子结构（16 个元素一组） |
| `search_fast_scan()` | 新增私有方法，ARM64 NEON 优化的搜索主循环 |
| `compute_factor()` | 新增私有方法，计算 FP16 因子 |
| `pack_codes_from_file()` | 新增私有方法，ARM64 打包量化码 |
| `fast_scan()` (ARM64 重载) | 参数类型改为 `float16_t`（query、data），新增 `low_dist_scale` 参数 |
| `fast_scan_mask()` | 新增静态方法，使用 NEON 掩码过滤的快速扫描 |

### Shell 脚本

原始的 `script/index.sh` 和 `script/search.sh` 使用硬编码参数和 g++ 编译。优化后新增 `run.sh` 作为统一入口：

```
# 原始
./script/index.sh     # 硬编码 sift, K=4096, g++ 编译
./script/search.sh    # 硬编码 sift, g++ 编译, 单进程

# 优化后
./run.sh <dataset> [fastscan|scan] [generate|index|search|all]
# 支持 sift/deep/glove/fashion/gist 数据集
# 自动检测架构（x86_64/aarch64）选择编译参数
# ARM64 下使用 clang++ 并链接 jemalloc
# 多 NUMA 节点并发搜索
```

---

## 非等价索引优化接口变动

非等价优化包含等价优化的全部接口变动，并在此基础上新增以下变更：

### Python 脚本

| 脚本 | 新增内容 |
|------|---------|
| `data/eval.py` | 全新脚本：`python eval.py <hdf5_path> <dataset> <K> <metric_type> <data_path> <BB>`，负责 ML 模型训练与导出 |
| `data/test.py` | 全新脚本：向量归一化测试工具 |

### C++ 命令行参数

在等价优化基础上，`search` 程序新增：

| 参数 | 说明 |
|------|------|
| `-t` (threshold) | ML 模型预测阈值，大于 0 时启用自适应 nprobe |
| `-e` (pred\_nprobe) | 预测的缩减 nprobe 值 |
| `-a` (soar\_lambda) | SOAR 参数，大于 0 时启用溢出簇搜索 |

### C++ IVFRN 类

| 变更 | 说明 |
|------|------|
| `IVFRN()` 构造函数 (ARM64) | 扩展为接受 SOAR 参数：`IVFRN(..., dist_to_spilled_labels*, x0_spilled*, spilled_labels*, binary_spilled*)`，传入溢出数据时自动启用 SOAR |
| `search()` (ARM64) | 签名扩展为 `search(query, rd_query, k, nprobe, soar_lambda, threshold, pred_nprobe, pred_func, distK)`，支持 SOAR 搜索和 ML 自适应 nprobe |
| `bool use_soar` | 新增成员变量，控制是否启用 SOAR 双簇搜索 |
| `fast_scan_soar()` | 新增静态方法，带去重的 SOAR 快速扫描，参数包含 `unordered_set<uint32_t> &seen` |
| `search_fast_scan<bool soar>()` | 新增模板私有方法，通过模板参数控制是否扫描溢出簇 |
| `PredictFunc` | 新增类型别名 `void (*)(Entry* data, int pred_margin, double* result)`，用于 ML 模型推理回调 |
| 溢出数据成员 | 新增 `start_spilled`, `len_spilled`, `id_spilled`, `binary_code_spilled`, `fac_f16_spilled`, `data_f16_spilled` 等成员 |

### Shell 脚本

`run.sh` 在等价优化的基础上新增两个阶段：

```
./run.sh <dataset> [fastscan|scan] [generate|index|search|train|eval|all]
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    新增 train 和 eval 阶段
```

- `train`：编译并运行 `search_model` 程序，生成每个查询的最优 nprobe 数据（`approximateGT.bin`、`expectedNprobe1.bin`）
- `eval`：运行 `eval.py`，训练 LightGBM 模型并导出为共享库（`libadaptivemodel_less.so`）
