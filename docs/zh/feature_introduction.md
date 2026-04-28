# 特性指南

## 技术架构

```text
原始RaBitQ (x86_64 AVX2)
│
├── 0002-rabitq-optimize-eqv.patch # 等价优化版本（不改变算法语义）
│   ├── ARM64 NEON向量化
│   ├── FP16精度 + 批量Factor结构
│   ├── 汇编级LUT加速 + SIMD掩码过滤
│   ├── 自适应批量处理 (64/96)
│   ├── HDF5数据格式 + 多度量类型支持
│   └── 多线程搜索框架 (pthread + NUMA)
│
└── 0001-rabitq-optimize-neq.patch  # 非等价优化版本（改变搜索行为）
    ├── 包含等价优化全部内容 +
    ├── SOAR溢出向量分配（索引阶段，扩展索引结构）
    ├── 双簇搜索与去重（搜索阶段）
    └── ML自适应nprobe（搜索阶段，不改变索引结构）
        ├── 离线：LightGBM训练 → treelite导出 .so
        └── 在线：查询特征提取 → 模型推理 → 动态调整nprobe
```

## 等价索引优化（0002-rabitq-optimize-eqv.patch）

在不改变算法语义的前提下，针对ARM64架构进行性能优化，搜索结果与原始实现保持一致。

### 主要优化内容

- **FP16精度优化**：对质心、数据向量和因子结构引入`float16_t`存储，降低内存占用和带宽消耗，提升SIMD吞吐量。
- **NEON SIMD向量化**：使用ARM NEON指令替代x86 AVX2实现，覆盖距离计算、位转置、标量量化等核心路径。
- **汇编级LUT优化**：新增`krl_table_lookup_fast_scan.s`，使用ARM64汇编实现批量查找表（LUT）查找，优化指令调度与预取策略。
- **自适应批量处理**：根据向量维度自动选择批量大小（64/96），相比原始固定32的批量大小，提升缓存利用率。
- **SIMD掩码过滤**：使用NEON向量比较生成掩码，批量跳过不满足距离阈值的候选向量，减少分支预测开销。
- **内存对齐优化**：对关键数据结构使用64/256字节对齐，提升 NEON 访存效率。
- **Factor结构批量化**：将Factor结构改为批量布局（4/16个向量一组），便于SIMD并行处理。

### 变更文件

| 文件 | 变更类型 | 说明 |
| ------ | --------- | ------ |
| `src/ivf_rabitq.h` | 修改 | 添加ARM64数据结构与搜索流程 |
| `src/ivf_rabitq_search.h` | 新增 | ARM64优化的搜索实现 |
| `src/index_io.h` | 新增 | ARM64索引I/O（含FP16转换） |
| `src/space.h` | 修改 | NEON SIMD实现（位操作、距离计算） |
| `src/fast_scan.h` | 修改 | 自适应批量打包与NEON快速扫描 |
| `src/matrix.h` | 修改 | 矩阵操作适配 |
| `src/index.cpp` | 修改 | 索引构建流程适配 |
| `src/search.cpp` | 修改 | 搜索流程适配 |
| `src/test_result.cpp` | 修改 | 测试结果处理 |
| `data/ivf.py` | 修改 | IVF聚类流程优化，新增SOAR分配与点积度量支持 |
| `data/rabitq.py` | 修改 | 量化索引构建适配 |
| `run.sh` | 新增 | 一键运行脚本 |

## 非等价索引优化（0001-rabitq-optimize-neq.patch）

引入SOAR（Spilled Over Assignment with Residual）算法，打破传统IVF的单簇分配限制，允许向量同时属于原始聚类簇和溢出聚类簇，以牺牲少量内存和计算开销换取更高的召回率。

### 主要优化内容

- **SOAR溢出向量分配**：基于距离项与投影能量的联合损失函数，为每个向量计算最优的溢出聚类簇分配。
  - 损失函数：`loss = ||x - c'||² + λ * (r·r')² / ||r||²`，其中 `λ`（soar\_lambda）控制投影项权重。
  - 支持批量处理（默认 batch=8000），提升分配效率。
- **溢出数据结构**：扩展索引结构以存储溢出向量的二进制编码FP16数据、Factor因子等。
- **SOAR搜索**：搜索时同时扫描原始簇和溢出簇，使用 `unordered_set` 去重避免重复计算。
- **ARM64优化**：与等价优化共享FP16、NEON SIMD、汇编 LUT等底层优化。
- **ML自适应nprobe**：使用LightGBM模型在搜索阶段动态调整nprobe，详见下文。

### 变更文件

| 文件 | 变更类型 | 说明 |
| ------ | --------- | ------ |
| `src/ivf_rabitq.h` | 修改 | 添加SOAR数据结构与搜索入口 |
| `src/ivf_rabitq_search.h` | 新增 | SOAR搜索实现（含去重与双簇扫描） |
| `src/index_io.h` | 新增 | SOAR索引的持久化（含溢出数据） |
| `src/space.h` | 修改 | NEON SIMD实现 |
| `src/fast_scan.h` | 修改 | 批量扫描优化 |
| `src/index.cpp` | 修改 | 索引构建（含溢出数据） |
| `src/krl_table_lookup_fast_scan.s` | 新增 | ARM64汇编优化 |
| `data/ivf.py` | 修改 | IVF聚类 + SOAR溢出分配 |
| `data/rabitq.py` | 修改 | 量化索引构建（含溢出数据） |
| `data/eval.py` | 新增 | 特征提取与LightGBM评估 |
| `data/test.py` | 新增 | 测试脚本 |
| `run.sh` | 新增 | 一键运行脚本 |
| `script/index.sh` | 修改 | 索引构建脚本（含SOAR参数） |
| `script/search.sh` | 修改 | 搜索脚本（含SOAR参数） |

### ML自适应nprobe详解

非等价优化中除SOAR溢出向量分配外，还引入了基于机器学习的**查询自适应nprobe预测**机制。该机制**不改变索引结构**，仅在搜索阶段根据查询向量特征动态调整探查簇数（nprobe），为“简单”查询减少计算量，从而提升整体吞吐。

### 离线训练阶段（`data/eval.py`）

1. **数据准备**：加载基向量数据和预计算的每查询最优nprobe（`expectedNprobe1.bin`）。
2. **特征提取**：对每个向量提取四维特征——均值（mean）、标准差（std）、稀疏度（sparsity）、范数（norm）。
3. **标签构造**：以nprobe的中位数为阈值进行二分类，nprobe大于中位数标记为1（"困难"查询），否则为0（"简单"查询）。
4. **模型训练**：使用LightGBM（GBDT, binary objective, AUC metric）训练分类模型，通过Youden's Index选取最优分类阈值。
5. **模型导出**：通过treelite + tl2cgen将训练好的LightGBM模型编译为C共享库`libadaptivemodel_less.so`，供C++搜索程序直接调用。

### 在线搜索阶段（`src/ivf_rabitq_search.h`）

搜索时，当`threshold > 0`且`pred_nprobe > 0`时启用自适应nprobe。

1. 通过`dlopen()`加载`libadaptivemodel_less.so`，获取`predict`函数指针。
2. 对每个查询向量实时提取相同的四维特征（mean, std, sparsity, norm）。
3. 调用模型推理：`pred(fdata, 0, &result)`。
4. 若`result <= threshold`（预测为“简单”查询），将nprobe从默认值缩减为`pred_nprobe`。
5. 否则保持默认nprobe不变。

### 对索引结构的影响

ML自适应nprobe机制**完全不改变索引结构**。索引的构建、存储和加载流程与不使用ML时完全一致。ML模型仅作为搜索阶段的旁路预测器，通过减少“简单”查询的nprobe来提升QPS，在基本不损失召回率的前提下降低平均搜索开销。该机制可通过设置`threshold=0`完全禁用。
