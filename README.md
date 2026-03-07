# RaBitQ介绍

## 最新消息

- 2026.03.30：RaBitQ优化补丁发布于Gitcode平台，实现等价索引优化和非等价索引优化。

## 项目介绍

RaBitQ是由NTU团队提出的面向高维向量近似最近邻搜索（ANN）的随机二值量化方法（SIGMOD 2024），能将D维向量量化为D位二进制字符串，并提供理论误差界。原始实现基于x8664 AVX2指令集。鲲鹏优化基于开源RaBitQ代码做侵入式修改，将其扩展至ARM64（AArch64）架构，引入FP16精度优化、NEON SIMD向量化、汇编级LUT加速、SOAR溢出向量分配、ML自适应nprobe等多项性能优化和功能增强。

## 目录结构

代码仓目录结构如下：

```text
rabitq/
├─ 0001-rabitq-optimize-neq.patch         // 非等价索引优化补丁（全量优化）
├─ 0002-rabitq-optimize-eqv.patch         // 等价索引优化补丁
```

使用补丁后RaBitQ完整的目录结构如下所示：

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

## 版本说明

关于RaBitQ的版本更新情况请参见《[RaBitQ版本说明书](docs/版本说明书.md)》。

## 学习文档


| 学习资源类别 | 学习资源名称                        | 学习资源简介                                                    |
| ------ | ----------------------------- | --------------------------------------------------------- |
| 文档     | [快速入门](docs/快速入门.md)         | 提供概述、前置条件、补丁应用方法和基本使用指导。                           |
| 文档     | [特性指南](docs/特性指南.md)     | 详细说明等价索引优化和非等价索引优化的技术内容，包括SOAR算法、ML自适应nprobe机制及技术架构。      |
| 文档     | [API参考](docs/API参考.md)     | 对比原始RaBitQ开源代码，详细列出Python脚本、C++命令行、IVFRN类和Shell脚本的全部接口变动。 |
| 文档     | [用户指南](docs/用户指南.md) | 提供run.sh测试脚本的详细使用方法，包括参数说明、数据集配置、搜索参数、环境配置和使用示例。          |


## 免责声明

此代码仓计划参与RaBitQ开源组件，编码风格遵照原生开源软件，继承原生开源软件安全设计，不破坏原生开源软件设计及编码风格和方式，软件的任何漏洞与安全问题，均由相应的上游社区根据其漏洞和安全响应机制解决。请密切关注上游社区发布的通知和版本更新。鲲鹏计算社区对软件的漏洞及安全问题不承担任何责任。

## License

本项目采用Apache License 2.0许可证授权，详见[LICENSE](LICENSE)文件

本项目的文档适用CC-BY 4.0许可证，具体请参见文件[LICENSE](docs/LICENSE)文件

## 贡献声明

欢迎大家为社区做贡献，如果使用过程中有任何问题/建议，或者需要反馈特性需求和bug报告，可以提交[Issues](https://gitcode.com/boostkit/community/blob/master/docs/contributor/issue-submit.md)联系我们，具体贡献方法可参考[这里](https://gitcode.com/boostkit/community/blob/master/docs/contributor/contributing.md)。同时也欢迎大家在[讨论专区](https://gitcode.com/boostkit/community/discussions)展开讨论交流。感谢您的支持。

## 致谢

RaBitQ由华为公司的下列部门联合贡献：

- 鲲鹏计算Boostkit开发部

感谢来自社区的每一个PR，欢迎贡献RaBitQ！
