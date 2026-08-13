# 量化基础 量化术语百科词条

<!-- waiver: G01 原因：本文件为量化基础目录的索引文档，不适用 term_<english_name>.md 词条命名 -->
<!-- waiver: S01 原因：索引/类别文档采用目录结构（术语列表 / 模式清单），不适用词条模板的「2. 词条介绍」必填章节，依 01 清单 S03 组织自身结构 -->

> **英文名称**：Quantization Basics
> **应用领域**：大语言模型量化压缩、推理加速、KVCache 压缩

---

## 一句话定位

本目录独立成篇，集中介绍量化领域的基础术语，是阅读[量化模式](../quantization_mode/README.md)等知识词条的前置知识，按术语拆分为以下词条：

| 术语 | 词条 | 内容 |
|------|------|------|
| 量化 / 反量化 | [量化与反量化](term_quantization.md) | 量化/反量化公式、scale/zero_point、静态/动态、量化粒度、对称性 |
| 数据类型 FP16 / BF16 | [数据类型：FP16 / BF16](term_fp16_bf16.md) | 16位浮点格式，未量化基准 |
| 数据类型 INT8 | [数据类型：INT8](term_int8.md) | 8位整数格式 |
| 数据类型 INT4 | [数据类型：INT4](term_int4.md) | 4位整数格式 |
| 数据类型 FP8 | [数据类型：FP8（E4M3）](term_fp8.md) | 8位浮点格式 |
| 数据类型 MXFP8 / MXFP4 | [数据类型：MXFP8 / MXFP4](term_mxfp.md) | 块级共享指数格式 |

建议按顺序阅读：先读[量化与反量化](term_quantization.md)理解量化机制，再按需查阅各数据类型词条；量化带来的计算收益（整数 GEMM）在[量化模式](../quantization_mode/README.md)中展开。

---

## WxAy 记法

模式名常用 `WxAy` 记法（如 W8A8、W4A4）表示[线性层量化](../quantization_mode/linear_layer_quantization/README.md)中**权重（W）与激活（A）的位宽组合**，其中 `y` 是可选后缀，表示**数据类型或参数获取方式**（如 W8A8 静态、W4A4 MX 动态）。完整说明见[量化模式](../quantization_mode/README.md)「如何理解一个量化模式」一节。

---

## 关联流程

- 《[一键量化完整指南](../../user_guide/usage_quick_quantization.md)》：量化任务的执行入口。
- 《[量化精度调优指南](../../user_guide/process_quantization_precision_tuning.md)》：数据类型与量化方案导致的精度问题可通过该流程排查与调优。

---

## 关联词条

- [量化模式](../quantization_mode/README.md)：目标阅读对象，本目录词条是其前置基础知识。
- [线性层量化](../quantization_mode/linear_layer_quantization/README.md)：下位概念，WxAy 记法与 GEMM 的具体应用。
- [KVCache 量化](../quantization_mode/kv_cache_quantization/README.md)：下位概念，量化在缓存 K/V 张量上的应用。
- [FA 量化](../quantization_mode/fa_quantization/README.md)：下位概念，量化在注意力矩阵运算上的应用。

---

## 参考资料

1. 《[量化模式](../quantization_mode/README.md)》
