---
toc_depth: 3
---
# 量化算法总览

msModelSlim 支持多种先进的量化算法，涵盖了从离群值抑制到低比特优化的各个环节。下表按类别总结了目前支持的核心算法及其主要特性。

## 离群值抑制算法

离群值抑制算法旨在平滑激活值的分布，减少量化带来的精度损失。

| 算法名称 | 核心思想 | 适用场景 | 详细说明 |
| :--- | :--- | :--- | :--- |
| **QuaRot** | 应用正交旋转矩阵平滑激活值分布 | 抑制激活离群值，提升精度 | [查看详情](outlier_suppression_algorithms/quarot.md) |
| **Adapt Rotation** | 在QuaRot基础上使用基于校准数据迭代优化 Hadamard 旋转矩阵 | 优化旋转矩阵，进一步提升低比特量化精度 | [查看详情](outlier_suppression_algorithms/adapt_rotation.md) |
| **SmoothQuant** | 协同缩放激活与权重，平滑离群值 | 抑制激活离群值 | [查看详情](outlier_suppression_algorithms/smooth_quant.md) |
| **Iterative Smooth** | 迭代式平滑缩放，更精细的分布调整 | 复杂分布下的精度优化 | [查看详情](outlier_suppression_algorithms/iterative_smooth.md) |
| **Flex Smooth Quant** | 二阶段网格搜索自动寻找最优 alpha/beta | 灵活适配不同架构 | [查看详情](outlier_suppression_algorithms/flex_smooth_quant.md) |
| **Flex AWQ SSZ** | 结合 AWQ 与 SSZ，使用真实量化器评估误差 | 自动搜索最优平滑参数 | [查看详情](outlier_suppression_algorithms/flex_awq_ssz.md) |
| **KV Smooth** | 针对 KV Cache 的平滑抑制算法 | 降低 KV Cache 显存占用 | [查看详情](outlier_suppression_algorithms/kv_smooth.md) |
|  **AWQ** | 基于激活值统计特征网格搜索最优缩放因子 | 自动搜索最优平滑参数 | [查看详情](outlier_suppression_algorithms/awq_smooth.md) |

## 量化算法

包含权重量化、激活量化以及针对特定结构的量化方案。

| 算法名称 | 类型 | 核心思想 | 适用场景 | 详细说明 |
| :--- | :--- | :--- | :--- | :--- |
| **AutoRound** | 权重量化优化 | 基于 SignSGD 优化舍入偏移，降低重构误差 | 4bit 等超低比特量化 | [查看详情](quantization_algorithms/autoround.md) |
| **FA3 Quant** | 激活量化 | 针对 Attention 激活的 per-head INT8 量化 | 长序列、MLA 架构模型 | [查看详情](quantization_algorithms/fa3_quant.md) |
| **GPTQ** | 权重量化优化 | 通过逐列优化和误差补偿最小化量化误差 | 高精度权重量化需求 | [查看详情](quantization_algorithms/gptq.md) |
| **KVCache Quant** | KV Cache 量化 | 针对 KV Cache 的量化方案 | 提升长序列推理效率 | [查看详情](quantization_algorithms/kvcache_quant.md) |
| **Linear Quant** | 基础量化 | 对线性层进行权重量化和激活量化 | 基础量化场景 | [查看详情](quantization_algorithms/linear_quant.md) |
| **PDMIX** | 混合阶段量化 | Prefilling 使用动态量化，Decoding 使用静态量化 | 大模型推理加速，平衡精度与性能 | [查看详情](quantization_algorithms/pdmix.md) |
| **Histogram** | 激活量化 | 分析直方图分布，搜索最优截断区间 | 过滤离群值，提高精度 | [查看详情](quantization_algorithms/histogram_activation_quantization.md) |
| **MinMax** | 基础量化 | 统计最大最小值确定量化范围 | 基础量化场景，计算开销低 | [查看详情](quantization_algorithms/minmax.md) |
| **SSZ** | 权重量化 | 迭代搜索最优缩放因子和偏移量 | 权重分布不均的精度优化 | [查看详情](quantization_algorithms/ssz.md) |
| **LAOS** | 低比特量化 | 针对 W4A4 等极低比特场景的优化 | 极致压缩需求 | [查看详情](quantization_algorithms/laos.md) |
| **Float Sparse** | 稀疏化 | 基于 ADMM 算法实现模型浮点 sparse | 高压缩率需求 | [查看详情](quantization_algorithms/float_sparse.md) |

## 自动调优策略

通过自动化策略寻找最优的量化配置。

| 算法名称 | 核心思想 | 适用场景 | 详细说明 |
| :--- | :--- | :--- | :--- |
| **Standing High** | 结合离群值策略，在满足精度条件下基于二分法尽量减少回退层数 | 需精细控制模板与策略，需要提供完整量化配置 | [查看详情](auto_tuning_strategies/standing_high.md) |
| **Standing High With Experience** | 仅需量化类型与结构配置，根据专家经验自动生成量化配置 | 熟悉模型结构，无需提供完整量化配置 | [查看详情](auto_tuning_strategies/standing_high_with_experience.md) |

## 敏感层分析算法

敏感层分析通过`msmodelslim analyze`在校准数据上度量各层或子结构对量化的敏感程度，得到排序结果以辅助回退与 YAML 调参。

| 算法名称 | 分析范围 | 核心思想 | 适用场景 | 详细说明 |
| :--- | :--- | :--- | :--- | :--- |
| **Std** | linear（线性层） | 用激活动态范围与标准差的比值刻画敏感度 | 量化前线性层粗筛、默认策略之一 | [查看详情](sensitive_layer_analysis/std.md) |
| **Quantile** | linear（线性层） | 基于分位数与 IQR 构造 score，对离群点相对稳健 | 激活尾部重、希望降低离群主导 | [查看详情](sensitive_layer_analysis/quantile.md) |
| **Kurtosis** | linear（线性层） | 估计激活峰度，识别尖峰与极端值影响 | 关注尖峰分布、配合回退或混精 | [查看详情](sensitive_layer_analysis/kurtosis.md) |
| **Attention MSE（mse）** | attn（attention 结构） | 浮点与量化权重下 attention 输出的 MSE | Attention 权重量化敏感度（需适配器接口） | [查看详情](sensitive_layer_analysis/attention_mse.md) |
| **层级 MSE（mse_layer_wise）** | layer（Decoder 块） | 块内选中子模块输出上 MSE 的块内均值 | 整层或整块（如 MLP / attention 段）回退 | [查看详情](sensitive_layer_analysis/mse_layer_wise.md) |
| **模型级 MSE（mse_model_wise）** | layer（链式前向） | 逐层量化扰动对**模型最终输出**的 MSE | 从最终隐藏状态视角看层敏感度 | [查看详情](sensitive_layer_analysis/mse_model_wise.md) |

## 算法选择建议

- **初学者**：建议优先使用《[一键量化 (V1)](../feature_guide/quick_quantization_v1/usage.md)》，它会自动集成合适的算法组合。
- **敏感层与回退**：在定稿 YAML 前可用《[敏感层分析](../feature_guide/sensitive_layer_analysis/usage.md)》结合上表 metrics 做层/结构排序；`linear`可首选**Kurtosis**，`layer`可优先**mse_layer_wise**。
- **追求极致精度**：可以尝试组合使用 **QuaRot** + **AutoRound**。
- **长序列推理**：推荐开启 **FA3 Quant** 和 **KVCache Quant**。
