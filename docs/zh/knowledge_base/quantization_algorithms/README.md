---
toc_depth: 3
---
# 量化算法总览

msModelSlim 支持多种先进的量化算法，涵盖了从离群值抑制到低比特优化的各个环节。下表按类别总结了目前支持的核心算法及其主要特性。

## 离群值抑制算法

离群值抑制算法旨在平滑激活值的分布，减少量化带来的精度损失。

| 算法名称 | 核心思想 | 适用场景 | 详细说明 |
| :--- | :--- | :--- | :--- |
| **QuaRot** | 应用正交旋转矩阵平滑激活值分布 | 抑制激活离群值，提升精度 | [查看详情](quarot/quarot.md) |
| **Adapt Rotation** | 在QuaRot基础上使用基于校准数据迭代优化 Hadamard 旋转矩阵 | 优化旋转矩阵，进一步提升低比特量化精度 | [查看详情](adapt_rotation/adapt_rotation.md) |
| **SmoothQuant** | 协同缩放激活与权重，平滑离群值 | 抑制激活离群值 | [查看详情](smooth_quant/smooth_quant.md) |
| **Iterative Smooth** | 迭代式平滑缩放，更精细的分布调整 | 复杂分布下的精度优化 | [查看详情](iterative_smooth/iterative_smooth.md) |
| **Flex Smooth Quant** | 二阶段网格搜索自动寻找最优 alpha/beta | 灵活适配不同架构 | [查看详情](flex_smooth_quant/flex_smooth_quant.md) |
| **Flex AWQ SSZ** | 结合 AWQ 与 SSZ，使用真实量化器评估误差 | 自动搜索最优平滑参数 | [查看详情](flex_awq_ssz/flex_awq_ssz.md) |
| **KV Smooth** | 针对 KV Cache 的平滑抑制算法 | 降低 KV Cache 显存占用 | [查看详情](kv_smooth/kv_smooth.md) |
|  **AWQ** | 基于激活值统计特征网格搜索最优缩放因子 | 自动搜索最优平滑参数 | [查看详情](awq_smooth/awq_smooth.md) |

## 量化算法

包含权重量化、激活量化以及针对特定结构的量化方案。

| 算法名称 | 类型 | 核心思想 | 适用场景 | 详细说明 |
| :--- | :--- | :--- | :--- | :--- |
| **AutoRound** | 权重量化优化 | 基于 SignSGD 优化舍入偏移，降低重构误差 | 4bit 等超低比特量化 | [查看详情](autoround/autoround.md) |
| **FA3 Quant** | 激活量化 | 针对 Attention 激活的 per-head INT8 量化 | 长序列、MLA 架构模型 | [查看详情](fa3_quant/fa3_quant.md) |
| **GPTQ** | 权重量化优化 | 通过逐列优化和误差补偿最小化量化误差 | 高精度权重量化需求 | [查看详情](gptq/gptq.md) |
| **KVCache Quant** | KV Cache 量化 | 针对 KV Cache 的量化方案 | 提升长序列推理效率 | [查看详情](kvcache_quant/kvcache_quant.md) |
| **Linear Quant** | 基础量化 | 对线性层进行权重量化和激活量化 | 基础量化场景 | [查看详情](linear_quant/linear_quant.md) |
| **PDMIX** | 混合阶段量化 | Prefilling 使用动态量化，Decoding 使用静态量化 | 大模型推理加速，平衡精度与性能 | [查看详情](pdmix/pdmix.md) |
| **Histogram** | 激活量化 | 分析直方图分布，搜索最优截断区间 | 过滤离群值，提高精度 | [查看详情](histogram_activation_quantization/histogram_activation_quantization.md) |
| **MinMax** | 基础量化 | 统计最大最小值确定量化范围 | 基础量化场景，计算开销低 | [查看详情](minmax/minmax.md) |
| **SSZ** | 权重量化 | 迭代搜索最优缩放因子和偏移量 | 权重分布不均的精度优化 | [查看详情](ssz/ssz.md) |
| **LAOS** | 低比特量化 | 针对 W4A4 等极低比特场景的优化 | 极致压缩需求 | [查看详情](laos/laos.md) |
| **Float Sparse** | 稀疏化 | 基于 ADMM 算法实现模型浮点 sparse | 高压缩率需求 | [查看详情](float_sparse/float_sparse.md) |
| **SVDQuant** | 综合方案 | 离群值迁移 + SVD 低秩残差 + 残差量化 | 扩散模型等低比特量化 | [查看详情](svdquant/svdquant.md) |
| **MSE_Round** | 权重量化 | 按 block 在 ceil/floor shared exponent 间按 MSE 择优 | MXFP8 权重量化精度优化 | [查看详情](mse_round/mse_round.md) |
| **FouroverSix** | 权重量化 | 自适应选择块缩放（Scale-to-6 / Scale-to-4） | mxFP4 量化误差优化 | [查看详情](fouroversix/fouroversix.md) |
| **Ceil_X** | 权重量化 | ceil + 可配置除数计算 shared exponent | MXFP4 大值截断抑制 | [查看详情](ceil_x/ceil_x.md) |
| **DualScale** | 权重量化 | 两级粒度递进缩放，缓解异常通道 | W4A4 等低比特场景 | [查看详情](dual_scale/dual_scale.md) |

## 敏感层分析算法

敏感层分析通过`msmodelslim analyze`在校准数据上度量各层或子结构对量化的敏感程度，得到排序结果以辅助回退与 YAML 调参。

| 算法名称 | 分析范围 | 核心思想 | 适用场景 | 详细说明 |
| :--- | :--- | :--- | :--- | :--- |
| **Std** | linear（线性层） | 用激活动态范围与标准差的比值刻画敏感度 | 量化前线性层粗筛、默认策略之一 | [查看详情](std/std.md) |
| **Quantile** | linear（线性层） | 基于分位数与 IQR 构造 score，对离群点相对稳健 | 激活尾部重、希望降低离群主导 | [查看详情](quantile/quantile.md) |
| **Kurtosis** | linear（线性层） | 估计激活峰度，识别尖峰与极端值影响 | 关注尖峰分布、配合回退或混精 | [查看详情](kurtosis/kurtosis.md) |
| **Attention MSE（mse）** | attn（attention 结构） | 浮点与量化权重下 attention 输出的 MSE | Attention 权重量化敏感度（需适配器接口） | [查看详情](attention_mse/attention_mse.md) |
| **层级 MSE（mse_layer_wise）** | layer（Decoder 块） | 块内选中子模块输出上 MSE 的块内均值 | 整层或整块（如 MLP / attention 段）回退 | [查看详情](mse_layer_wise/mse_layer_wise.md) |
| **模型级 MSE（mse_model_wise）** | layer（链式前向） | 逐层量化扰动对**模型最终输出**的 MSE | 从最终隐藏状态视角看层敏感度 | [查看详情](mse_model_wise/mse_model_wise.md) |

## 算法选择建议

- **初学者**：建议优先使用《[一键量化 (V1)](../../user_guide/usage_quick_quantization.md)》，它会自动集成合适的算法组合。
- **敏感层与回退**：在定稿 YAML 前可按 scope 选用《[线性层敏感层分析使用指南](../../user_guide/usage_sensitive_linear_analysis.md)》、《[层级敏感层分析使用指南](../../user_guide/usage_sensitive_layer_wise_analysis.md)》或《[Attention 敏感层分析使用指南](../../user_guide/usage_sensitive_attn_analysis.md)》，结合上表 metrics 做层/结构排序；`linear`可首选**Kurtosis**，`layer`可优先**mse_layer_wise**。
- **自动调优**：精度不达标且希望自动搜索配置时，参见《[自动调优策略总览](../tuning_strategies/README.md)》。
- **追求极致精度**：可以尝试组合使用 **QuaRot** + **AutoRound**。
- **长序列推理**：推荐开启 **FA3 Quant** 和 **KVCache Quant**。
