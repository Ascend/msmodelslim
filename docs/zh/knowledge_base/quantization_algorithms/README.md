---
toc_depth: 3
---
# 量化算法总览

msModelSlim 支持多种先进的量化算法，涵盖了从离群值抑制到低比特优化的各个环节。下表按类别总结了目前支持的核心算法及其主要特性。

## 离群值抑制算法

离群值抑制算法旨在平滑激活值的分布，减少量化带来的精度损失。

| 算法名称 | 核心思想 | 适用场景 | 词条 | 使用指南 |
| :--- | :--- | :--- | :--- | :--- |
| **QuaRot** | 应用正交旋转矩阵平滑激活值分布 | 抑制激活离群值，提升精度 | [词条](quarot/term_quarot.md) | [使用指南](quarot/usage_quarot.md) |
| **Adapt Rotation** | 在QuaRot基础上使用基于校准数据迭代优化 Hadamard 旋转矩阵 | 优化旋转矩阵，进一步提升低比特量化精度 | [词条](adapt_rotation/term_adapt_rotation.md) | [使用指南](adapt_rotation/usage_adapt_rotation.md) |
| **SmoothQuant** | 协同缩放激活与权重，平滑离群值 | 抑制激活离群值 | [词条](smooth_quant/term_smooth_quant.md) | [使用指南](smooth_quant/usage_smooth_quant.md) |
| **Iterative Smooth** | 迭代式平滑缩放，更精细的分布调整 | 复杂分布下的精度优化 | [词条](iterative_smooth/term_iterative_smooth.md) | [使用指南](iterative_smooth/usage_iterative_smooth.md) |
| **Flex Smooth Quant** | 二阶段网格搜索自动寻找最优 alpha/beta | 灵活适配不同架构 | [词条](flex_smooth_quant/term_flex_smooth_quant.md) | [使用指南](flex_smooth_quant/usage_flex_smooth_quant.md) |
| **Flex AWQ SSZ** | 结合 AWQ 与 SSZ，使用真实量化器评估误差 | 自动搜索最优平滑参数 | [词条](flex_awq_ssz/term_flex_awq_ssz.md) | [使用指南](flex_awq_ssz/usage_flex_awq_ssz.md) |
| **KV Smooth** | 针对 KV Cache 的平滑抑制算法 | 降低 KV Cache 显存占用 | [词条](kv_smooth/term_kv_smooth.md) | [使用指南](kv_smooth/usage_kv_smooth.md) |
|  **AWQ** | 基于激活值统计特征网格搜索最优缩放因子 | 自动搜索最优平滑参数 | [词条](awq_smooth/term_awq_smooth.md) | [使用指南](awq_smooth/usage_awq_smooth.md) |

## 量化算法

包含权重量化、激活量化以及针对特定结构的量化方案。

| 算法名称 | 类型 | 核心思想 | 适用场景 | 词条 | 使用指南 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **AutoRound** | 权重量化优化 | 基于 SignSGD 优化舍入偏移，降低重构误差 | 4bit 等超低比特量化 | [词条](autoround/term_autoround.md) | [使用指南](autoround/usage_autoround.md) |
| **FA3 Quant** | 激活量化 | 针对 Attention 激活的 per-head INT8 量化 | 长序列、MLA 架构模型 | [词条](fa3_quant/term_fa3_quant.md) | [使用指南](fa3_quant/usage_fa3_quant.md) |
| **GPTQ** | 权重量化优化 | 通过逐列优化和误差补偿最小化量化误差 | 高精度权重量化需求 | [词条](gptq/term_gptq.md) | [使用指南](gptq/usage_gptq.md) |
| **KVCache Quant** | KV Cache 量化 | 针对 KV Cache 的量化方案 | 提升长序列推理效率 | [词条](kvcache_quant/term_kvcache_quant.md) | [使用指南](kvcache_quant/usage_kvcache_quant.md) |
| **Linear Quant** | 基础量化 | 对线性层进行权重量化和激活量化 | 基础量化场景 | [词条](linear_quant/term_linear_quant.md) | [使用指南](linear_quant/usage_linear_quant.md) |
| **PDMIX** | 混合阶段量化 | Prefilling 使用动态量化，Decoding 使用静态量化 | 大模型推理加速，平衡精度与性能 | [词条](pdmix/term_pdmix.md) | [使用指南](pdmix/usage_pdmix.md) |
| **Histogram** | 激活量化 | 分析直方图分布，搜索最优截断区间 | 过滤离群值，提高精度 | [词条](histogram_activation_quantization/term_histogram_activation_quantization.md) | [使用指南](histogram_activation_quantization/usage_histogram_activation_quantization.md) |
| **MinMax** | 基础量化 | 统计最大最小值确定量化范围 | 基础量化场景，计算开销低 | [词条](minmax/term_minmax.md) | [使用指南](minmax/usage_minmax.md) |
| **SSZ** | 权重量化 | 迭代搜索最优缩放因子和偏移量 | 权重分布不均的精度优化 | [词条](ssz/term_ssz.md) | [使用指南](ssz/usage_ssz.md) |
| **LAOS** | 低比特量化 | 针对 W4A4 等极低比特场景的优化 | 极致压缩需求 | [词条](laos/term_laos.md) | [使用指南](laos/usage_laos.md) |
| **Float Sparse** | 稀疏化 | 基于 ADMM 算法实现模型浮点 sparse | 高压缩率需求 | [词条](float_sparse/term_float_sparse.md) | [使用指南](float_sparse/usage_float_sparse.md) |
| **SVDQuant** | 综合方案 | 离群值迁移 + SVD 低秩残差 + 残差量化 | 扩散模型等低比特量化 | [词条](svdquant/term_svdquant.md) | [使用指南](svdquant/usage_svdquant.md) |
| **MSE_Round** | 权重量化 | 按 block 在 ceil/floor shared exponent 间按 MSE 择优 | MXFP8 权重量化精度优化 | [词条](mse_round/term_mse_round.md) | [使用指南](mse_round/usage_mse_round.md) |
| **FouroverSix** | 权重量化 | 自适应选择块缩放（Scale-to-6 / Scale-to-4） | MXFP4 量化误差优化 | [词条](fouroversix/term_fouroversix.md) | [使用指南](fouroversix/usage_fouroversix.md) |
| **Ceil_X** | 权重量化 | ceil + 可配置除数计算 shared exponent | MXFP4 大值截断抑制 | [词条](ceil_x/term_ceil_x.md) | [使用指南](ceil_x/usage_ceil_x.md) |
| **DualScale** | 权重量化 | 两级粒度递进缩放，缓解异常通道 | W4A4 等低比特场景 | [词条](dual_scale/term_dual_scale.md) | [使用指南](dual_scale/usage_dual_scale.md) |

## 敏感层分析算法

敏感层分析通过`msmodelslim analyze`在校准数据上度量各层或子结构对量化的敏感程度，得到排序结果以辅助回退与 YAML 调参。

| 算法名称 | 分析范围 | 核心思想 | 适用场景 | 词条 | 使用指南 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Std** | linear（线性层） | 用激活动态范围与标准差的比值刻画敏感度 | 量化前线性层粗筛、默认策略之一 | [词条](std/term_std.md) | [使用指南](std/usage_std.md) |
| **Quantile** | linear（线性层） | 基于分位数与 IQR 构造 score，对离群点相对稳健 | 激活尾部重、希望降低离群主导 | [词条](quantile/term_quantile.md) | [使用指南](quantile/usage_quantile.md) |
| **Kurtosis** | linear（线性层） | 估计激活峰度，识别尖峰与极端值影响 | 关注尖峰分布、配合回退或混精 | [词条](kurtosis/term_kurtosis.md) | [使用指南](kurtosis/usage_kurtosis.md) |
| **Attention MSE（mse）** | attn（attention 结构） | 浮点与量化权重下 attention 输出的 MSE | Attention 权重量化敏感度（需适配器接口） | [词条](attention_mse/term_attention_mse.md) | [使用指南](attention_mse/usage_attention_mse.md) |
| **层级 MSE（mse_layer_wise）** | layer（Decoder 块） | 块内选中子模块输出上 MSE 的块内均值 | 整层或整块（如 MLP / attention 段）回退 | [词条](mse_layer_wise/term_mse_layer_wise.md) | [使用指南](mse_layer_wise/usage_mse_layer_wise.md) |
| **模型级 MSE（mse_model_wise）** | layer（链式前向） | 逐层量化扰动对**模型最终输出**的 MSE | 从最终隐藏状态视角看层敏感度 | [词条](mse_model_wise/term_mse_model_wise.md) | [使用指南](mse_model_wise/usage_mse_model_wise.md) |

## 算法选择建议

初学者可优先使用《[一键量化 (V1)](../../user_guide/usage_quick_quantization.md)》，自动集成已验证的算法组合。需要自动搜索配置时，参见《[自动调优策略总览](../tuning_strategies/README.md)》。实践配置亦可参考 `lab_practice/` 下对应 YAML。

### 量化算法

- **W8A8**：最常用 **MinMax**——统计最大最小值确定量化范围，计算开销低，适合作为默认起步方案。
- **W4A8**：权重侧用 **SSZ** 迭代搜索缩放因子与偏移，激活 A8 仍用 **MinMax**，二者配合使用。
- **W4A4 MXFP**：优先 **Ceil_X**，用 ceil + 可配置除数收紧 shared exponent，抑制 floor 缩放带来的大值截断。
- **W4A4（INT / MXFP）**：可选用基于训练的 **AutoRound** 进一步抬精度，INT 与 MXFP 均支持；但对算力要求更高，量化耗时通常成倍高于其他大多数算法，选用时需权衡资源与时延。
- **长序列 / C8**：产品上将 **KVCache Quant** 与 **FA3 Quant** 都纳入 C8。前者量化写入缓存的 Key/Value，专攻缩小 KV Cache、缓解推理显存压力；后者量化 Attention 路径上的 Q/K/V 激活以加速 attention 运算（仅支持 MLA）。二者机制不同，实践中一般只开其一，按显存或算力瓶颈选择。

### 离群值抑制算法

- 常用 **Flex Smooth Quant** 与 **QuaRot**：可独立使用，也可串联叠加。前者二阶段网格搜索 alpha / beta，适配面广；后者正交旋转平滑激活离群，精度收益往往更明显，但对模型适配要求更高。
- **Flex AWQ SSZ** 在 4bit 低精场景下效果较好，但搜索相对较慢，适合精度优先、可接受更长量化时间的场景。

### 敏感层分析

当前敏感层分析支持按不同范围（`linear` / `layer` / `attn`）度量敏感度，并据此做对应粒度的回退或混精调参。使用指南：《[线性层](../../user_guide/usage_sensitive_linear_analysis.md)》、《[层级](../../user_guide/usage_sensitive_layer_wise_analysis.md)》、《[Attention](../../user_guide/usage_sensitive_attn_analysis.md)》。

- **linear**（线性层）：首选 **Kurtosis**，用激活峰度刻画尖峰与尾部影响，辅助识别需回退或提位宽的线性层。
- **layer**（Decoder 块）：首选 **mse_layer_wise**，适合整层 / 整块（如 MLP、attention 段）回退。
- **attn**（Attention 结构）：首选 **Attention MSE（mse）**，主要用于配合 **FA3 Quant** 识别需回退的 Attention 模块（需适配器接口）。
