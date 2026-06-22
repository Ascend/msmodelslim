# 模型级 MSE（mse_model_wise）：敏感层分析算法说明

## 简介

- **概述**：`mse_model_wise` 用于**layer**范围分析：逐层对比仅该层相关子结构量化前后的**模型最终输出**（通常为最后一层 Decoder 输出），计算 MSE（均方误差，Mean Squared Error），得到层敏感度排序，用于整层或整块回退。
- **核心思想**：最终输出误差刻画该层量化对端到端行为的累计影响；依赖逐层链式前向，将上一层输出作为下一层输入以模拟真实推理路径。

## 使用前准备

安装 msModelSlim 工具，详情请参见《[msModelSlim工具安装指南](../../../install_guide/install_guide.md)》。

## 原理

1. 对同一批校准数据，在链式前向中依次评估各 Decoder 层：对该层内由 `quant_modules` 选中的结构做量化前后对比，采集**模型最终输出**。
2. 以最终输出的 MSE 作为该层的敏感度分数；分数越大表示该层对量化越敏感（即该层量化对模型最终输出影响越大）。
3. **链式约束**：若在某层出现层间张量形状或语义无法对齐，实现会从该层起跳过后续层并打印 warning，排名仅包含成功完成的层（常见于 MTP 等结构）。

## 适用要求

- **推荐场景**：希望通过**模型最终输出**视角评估各层量化敏感度，辅助整层或整块回退决策。
- **资源与数据**：校准批次数与序列长度会显著增加前向次数与中间缓存，大模型上可能 **OOM**，建议控制校准规模。
- **模型适配**：无需额外分析接口；`model_type` 支持范围与 ModelslimV1 量化一致，参见《[大模型支持矩阵](../../model_support/foundation_model_support_matrix.md)》。部分架构上受链式前向对齐限制，详见下文[FAQ](#faq)。

## 功能介绍

### 命令行示例

```bash
msmodelslim analyze layer \
    --model_type Qwen3-32B \
    --model_path ${model_path} \
    --metrics mse_model_wise \
    --quant_modules "*mlp*" \
    --calib_dataset ${calib_dataset} \
    --topk 15 \
    --device npu
```

### 命令行参数说明

| 参数 | 说明 |
|------|------|
| `layer` | Decoder 块级敏感度分析 |
| `--metrics` | 指定分析算法，取值为 `mse_model_wise` 时使用本算法 |
| `--quant_modules` | 通配符列表，指定参与量化对比的模块范围 |

完整参数见[敏感层分析工具使用指南参数说明](../../feature_guide/sensitive_layer_analysis/usage.md#34-参数说明)。

## FAQ

### 分析中途出现 warning 且后续层未出现在结果中？

**现象**: 日志出现 warning，且后续 Decoder 层未出现在分析结果中。

**解决方案**: 多为链式前向在某层无法对齐输入输出；可查阅日志定位层号，对该层或特殊结构（如 MTP）优先回退或换用 `mse_layer_wise` 做块级评估。

### 与 `mse_layer_wise` 的主要差异？

**现象**: 需要在 `mse_layer_wise` 与 `mse_model_wise` 之间选择，不清楚二者度量视角差异。

**解决方案**: `mse_layer_wise` 关注**单块输出**上的局部重构误差，反映该块自身的量化影响；`mse_model_wise` 关注**模型最终输出**上的全局累积误差，更贴近端到端效果，但计算与对齐要求更高。
