# 层级 MSE（mse_layer_wise）：敏感层分析算法说明

## 简介

- **概述**：`mse_layer_wise`用于**layer**范围分析：在每个**Decoder 块**内，对由`quant_modules`选中的子模块（实现上主要为块内 Linear）分别做浮点与量化前向，在子模块输出上计算 MSE 并在**块内取均值**得到敏感度分数，输出**Decoder 块粒度**排序，指导整层或整块（如整段 attention / MLP）回退。
- **核心思想**：块内聚合的量化重构误差越大，该块在当前量化子集下越敏感。

## 使用前准备

安装 msModelSlim 工具，详情请参见《[msModelSlim工具安装指南](../../getting_started/install_guide.md)》。

## 原理

1. 对同一批校准样本，按 Decoder 层遍历：在块内对命中配置的子模块采集**浮点**与**量化**前向输出。
2. 对每一路可对齐的输出计算 MSE；将该块内所有有效 MSE **取均值**作为该块的 score。
3. **解读**：score 越大表示该 Decoder 块在当前 `quant_modules` 配置下对量化越敏感。

## 适用要求

- **推荐场景**：希望从 **Decoder 块内子模块输出**角度比较各层敏感度、做整层/整块回退决策。
- **模型适配**：**无需**模型适配器额外实现分析接口；`model_type` 支持范围与 ModelslimV1 量化一致，参见《[使用指南](../../feature_guide/sensitive_layer_analysis/usage.md#参数说明)》。
- **配置影响**：`quant_modules` 决定参与对比量化的子模块集合，不同配置会得到不同排序。

## 功能介绍

### 命令行示例

```bash
msmodelslim analyze layer \
    --model_type Qwen3-32B \
    --model_path ${model_path} \
    --metrics mse_layer_wise \
    --quant_modules "*mlp*" \
    --calib_dataset ${calib_dataset} \
    --topk 15 \
    --device npu
```

`layer` scope 下 `--metrics` 默认值为 `mse_layer_wise`。

### 命令行参数说明

| 参数 | 说明 |
|------|------|
| `layer`（scope） | Decoder 块级敏感度分析 |
| `--metrics mse_layer_wise` | 指定本算法 |
| `--quant_modules` | 通配符列表，指定参与量化对比的模块范围 |

完整参数见《[量化敏感层分析工具使用指南](../../feature_guide/sensitive_layer_analysis/usage.md)》。

## FAQ

### 换 `quant_modules` 后层排序变了，是否正常？

**现象**: 调整 `quant_modules` 通配后，各层 Top-K 顺序与调整前不一致。  

**解决方案**: 属于预期；`quant_modules` 改变被量化对比的子结构后，块内 MSE 与各层相对名次会变，排序随之变化。请按目标量化方案固定一版配置后再解读；同一命令仍不稳定时再查校准顺序与随机性。
