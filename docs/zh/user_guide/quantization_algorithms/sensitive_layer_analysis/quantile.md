# Quantile：敏感层分析算法说明

## 简介

- **概述**：Quantile（分位数）度量用于**linear**范围分析：基于激活的分位数与四分位距（IQR）构造 score，对离群点相对稳健，用于**线性层粒度**敏感度排序。
- **核心思想**：用下四分位数 $Q_1$（第 1/4 分位数）、上四分位数 $Q_3$（第 3/4 分位数）描述激活分布的主体宽度，结合绝对幅度构造分数；四分位距 $\text{IQR} = Q_3 - Q_1$ 越大表示主体分布越分散，在相同动态范围下对量化相对更不敏感。

## 使用前准备

安装 msModelSlim 工具，详情请参见《[msModelSlim工具安装指南](../../../install_guide/install_guide.md)》。

## 原理

1. 计算激活的第 1/4、第 3/4 分位数 $Q_1$、$Q_3$，以及用于幅度项的统计量。
2. 计算 score：`score = 2 × max(|max_value|, |min_value|) / 254 / (Q3 - Q1)`，其中分子 `max(|max_value|, |min_value|)` 为激活绝对值的最大值，分母 `Q3 - Q1` 即四分位距（IQR）。
3. **解读**：score 越大表示该层对量化越敏感。具体地：
   - 激活绝对值的最大值越大 —— 量化步长相对越大，量化误差越显著，score 越大；
   - IQR 越大 —— 激活主体分布越分散，相同动态范围下每个量化区间的代表值越多，相对误差越小，score 越小。

## 适用要求

- **推荐场景**：激活分布尾部较重，希望降低离群点对单层分数的主导影响，使敏感度排序更稳健。
- **模型适配**：无需模型适配器额外实现分析接口。`model_type` 支持范围参见参见《[大模型支持矩阵](../../model_support/foundation_model_support_matrix.md)》。

## 功能介绍

### 命令行示例

```bash
msmodelslim analyze linear \
    --model_type Qwen2.5-7B-Instruct \
    --model_path ${model_path} \
    --metrics quantile \
    --calib_dataset ${calib_dataset} \
    --pattern "*.down_proj*" "*.o_proj*" \
    --topk 15 \
    --device npu
```

### 命令行参数说明

| 参数 | 说明 |
|------|------|
| `linear` | 线性层敏感度分析 |
| `--metrics` | 指定分析算法，取值为 `quantile` 时使用本算法 |
| `--pattern` | 层名通配符，过滤待分析线性层 |

完整参数见[敏感层分析工具使用指南参数说明](../../feature_guide/sensitive_layer_analysis/usage.md#34-参数说明)。
