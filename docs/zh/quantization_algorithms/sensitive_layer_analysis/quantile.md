# Quantile：敏感层分析算法说明

## 简介

- **概述**：Quantile（分位数）度量用于**linear**范围分析：基于激活的分位数与四分位距（IQR）构造 score，对离群点相对稳健，用于**线性层粒度**敏感度排序。
- **核心思想**：用 $Q_1$、$Q_3$ 描述主体分布宽度，结合绝对幅度构造分数；IQR 越大表示主体分布越分散，在相同动态范围下对量化相对更不敏感。

## 使用前准备

安装 msModelSlim 工具，详情请参见《[msModelSlim工具安装指南](../../getting_started/install_guide.md)》。

## 原理

1. 计算激活的第 1/4、第 3/4 分位数 $Q_1$、$Q_3$，以及用于幅度项的统计量。
2. 使用文档口径中的 score 形式（与实现一致）：`score = 2 * max_abs / 254 / (Q3 - Q1)`，基于 IQR 刻画分布特征。
3. **解读**：在既定公式下，IQR 越大 score 越小（主体越分散）；绝对值越大 score 越大。具体排序用于相对比较敏感层。

## 适用要求

- **推荐场景**：激活分布尾部较重、希望降低离群点对单层分数主导的场景。
- **模型适配**：**无需**模型适配器额外实现分析接口。

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
| `linear`（scope） | 线性层敏感度分析 |
| `--metrics quantile` | 指定本算法 |

完整参数见《[量化敏感层分析工具使用指南](../../feature_guide/sensitive_layer_analysis/usage.md)》。
