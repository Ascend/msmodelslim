# MinMax：最小最大值量化算法说明

## 简介

- **概述**：MinMax 是一种最基础且最常用的量化算法。它通过统计张量（权重或激活值）中的最小值和最大值来确定量化范围，从而计算量化缩放因子（scale）和偏移量（offset）。
- **核心思想**：将原始浮点数范围线性映射到目标数值范围（如 INT8 的 [-128, 127] 或 FP8 的表示范围）。该算法简单高效，计算开销极低，是大多数常规量化场景的首选。

## 使用前准备

安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../../install_guide/install_guide.md)。

## 原理和实现

### 原理

MinMax 算法基于以下公式计算量化参数：

1. **确定范围**：
   - $V_{min} = \min(X)$
   - $V_{max} = \max(X)$

2. **计算缩放因子 (Scale)**：
   - 对于非对称量化：$S = \frac{V_{max} - V_{min}}{Q_{max} - Q_{min}}$
   - 对于对称量化：$S = \frac{\max(|V_{min}|, |V_{max}|)}{Q_{max}}$

3. **计算偏移量 (Offset)**：
   - 对于非对称量化：$Z = Q_{min} - \text{round}(\frac{V_{min}}{S})$
   - 对于对称量化：$Z = 0$

其中 $Q_{max}$ 和 $Q_{min}$ 是目标数据类型的数值范围最大值和最小值。例如对于 INT8 对称量化，$Q_{max}=127$；对于 FP8 等浮点量化场景，则对应其格式所能表示的范围。

### 实现

算法在 [`msmodelslim/core/quantizer/impl/minmax.py`](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/core/quantizer/impl/minmax.py) 中实现。

## 功能介绍

### YAML配置示例

MinMax 作为 [`linear_quant`](linear_quant.md) 处理器中激活值或权重的量化方法，通过 `qconfig.act.method` 或 `qconfig.weight.method` 字段指定，YAML配置示例如下：

```yaml
spec:
  process:
    - type: "linear_quant"
      qconfig:
        act:
          scope: "per_tensor"   # 激活值量化范围
          dtype: "int8"         # 激活值量化数据类型
          symmetric: false      # 激活值非对称量化
          method: "minmax"      # 激活值使用 minmax
        weight:
          scope: "per_channel"  # 权重量化范围
          dtype: "int8"         # 权重量化数据类型
          symmetric: true       # 权重对称量化
          method: "minmax"      # 权重使用 minmax
      include: ["*"]            # 包含的层
      exclude: []               # 排除的层
```

### YAML配置字段详解

以下字段均属于 `linear_quant` 处理器配置，完整的 `linear_quant` 字段说明请参考[线性量化配置字段详解](linear_quant.md#yaml配置字段详解)。

| 参数名 | 所属配置 | 可选值 | 说明 |
|--------|----------|--------|------|
| `qconfig.act.method` | 激活值量化 | `"minmax"` | 指定激活值使用 MinMax 算法 |
| `qconfig.weight.method` | 权重量化 | `"minmax"` | 指定权重使用 MinMax 算法 |

## FAQ

### 精度问题

**现象**：MinMax 在低比特（如 INT8、INT4）下精度下降明显。

**解决方案**：MinMax 算法对离群值（Outliers）非常敏感。如果张量中存在极少数数值巨大的点，MinMax 会为了覆盖这些点而拉大整体量化范围，导致大部分正常数值的量化精度丢失。在低比特或有限位宽的浮点量化场景下，建议配合离群值抑制算法（如 SmoothQuant）或使用更高级的量化算法（如 SSZ）。
