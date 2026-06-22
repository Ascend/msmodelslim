# Std：敏感层分析算法说明

## 简介

- **概述**：std 度量用于`msmodelslim analyze`的**linear**范围分析：在线性层（及支持的卷积层）激活上采集统计量，用数值范围与标准差的比值作为敏感度分数，对**线性层粒度**排序。
- **核心思想**：量化误差与动态范围、离散程度相关；在激活标准差较大时，相同动态范围下的相对扰动更小，故用 `max(|max|,|min|)/std` 形式的 score 刻画层对量化的敏感程度。

## 使用前准备

安装 msModelSlim 工具，详情请参见《[msModelSlim工具安装指南](../../../install_guide/install_guide.md)》。

## 原理

1. **统计量**：在校准前向过程中，对目标层激活统计全局最大/最小值及基于数据中心平移后的标准差。
2. **分数（score）**：
   - $\text{abs\_max} = \max(|\text{max\_value}|, |\text{min\_value}|)$
   - $\text{score} = \text{abs\_max} / \text{std}$（实现中对 $\text{std}=0$ 等情况有防护处理）
3. **解读**：score 越大表示该层对量化越敏感，反映其激活具有较大的范围/波动比值；具体阈值需结合模型与业务精度要求判断。

## 适用要求

- **推荐场景**：常规量化前的敏感层粗筛。
- **计算特点**：实现相对轻量，运行速度较快。
- **模型适配**：无需模型适配器额外实现分析接口；`model_type` 支持范围参见参见《[大模型支持矩阵](../../model_support/foundation_model_support_matrix.md)》。

## 功能介绍

### 命令行示例

```bash
msmodelslim analyze linear \
    --model_type Qwen2.5-7B-Instruct \
    --model_path ${model_path} \
    --metrics std \
    --calib_dataset ${calib_dataset} \
    --pattern "*.down_proj*" "*.o_proj*" \
    --topk 15 \
    --device npu
```

### 命令行参数说明

| 参数 | 说明 |
|------|------|
| `linear` | 线性层敏感度分析 |
| `--metrics` | 指定分析算法，取值为 `std` 时使用本算法 |
| `--pattern` | 层名通配符，过滤待分析线性层 |

完整参数见[敏感层分析工具使用指南参数说明](../../feature_guide/sensitive_layer_analysis/usage.md#34-参数说明)。

## FAQ

### 与其他 linear 指标（quantile、kurtosis）如何选用？

**现象**: 不确定 `std` 与 `quantile`、`kurtosis` 如何取舍，或希望了解各指标侧重点。

**解决方案**: 三者均针对线性层激活分布，`std` 侧重范围与离散度比值；若数据含较多离群点可配合 `quantile`；若关注尖峰与尾部可配合 `kurtosis`。可阅读对应算法说明后按场景选择。
