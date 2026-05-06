# Kurtosis：敏感层分析算法说明

## 简介

- **概述**：Kurtosis（峰度）度量用于**linear**范围分析：对激活采样后估计峰度，用于识别分布尖峰与尾部极端值对量化的影响，输出**线性层粒度**排序。`linear` scope 下 `--metrics` 的默认值为 `kurtosis`。
- **核心思想**：超额峰度刻画相对正态的尖峭程度；分布越集中、极端值越突出，量化截断带来的相对风险往往越高。

## 使用前准备

安装 msModelSlim 工具，详情请参见《[msModelSlim工具安装指南](../../getting_started/install_guide.md)》。

## 原理

1. 对层激活做排序与步进采样（控制内存与计算），在采样序列上估计峰度。
2. 超额峰度常用形式：$\text{kurtosis} = \mathbb{E}[(X-\mu)^4]/\sigma^4 - 3$。
3. **解读**：峰度越大，通常表示分布越尖、对极端值越敏感；接近 0 则更接近高斯形态（实现中以具体 `compute_score` 输出为准，用于层间排序）。

## 适用要求

- **推荐场景**：需要细粒度识别激活尖峰、配合敏感层回退或混合精度策略时。
- **模型适配**：**无需**模型适配器额外实现分析接口。

## 功能介绍

### 命令行示例

```bash
msmodelslim analyze linear \
    --model_type Qwen2.5-7B-Instruct \
    --model_path ${model_path} \
    --metrics kurtosis \
    --calib_dataset ${calib_dataset} \
    --pattern "*.down_proj*" "*.o_proj*" \
    --topk 15 \
    --device npu
```

### 命令行参数说明

| 参数 | 说明 |
|------|------|
| `linear`（scope） | 线性层敏感度分析 |
| `--metrics kurtosis` | 指定本算法（亦为 `linear` 默认 metrics） |

完整参数见《[量化敏感层分析工具使用指南](../../feature_guide/sensitive_layer_analysis/usage.md)》。
