# PDMIX：激活值阶段间混合量化算法说明

## 简介

- 问题：传统 W8A8 静态量化采用静态的激活量化参数，在长上下文或分布漂移场景中易产生较大量化误差，需要回退大量层才能控制精度损失，却因此损失性能收益。
- 目标：阶段间混合 W8A8 动态与 W8A8 静态两种策略，回退少量层即可控制精度损失，输出时获取与W8A8静态量化相近的性能收益。
    - prefilling 阶段：W8A8 动态量化（`per-token`），减少输入上下文的量化信息损失，控制量化精度损失；
    - decoding 阶段：W8A8 静态量化（`per-tensor`），获取输出时的量化性能收益，提高推理性能。

> 说明：阶段间权重量化方式必须保持一致，否则需要存储两份量化权重，因此 PDMIX 量化算法可视为一种`激活值量化`算法，与 W8 per-channel 权重量化结合。

## 使用前准备

安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../../install_guide/install_guide.md)。

## 原理与实现

### 原理

- 保持权重量化不变，激活值量化采用“阶段自适应”的混合量化。
  - prefilling：`per-token` 动态量化，token 级颗粒度在线计算量化参数，减小量化误差；
  - decoding：`per-tensor` 静态量化，离线计算激活量化参数，减少量化参数计算操作，降低推理时延，提高吞吐量。

### 实现

具体实现位置如下：

- 量化校准：[`msmodelslim/core/quantizer/impl/minmax.py`](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/core/quantizer/impl/minmax.py) 中的 `ActPDMixMinmax`
- 量化模式 IR：[`msmodelslim/ir/w8a8_pdmix.py`](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/ir/w8a8_pdmix.py) 中的 `W8A8PDMixFakeQuantLinear`
- 相关常量：[`msmodelslim/ir/const.py`](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/ir/const.py) 定义 `int8_pd_mix_asym`

## 适用要求

- 当前仅支持MindIE推理部署
- 当静态量化精度损失大，需要回退的层较多时，可尝试替换为 PDMIX 量化
- 该算法为线性层量化算法中的激活值量化算法，线性层由 torch.nn.Linear 实现即满足算法需求

## 功能介绍

### 昇腾AI处理器支持情况

- 当前仅支持Atlas A2训练、推理、Atlas A3训练、推理系列产品推理部署

### YAML配置示例

作为Processor使用，YAML配置示例如下：

```yaml
spec:
  process:
    - type: "linear_quant"     # 线性层量化模式处理器
      qconfig:
        act: # 激活值量化配置
          scope: "pd_mix"      # prefilling: per-token；decoding: per-tensor
          dtype: "int8"        # 暂时仅支持 INT8
          symmetric: false     # PDMIX 量化总体为非对称
          method: "minmax"     # 暂时仅支持 MinMax 算法
        weight: # 权重量化配置
          scope: "per_channel" # 暂时仅支持搭配权重 per_channel 量化
          dtype: "int8"        # 暂时仅支持搭配权重 INT8 量化
          symmetric: true      # 仅支持搭配权重对称量化
          method: "minmax"     # 权重量化算法
```

目前仅MindIE支持且只支持W8A8 PDMIX一种量化模式，因此量化配置除了`qconfig.weight.method`可调整外，其他配置组合均未有对应实现。

### YAML配置字段详解

#### qconfig.weight (权重量化配置)

| 参数名       | 作用     | 可选值                            | 说明                                            | 默认值                         |
|-----------|--------|--------------------------------|-----------------------------------------------|-----------------------------|
| scope     | 量化范围   | `"per_channel"` | per_channel: 每个通道独立参数，仅支持搭配权重 per_channel 量化 | `"per_channel"` |
| dtype     | 量化数据类型 | `"int8"` | 8位整数量化，仅支持搭配权重 INT8 量化 | `"int8"` |
| symmetric | 是否对称量化 | `true` | true: 对称量化，零点为0，仅支持搭配权重对称量化 | `true` |
| method    | 量化方法   | `"minmax"` | minmax: minmax量化 | `"minmax"` |
