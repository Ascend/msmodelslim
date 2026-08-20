<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.processor.quant.autoround.AutoroundProcessorConfig -->
# autoround_quant 配置说明

## 1. 配置概述

autoround 量化处理器配置。

| 项目 | 内容 |
|------|------|
| 配置类 | `AutoroundProcessorConfig` |
| 源码 | [autoround.py](../../../../../msmodelslim/processor/quant/autoround.py) |

## 2. 参数列表

<h3 id="2-1-autoround-quant">2.1 AutoroundProcessorConfig</h3>

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 可选 | `autoround_quant` | `autoround_quant` | 处理器类型，固定为 `autoround_quant`。 | 无 |
| `iters` | `int` | 可选 | `10` | — | 迭代次数，必须大于0 | 无 |
| `enable_minmax_tuning` | `bool` | 可选 | `true` | — | 是否启用最小最大值调优 | 无 |
| `enable_round_tuning` | `bool` | 可选 | `true` | — | 是否启用舍入调优 | 无 |
| `strategies` | `list[object]` | 可选 | `[]` | — | 量化策略配置列表，至少配置一个 | 本页 <a href="#2-2-autoround-quant-strategy-config">§2.2</a> |

**配置约束**

- 校验 strategies：非空；每个 strategy 的 qconfig 满足 group_size 规则（scope=per_group 时 ext.group_size 必须为正整数，scope≠per_group 时不得含 group_size）；并通过量化器 layer 配置预检。

<h3 id="2-2-autoround-quant-strategy-config">2.2 QuantStrategyConfig</h3>

autoround 量化策略：对匹配的线性层应用一组量化配置。

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `qconfig` | `object` | 必选 | 无 | — | 激活与权重的量化配置，见《LinearQConfig 配置说明》。 | 本页 <a href="#2-3-linear-qconfig">§2.3</a> |
| `include` | `list[string]` | 可选 | `['*']` | — | 包含的模块名称模式，默认 `*` 匹配全部模块 | 无 |
| `exclude` | `list[string]` | 可选 | `[]` | — | 排除的模块名称模式，优先级高于 `include` | 无 |

**配置约束**

- 无。

<h3 id="2-3-linear-qconfig">2.3 LinearQConfig</h3>

线性层（Linear）的量化配置，含激活与权重两路量化。

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `act` | `object` | 可选 | `{'dtype': 'float', 'scope': 'per_tensor', 'symmetric': True, 'method': 'none', 'ext': {}}` | — | 激活值的量化配置。默认 `float`（不量化激活），仅对权重做量化。 | 本页 <a href="#2-4-qconfig">§2.4</a> |
| `weight` | `object` | 必选 | 无 | — | 权重的量化配置，必选。 | 本页 <a href="#2-4-qconfig">§2.4</a> |

**配置约束**

- 无。

<h3 id="2-4-qconfig">2.4 QConfig</h3>

描述单个张量（权重或激活）的量化方式。

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `dtype` | `string` | 必选 | 无 | `float`、`int8`、`int4`、`mxfp8`、`mxfp4`、`fp8_e4m3` | 量化数据类型，如 `int8`、`int4`、`mxfp8`、`mxfp4`、`fp8_e4m3`；`float` 表示该张量不量化。 | 无 |
| `scope` | `string` | 必选 | 无 | `per_tensor`、`per_channel`、`per_group`、`per_block`、`per_token`、`pd_mix`、`per_head`、`dual_scale` | 量化粒度，即 scale/zero_point 的计算范围：`per_tensor`（整张量一个尺度）、`per_channel`（按通道）、`per_group`/`per_block`（按分组或固定块）、`per_token`（按 token）、`per_head`（按注意力头）、`dual_scale`（双尺度）等；合法取值组合取决于 `dtype` 与量化器实现。 | 无 |
| `symmetric` | `bool` | 必选 | 无 | — | 是否对称量化。对称量化只保存 scale；非对称量化额外保存 zero_point，可用性取决于 `dtype`/`scope` 组合。 | 无 |
| `method` | `string` | 必选 | 无 | — | 量化参数估计算法，如 `minmax`、`mse_round`、`histogram`、`ssz`、`none` 等；可用取值取决于 `dtype`/`scope`/`symmetric` 组合，`none` 表示不估计参数（配合 `float` 使用）。 | 无 |
| `ext` | `object` | 可选 | `{}` | — | 量化器扩展参数，随 `method` 与量化器实现而定（如 gptq 的 `percdamp`/`group_size`）；空对象表示无扩展参数。 | 无 |

**配置约束**

- 无。

## 3. 完整配置参考

```yaml
apiversion: modelslim_v1
spec:
  process:
  - type: autoround_quant
    iters: 10
    enable_minmax_tuning: true
    enable_round_tuning: true
    strategies:
    - qconfig:
        act:
          dtype: float
          scope: per_tensor
          symmetric: true
          method: none
          ext: {}
        weight:
          dtype: int8
          scope: per_channel
          symmetric: true
          method: minmax
          ext: {}
      include:
      - '*'
      exclude: []
```
