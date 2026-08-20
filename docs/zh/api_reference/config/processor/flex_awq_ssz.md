<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.processor.anti_outlier.flex_smooth.processor.FlexAWQSSZProcessorConfig -->
# flex_awq_ssz 配置说明

## 1. 配置概述

FlexAWQSSZ 平滑+AWQ 处理器配置。

| 项目 | 内容 |
|------|------|
| 配置类 | `FlexAWQSSZProcessorConfig` |
| 源码 | [processor.py](../../../../../msmodelslim/processor/anti_outlier/flex_smooth/processor.py) |

## 2. 参数列表

<h3 id="2-1-flex-awq-ssz">2.1 FlexAWQSSZProcessorConfig</h3>

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 可选 | `flex_awq_ssz` | `flex_awq_ssz` | 处理器类型，固定为 `flex_awq_ssz`。 | 无 |
| `alpha` | `float` | 可选 | `null` | — | 激活→权重的平滑迁移强度（0~1），越大迁移越多离群值到权重。 | 无 |
| `beta` | `float` | 可选 | `null` | — | 额外的平滑调优系数（0~1），配合 alpha 使用。 | 无 |
| `enable_subgraph_type` | `list[any]` | 可选 | `['norm-linear', 'linear-linear', 'ov', 'up-down']` | — | 应用平滑的子图类型列表，默认 `norm-linear`、`linear-linear`、`ov`、`up-down`。 | 无 |
| `include` | `list[string] / null` | 可选 | `null` | — | 包含的模块名称模式；不设置表示全部匹配。 | 无 |
| `exclude` | `list[string] / null` | 可选 | `null` | — | 排除的模块名称模式，优先级高于 `include`。 | 无 |
| `qconfig` | `object` | 必选 | 无 | — | 激活与权重的量化配置，见《LinearQConfig 配置说明》。 | 本页 <a href="#2-2-linear-qconfig">§2.2</a> |

**配置约束**

- 模型级前置校验：拦截 qconfig 缺失的场景

<h3 id="2-2-linear-qconfig">2.2 LinearQConfig</h3>

线性层（Linear）的量化配置，含激活与权重两路量化。

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `act` | `object` | 可选 | `{'dtype': 'float', 'scope': 'per_tensor', 'symmetric': True, 'method': 'none', 'ext': {}}` | — | 激活值的量化配置。默认 `float`（不量化激活），仅对权重做量化。 | 本页 <a href="#2-3-qconfig">§2.3</a> |
| `weight` | `object` | 必选 | 无 | — | 权重的量化配置，必选。 | 本页 <a href="#2-3-qconfig">§2.3</a> |

**配置约束**

- 无。

<h3 id="2-3-qconfig">2.3 QConfig</h3>

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
  - type: flex_awq_ssz
    enable_subgraph_type:
    - norm-linear
    - linear-linear
    - ov
    - up-down
    qconfig:
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
```
