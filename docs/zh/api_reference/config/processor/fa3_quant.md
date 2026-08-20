<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.processor.quant.fa3.processor.FA3QuantProcessorConfig -->
# fa3_quant 配置说明

## 1. 配置概述

FA3（FlashAttention-3）量化处理器配置。

| 项目 | 内容 |
|------|------|
| 配置类 | `FA3QuantProcessorConfig` |
| 源码 | [processor.py](../../../../../msmodelslim/processor/quant/fa3/processor.py) |

## 2. 参数列表

<h3 id="2-1-fa3-quant">2.1 FA3QuantProcessorConfig</h3>

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 可选 | `fa3_quant` | `fa3_quant` | 处理器类型，固定为 `fa3_quant`。 | 无 |
| `qconfig` | `object / null` | 可选 | `null` | — | 统一量化配置；不提供时默认使用 INT8 per-head symmetric，见《QConfig 配置说明》 | 本页 <a href="#2-2-qconfig">§2.2</a> |
| `include` | `list[string]` | 可选 | `['*']` | — | 包含的模块名称模式，默认 `*` 匹配全部模块 | 无 |
| `exclude` | `list[string]` | 可选 | `[]` | — | 排除的模块名称模式，优先级高于 `include` | 无 |
| `details` | `object / null` | 可选 | `null` | — | Q/K/V 分支级量化配置（FA3AttentionDetails），见《FA3AttentionDetails 配置说明》 | 本页 <a href="#2-3-fa3-attention-details">§2.3</a> |

**配置约束**

- 校验 qconfig 与 details 互斥：两者都提供时报错；两者都未提供时默认 INT8 per-head symmetric。

<h3 id="2-2-qconfig">2.2 QConfig</h3>

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

<h3 id="2-3-fa3-attention-details">2.3 FA3AttentionDetails</h3>

FA3 注意力分支级量化配置，分别指定 Q/K/V 的量化方式。

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `fa_q` | `object` | 可选 | `null` | — | Query 分支的量化配置，见《QConfig 配置说明》 | 本页 <a href="#2-2-qconfig">§2.2</a> |
| `fa_k` | `object` | 可选 | `null` | — | Key 分支的量化配置，见《QConfig 配置说明》 | 本页 <a href="#2-2-qconfig">§2.2</a> |
| `fa_v` | `object` | 可选 | `null` | — | Value 分支的量化配置，见《QConfig 配置说明》 | 本页 <a href="#2-2-qconfig">§2.2</a> |

**配置约束**

- 无。

## 3. 完整配置参考

```yaml
apiversion: modelslim_v1
spec:
  process:
  - type: fa3_quant
    include:
    - '*'
    exclude: []
```
