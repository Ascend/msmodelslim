<!-- generated-by: tools.config_docgen ; class: msmodelslim.core.quantization_algorithm.quant.fa3.processor.FA3QuantProcessorConfig -->
> **自动生成文档，请勿手工修改。**
>
> 本页由 `tools.config_docgen` 根据源码注解生成。
> 修改字段描述、类概述或校验逻辑后，请重新生成并提交生成稿。

# fa3_quant 配置说明

## 1. 配置概述

FA3（FlashAttention-3）量化处理器配置。

| 项目 | 内容 |
|------|------|
| 配置类 | `FA3QuantProcessorConfig` |
| 源码 | [processor.py](../../../../../msmodelslim/core/quantization_algorithm/quant/fa3/processor.py) |

## 2. 参数列表

<h3 id="2-1-fa3-quant">2.1 FA3QuantProcessorConfig</h3>

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 可选 | `fa3_quant` | `fa3_quant` | 处理器类型，固定为 `fa3_quant`。 | 无 |
| `qconfig` | `object / null` | 可选 | `null` | — | 统一量化配置；不提供时默认使用 `dtype: int8`、`scope: per_head`、`symmetric: true`、`method: minmax`（对称量化），见<a href="#2-2-qconfig">《QConfig 配置说明》</a>。 | 本页 <a href="#2-2-qconfig">§2.2</a> |
| `include` | `list[string]` | 可选 | `['*']` | 列表元素：字符串长度 ≤ 200 | 包含的模块名称模式，默认 `*` 匹配全部模块 | 无 |
| `exclude` | `list[string]` | 可选 | `[]` | 列表元素：字符串长度 ≤ 200 | 排除的模块名称模式，优先级高于 `include` | 无 |
| `details` | `object / null` | 可选 | `null` | — | Q/K/V 分支级量化配置（FA3AttentionDetails），见<a href="#2-3-fa3-attention-details">《FA3AttentionDetails 配置说明》</a> | 本页 <a href="#2-3-fa3-attention-details">§2.3</a> |

**配置约束**

- `qconfig` 与 `details.fa_q`/`details.fa_k`/`details.fa_v` 都按 FA3 的 Q/K/V 注意力张量解释；`scope` 支持 `per_head`、`per_channel`、`per_tensor`、`per_token` 和 `per_block`。
- `method: static` 目前仅支持 HiF8，且仅支持 `per_head` 或 `per_tensor`；其他 `scope` 不支持静态固定 scale。
- `per_token` 和 `per_block` 是数据无关的动态量化路径；`per_head`、`per_channel` 和 `per_tensor` 需要校准统计。
- 具体 `(dtype, scope, symmetric, method)` 组合仍需有对应的 FA3/激活量化实现；未注册的组合会在创建量化器时失败。
- 校验 qconfig 与 details 互斥：两者都提供时报错；两者都未提供时使用默认 `{dtype: int8, scope: per_head, symmetric: true, method: minmax}`。

<h3 id="2-2-qconfig">2.2 QConfig</h3>

描述单个张量（权重或激活）的量化方式。

- `QConfig` 是量化配置的公共组成单元，字段直接内联在所属配置下（如 `LinearQConfig.qconfig.act` / `qconfig.weight`、AWQ 的 `weight_qconfig`）。
- `scope` 不是所有张量共享的自由组合项，必须与字段所属的张量角色一致；本类仅说明激活和权重的通用粒度，其他张量角色的候选范围由所属处理器配置页补充说明。
- `dtype`、`scope` 与 `symmetric` 共同组成量化方案（scheme），`method` 在该方案下选择参数估计算法；四者与 `ext` 的可用组合由所属量化器或处理器实现决定。

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `dtype` | `string` | 必选 | 无 | `float`、`int8`、`int4`、`mxfp8`、`mxfp4`、`fp8_e4m3`、`hif8` | 量化数据类型，如 `int8`、`int4`、`mxfp8`、`mxfp4`、`fp8_e4m3`；`float` 表示该张量不量化。 | 无 |
| `scope` | `string` | 必选 | 无 | `per_tensor`、`per_channel`、`per_group`、`per_block`、`per_token`、`pd_mix`、`per_head`、`dual_scale`、`per_channel_neg_offset` | 量化粒度，即 scale/zero_point 的计算范围。 | 无 |
| `symmetric` | `bool` | 必选 | 无 | — | 是否对称量化。对称量化只保存 scale；非对称量化额外保存 zero_point，可用性取决于 `dtype`/`scope` 组合。 | 无 |
| `method` | `string` | 必选 | 无 | — | 量化参数估计算法。算法按张量类型分属两套量化器：激活量化器支持 `minmax`、`histogram`、`none`（`none` 仅配合 `float`）；权重量化器支持 `minmax`、`mse_round`、`ssz`、`gptq` 等。具体可用值取决于 `dtype`、`scope`、`symmetric` 与对应量化器实现。 | 无 |
| `ext` | `object` | 可选 | `{}` | — | 量化器扩展参数，随 `method` 与量化器实现而定（如 gptq 的 `percdamp`/`group_size`）；空对象表示无扩展参数。 | 无 |

**配置约束**

- 激活 `scope` 支持 `per_tensor`（整张量一个尺度）、`per_token`（按 token）、`per_channel`（按通道）、`per_block`（按固定块）、`pd_mix`（Prefill/Decode 混合粒度）和 `dual_scale`（双尺度）；该组粒度仅适用于激活字段。
- 权重 `scope` 支持 `per_channel`（按通道）、`per_group`（按分组）、`per_block`（按固定块）、`dual_scale`（双尺度）和 `per_channel_neg_offset`（按通道且 offset 取反的部署语义）；不支持激活专用的 `per_tensor`、`per_token`、`pd_mix`。
- 除激活和权重外，其他张量角色可能定义额外粒度；以所属处理器配置页的约束为准。
- `dtype` 的可选范围取决于张量角色、量化方案和所属量化器实现；例如 `float` 表示不量化，此时激活使用 `method=none`，不能按其他整数/浮点量化类型理解。
- `method` 按张量角色分为两类：激活量化器支持 `minmax`、`histogram`、`none`，其中 `none` 仅配合 `dtype=float`；权重量化器支持 `minmax`、`mse_round`、`ssz`、`gptq` 等。
- `symmetric` 与 `ext` 也受 `dtype`、`scope`、`method` 和量化器实现约束；例如非对称量化 `symmetric=false` 需要实现支持 zero_point，`ext` 只能填写所选 `method` 声明的扩展参数。
- 上述粒度只是各角色允许的候选范围；最终 `(dtype, scope, symmetric, method)` 组合是否存在，仍以所属量化器/处理器的已注册实现为准，未注册的组合会在创建量化器时失败。

<h3 id="2-3-fa3-attention-details">2.3 FA3AttentionDetails</h3>

FA3 注意力分支级量化配置，分别指定 Q/K/V 的量化方式。

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `fa_q` | `object` | 可选 | `null` | — | Query 分支的量化配置，见<a href="#2-2-qconfig">《QConfig 配置说明》</a> | 本页 <a href="#2-2-qconfig">§2.2</a> |
| `fa_k` | `object` | 可选 | `null` | — | Key 分支的量化配置，见<a href="#2-2-qconfig">《QConfig 配置说明》</a> | 本页 <a href="#2-2-qconfig">§2.2</a> |
| `fa_v` | `object` | 可选 | `null` | — | Value 分支的量化配置，见<a href="#2-2-qconfig">《QConfig 配置说明》</a> | 本页 <a href="#2-2-qconfig">§2.2</a> |

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
