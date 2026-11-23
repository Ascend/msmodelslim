# dynamic_cache 配置说明

<!-- generated-by: tools.config_docgen ; class: msmodelslim.core.quantization_algorithm.quant.kvcache.processor.DynamicCacheProcessorConfig -->

> **自动生成文档，请勿手工修改。**
>
> 本页由 `tools.config_docgen` 根据源码注解生成。
> 修改字段描述、类概述或校验逻辑后，请重新生成并提交生成稿。

## 1. 配置概述

KV cache 量化处理器配置。

| 项目 | 内容 |
|------|------|
| 配置类 | `DynamicCacheProcessorConfig` |
| 源码 | [processor.py](../../../../../msmodelslim/core/quantization_algorithm/quant/kvcache/processor.py) |

## 2. 参数列表

<h3 id="2-1-dynamic-cache">2.1 DynamicCacheProcessorConfig</h3>

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 可选 | `dynamic_cache` | `dynamic_cache` | 处理器类型，固定为 `dynamic_cache`。 | 无 |
| `qconfig` | `object / null` | 可选 | `null` | — | K/V 共用的量化配置，与 `details` 互斥。 | 本页 <a href="#2-2-qconfig">§2.2</a> |
| `details` | `object / null` | 可选 | `null` | — | K/V 分支独立的量化配置，与 `qconfig` 互斥。 | 本页 <a href="#2-3-dynamic-cache-details">§2.3</a> |
| `include` | `list[string]` | 可选 | `[]` | 列表元素：字符串长度 ≤ 200 | 包含的模块名称模式 | 无 |
| `exclude` | `list[string]` | 可选 | `[]` | 列表元素：字符串长度 ≤ 200 | 排除的模块名称模式，优先级高于 `include` | 无 |

**配置约束**

- 校验 `qconfig` 与 `details` 至多配置一个；两者均未配置时回落 int8/per_channel 对称 minmax。

<h3 id="2-2-qconfig">2.2 QConfig</h3>

描述单个张量（权重或激活）的量化方式。

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `dtype` | `string` | 必选 | 无 | `float`、`int8`、`int4`、`mxfp8`、`mxfp4`、`hif4`、`fp8_e4m3`、`fp4_e2m1`、`hif8` | 量化数据类型，如 `int8`、`int4`、`mxfp8`、`mxfp4`、`fp8_e4m3`；`float` 表示该张量不量化。 | 无 |
| `scope` | `string` | 必选 | 无 | `per_tensor`、`per_channel`、`per_group`、`per_block`、`per_token`、`pd_mix`、`per_head`、`per_tile`、`dual_scale`、`per_channel_neg_offset` | 量化粒度，即 scale/zero_point 的计算范围。 | 无 |
| `symmetric` | `bool` | 必选 | 无 | — | 是否对称量化。对称量化只保存 scale；非对称量化额外保存 zero_point，可用性取决于 `dtype`/`scope` 组合。 | 无 |
| `method` | `string` | 必选 | 无 | — | 量化参数估计算法。算法按张量类型分属两套量化器：激活量化器支持 `minmax`、`histogram`、`none`（`none` 仅配合 `float`）；权重量化器支持 `minmax`、`mse_round`、`ssz`、`gptq` 等。具体可用值取决于 `dtype`、`scope`、`symmetric` 与对应量化器实现。 | 无 |
| `ext` | `object` | 可选 | `{}` | — | 量化器扩展参数，随 `method` 与量化器实现而定（如 gptq 的 `percdamp`/`group_size`）；空对象表示无扩展参数。 | 无 |

**配置约束**

- 激活：`per_tensor`（整张量一个尺度）、`per_token`（按 token）、`per_channel`（按通道）、`per_block`（按固定块）、`pd_mix`（Prefill/Decode 混合粒度）、`dual_scale`（双尺度）；<br>权重：`per_channel`（按通道）、`per_group`（按分组）、`per_block`（按固定块）、`dual_scale`（双尺度）、`per_channel_neg_offset`（按通道且 offset 取反的部署语义）；<br>FA3 注意力：`per_head`（按注意力头）、`per_channel`（按通道）、`per_token`（按 token）、`per_block`（按固定块）。具体可用组合取决于 `dtype`、`symmetric`、`method` 与所属量化器/处理器实现。

<h3 id="2-3-dynamic-cache-details">2.3 DynamicCacheDetails</h3>

KV cache 的 K/V 分支独立量化配置。

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `k_qconfig` | `object` | 可选 | `null` | — | Key 分支量化配置 | 本页 <a href="#2-2-qconfig">§2.2</a> |
| `v_qconfig` | `object` | 可选 | `null` | — | Value 分支量化配置 | 本页 <a href="#2-2-qconfig">§2.2</a> |

**配置约束**

- 无。

## 3. 完整配置参考

```yaml
apiversion: modelslim_v1
spec:
  process:
  - type: dynamic_cache
    include: []
    exclude: []
```
