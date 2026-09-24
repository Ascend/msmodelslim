<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.processor.quant.linear.LinearProcessorConfig -->
# linear_quant 配置说明

## 1. 配置概述

线性层（Linear）量化处理器配置。

| 项目 | 内容 |
|------|------|
| 配置类 | `LinearProcessorConfig` |
| 源码 | [linear.py](../../../../../msmodelslim/processor/quant/linear.py) |

## 2. 参数列表

<h3 id="2-1-linear-quant">2.1 LinearProcessorConfig</h3>

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 可选 | `linear_quant` | `linear_quant` | 处理器类型，固定为 `linear_quant`。 | 无 |
| `qconfig` | `object` | 必选 | 无 | — | 激活与权重的量化配置，见<a href="#2-2-linear-qconfig">《LinearQConfig 配置说明》</a>。 | 本页 <a href="#2-2-linear-qconfig">§2.2</a> |
| `include` | `list[string]` | 可选 | `['*']` | — | 包含的模块名称模式，默认 `*` 匹配全部模块 | 无 |
| `exclude` | `list[string]` | 可选 | `[]` | — | 排除的模块名称模式，优先级高于 `include` | 无 |

**配置约束**

- 校验 qconfig：act/weight 的 (dtype, scope, symmetric, method) 组合必须有已注册量化器实现（如 int8_per_channel+minmax），否则报错；再调用所选量化器的 validate_ext_config，当前 GPTQ 要求 ext 中的 percdamp/block_size/group_size 均为正数。

<h3 id="2-2-linear-qconfig">2.2 LinearQConfig</h3>

线性层（Linear）的量化配置，含激活与权重两路量化。

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `act` | `object` | 可选 | `{'dtype': 'float', 'scope': 'per_tensor', 'symmetric': True, 'method': 'none', 'ext': {}}` | — | 激活值的量化配置。由激活量化器实现，`method` 支持 `minmax`、`histogram`、`none`；`scope` 支持 `per_tensor`、`per_token`、`per_channel`（限 int8/fp8_e4m3/mxfp8 的 minmax）等。默认 `float`（不量化激活），仅对权重做量化。 | 本页 <a href="#2-2-1-qconfig">§2.2.1</a> |
| `weight` | `object` | 必选 | 无 | — | 权重的量化配置，必选。由权重量化器实现，`method` 支持 `minmax`、`mse_round`、`ssz`、`gptq` 等；`scope` 支持 `per_channel`、`per_group`、`per_block` 等（不支持 `per_tensor`/`per_token`）。 | 本页 <a href="#2-2-1-qconfig">§2.2.1</a> |

**配置约束**

- 无。

<h4 id="2-2-1-qconfig">2.2.1 QConfig</h4>

描述单个张量（权重或激活）的量化方式。

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `dtype` | `string` | 必选 | 无 | `float`、`int8`、`int4`、`mxfp8`、`mxfp4`、`fp8_e4m3` | 量化数据类型，如 `int8`、`int4`、`mxfp8`、`mxfp4`、`fp8_e4m3`；`float` 表示该张量不量化。 | 无 |
| `scope` | `string` | 必选 | 无 | `per_tensor`、`per_channel`、`per_group`、`per_block`、`per_token`、`pd_mix`、`per_head`、`dual_scale`、`per_channel_neg_offset` | 量化粒度，即 scale/zero_point 的计算范围。各取值含义：`per_tensor`（整张量一个尺度）、`per_channel`（按通道）、`per_group`/`per_block`（按分组或固定块）、`per_token`（按 token）、`per_head`（按注意力头）、`pd_mix`（Prefill/Decode 混合粒度）、`dual_scale`（双尺度）等；按场景可用取值如下（还须与 `dtype`/`method` 组合匹配）：激活：`per_tensor`、`per_token`、`per_channel`（当前为 `int8`/`fp8_e4m3`/`mxfp8` 的 minmax）、`per_block`（MX minmax）、`pd_mix`、`dual_scale`；权重：`per_channel`、`per_group`（GPTQ）、`per_block`（MX）、`dual_scale`；FA3 注意力：`per_head`（默认）、`per_channel`、`per_token`、`per_block`。 | 无 |
| `symmetric` | `bool` | 必选 | 无 | — | 是否对称量化。对称量化只保存 scale；非对称量化额外保存 zero_point，可用性取决于 `dtype`/`scope` 组合。 | 无 |
| `method` | `string` | 必选 | 无 | — | 量化参数估计算法，如 `minmax`、`mse_round`、`histogram`、`ssz`、`none`、`gptq` 等；算法按张量类型分属两套量化器：激活量化器支持 `minmax`、`histogram`、`none`（配合 `float`）；权重量化器支持 `minmax`、`mse_round`、`ssz`、`gptq` 等。可用取值还取决于 `dtype`/`scope`/`symmetric` 组合，`none` 表示不估计参数。 | 无 |
| `ext` | `object` | 可选 | `{}` | — | 量化器扩展参数，随 `method` 与量化器实现而定（如 gptq 的 `percdamp`/`group_size`）；空对象表示无扩展参数。 | 无 |

**配置约束**

- 无。

## 3. 完整配置参考

```yaml
apiversion: modelslim_v1
spec:
  process:
  - type: linear_quant
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
    include:
    - '*'
    exclude: []
```
