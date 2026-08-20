<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.processor.flat_quant.flat_quant.FlatQuantProcessorConfig -->
# flatquant 配置说明

## 1. 配置概述

FlatQuant处理器配置：定义量化训练参数、策略、混合精度等

| 项目 | 内容 |
|------|------|
| 配置类 | `FlatQuantProcessorConfig` |
| 源码 | [flat_quant.py](../../../../../msmodelslim/processor/flat_quant/flat_quant.py) |

## 2. 参数列表

<h3 id="2-1-flatquant">2.1 FlatQuantProcessorConfig</h3>

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 可选 | `flatquant` | `flatquant` | 处理器类型标识，固定为 'flatquant' | 无 |
| `include` | `list[string]` | 可选 | `['*']` | — | 包含的模块名称 | 无 |
| `exclude` | `list[string]` | 可选 | `[]` | — | 排除的模块名称 | 无 |
| `strategies` | `list[object]` | 可选 | `[]` | — | 量化策略配置列表 | 本页 <a href="#2-2-flatquant-quant-strategy-config">§2.2</a> |
| `seed` | `int` | 可选 | `0` | — | 随机种子，用于复现结果 | 无 |
| `diag_relu` | `bool` | 可选 | `true` | — | 是否启用 diag_relu 激活函数实现变换矩阵 | 无 |
| `amp_dtype` | `string` | 可选 | `bfloat16` | — | 混合精度类型，用于加速训练 | 无 |
| `a_bits` | `int` | 可选 | `4` | — | 校准训练时激活量化的位宽（如 4bit） | 无 |
| `a_groupsize` | `int` | 可选 | `-1` | — | 校准训练时激活量化的组大小（-1 表示按张量分组） | 无 |
| `a_asym` | `bool` | 可选 | `false` | — | 校准训练时激活量化是否为非对称量化 | 无 |
| `a_per_tensor` | `bool` | 可选 | `false` | — | 校准训练时激活量化是否按张量进行（而非按通道） | 无 |
| `w_bits` | `int` | 可选 | `4` | — | 校准训练时权重量化的位宽（如 4bit） | 无 |
| `w_groupsize` | `int` | 可选 | `-1` | — | 校准训练时权重量化的组大小（-1 表示按张量分组） | 无 |
| `w_asym` | `bool` | 可选 | `false` | — | 校准训练时权重量化是否为非对称量化 | 无 |
| `epochs` | `int` | 可选 | `10` | — | 校准训练的总轮数 | 无 |
| `nsamples` | `int / null` | 可选 | `null` | — | 用于校准的样本数量 | 无 |
| `cali_bsz` | `int` | 可选 | `4` | — | 校准阶段的批次大小 | 无 |
| `flat_lr` | `float` | 可选 | `0.001` | — | FlatQuant 量化训练的学习率 | 无 |
| `add_diag` | `bool` | 可选 | `true` | — | 是否启用对角缩放矩阵，用于全局缩放 | 无 |
| `lwc` | `bool` | 可选 | `true` | — | 是否启用权重校准（训练权重量化参数） | 无 |
| `lac` | `bool` | 可选 | `true` | — | 是否启用激活校准（训练激活量化参数） | 无 |
| `diag_init` | `string` | 可选 | `one_style` | — | 对角缩放矩阵的初始化方式,支持sq_style以及one_style | 无 |
| `diag_alpha` | `float` | 可选 | `0.3` | — | 对角线缩放参数，控制缩放强度 | 无 |
| `warmup` | `bool` | 可选 | `true` | — | 是否启用训练预热机制，提升稳定性 | 无 |
| `deactive_amp` | `bool` | 可选 | `true` | — | 是否禁用混合精度训练（用于调试） | 无 |
| `tran_type` | `string` | 可选 | `svd` | — | 变换矩阵实现方式：svd 表示基于 SVD 分解 | 无 |

**配置约束**

- 校验逻辑：带 `init=False` 的内部字段不允许在 YAML 中显式赋值；如果赋值则抛出错误。

<h3 id="2-2-flatquant-quant-strategy-config">2.2 QuantStrategyConfig</h3>

量化策略配置：定义量化参数、包含/排除模块规则

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `qconfig` | `object` | 必选 | 无 | — | 量化配置参数 | 本页 <a href="#2-3-linear-qconfig">§2.3</a> |
| `include` | `list[string]` | 可选 | `['*']` | — | 要包含的模块名称（支持通配符 *） | 无 |
| `exclude` | `list[string]` | 可选 | `[]` | — | 要排除的模块名称（优先于 include） | 无 |

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
  - type: flatquant
    include:
    - '*'
    exclude: []
    strategies: []
    seed: 0
    diag_relu: true
    amp_dtype: bfloat16
    a_bits: 4
    a_groupsize: -1
    a_asym: false
    a_per_tensor: false
    w_bits: 4
    w_groupsize: -1
    w_asym: false
    epochs: 10
    cali_bsz: 4
    flat_lr: 0.001
    add_diag: true
    lwc: true
    lac: true
    diag_init: one_style
    diag_alpha: 0.3
    warmup: true
    deactive_amp: true
    tran_type: svd
```
