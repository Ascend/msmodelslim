<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.processor.adapt_rotation.adapt_rotation.AdaptRotationProcessorConfig -->
# adapt_rotation 配置说明

## 1. 配置概述

自适应旋转（adapt_rotation）处理器配置。

| 项目 | 内容 |
|------|------|
| 配置类 | `AdaptRotationProcessorConfig` |
| 源码 | [adapt_rotation.py](../../../../../msmodelslim/processor/adapt_rotation/adapt_rotation.py) |

## 2. 参数列表

<h3 id="2-1-adapt-rotation">2.1 AdaptRotationProcessorConfig</h3>

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 可选 | `adapt_rotation` | `adapt_rotation` | 处理器类型，固定为 `adapt_rotation`。 | 无 |
| `stage` | `int` | 必选 | 无 | `1`、`2` | 旋转适配阶段：1 或 2，决定使用哪个阶段配置。 | 无 |
| `stage_config` | `object` | 必选 | 无 | — | 阶段配置对象（内部自动组装字段，必选但由 before-validator 根据 `stage` 自动生成，用户无需在 YAML 中配置）；YAML 中不要直接配置该字段，请把阶段字段（如 steps、quant_dtype）平铺在处理器下，见《AdaptRotationStage1ProcessorConfig 配置说明》/《AdaptRotationStage2ProcessorConfig 配置说明》。 | 本页 <a href="#2-2-processorconfig">§2.2</a> |

**配置约束**

- 按 stage 把扁平字段组装为 stage_config：仅允许对应阶段字段，多余字段报错。

<h3 id="2-2-processorconfig">2.2 ProcessorConfig</h3>

**派生类**

- `AdaptRotationStage1ProcessorConfig` — adapt_rotation 阶段1的配置。 本页 <a href="#2-3-adapt-rotation-stage1">§2.3</a>
- `AdaptRotationStage2ProcessorConfig` — adapt_rotation 阶段2的配置。 本页 <a href="#2-4-adapt-rotation-stage2">§2.4</a>

<h4 id="2-3-adapt-rotation-stage1">2.3 AdaptRotationStage1ProcessorConfig</h4>

adapt_rotation 阶段1的配置。

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 可选 | `_adapt_rotation_stage1` | `_adapt_rotation_stage1` | 阶段配置类型，内部标识，无需配置。 | 无 |
| `steps` | `int` | 可选 | `20` | — | 迭代优化步数 | 无 |
| `quant_dtype` | `string` | 可选 | `int4` | `int4`、`int8` | 量化比特数，应与下游量化中激活值量化类型一致（如 w4a4 用 int4，w8a8 用 int8） | 无 |
| `layer_type` | `list[string]` | 可选 | `['up_proj']` | 最少1项 | 要收集激活的层名子串列表 | 无 |
| `block_size` | `int` | 可选 | `-1` | — | 块大小，-1 表示 hidden_dim | 无 |
| `max_samples` | `int` | 可选 | `2048` | — | 每层最大采样数 | 无 |

**配置约束**

- 校验 layer_type：每个元素为非空字符串且长度 <= 128
- 校验 block_size：取值范围为-1或2的非负整数次幂

<h4 id="2-4-adapt-rotation-stage2">2.4 AdaptRotationStage2ProcessorConfig</h4>

adapt_rotation 阶段2的配置。

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 可选 | `_adapt_rotation_stage2` | `_adapt_rotation_stage2` | 阶段配置类型，内部标识，无需配置。 | 无 |
| `online` | `bool` | 可选 | `false` | — | 是否启用在线旋转 | 无 |
| `block_size` | `int` | 可选 | `-1` | — | 块大小，-1 表示 hidden_dim | 无 |
| `down_proj_online_layers` | `list[int]` | 可选 | `[]` | — | down_proj 在线层索引列表 | 无 |
| `max_tp_size` | `int` | 可选 | `4` | — | 最大 TP 并行度 | 无 |

**配置约束**

- 校验 down_proj_online_layers：每个元素为非负整数
- 校验 max_tp_size：必须大于等于1且为2的幂
- 校验 block_size：取值范围为-1或2的非负整数次幂

## 3. 完整配置参考

```yaml
apiversion: modelslim_v1
spec:
  process:
  - type: adapt_rotation
    stage: 1
    steps: 20
    quant_dtype: int4
    layer_type:
    - up_proj
    block_size: -1
    max_samples: 2048
```
