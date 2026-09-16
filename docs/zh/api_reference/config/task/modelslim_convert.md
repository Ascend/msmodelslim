<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.core.quant_service.modelslim_convert.quant_config.ModelslimConvertQuantConfig -->
# modelslim_convert 配置说明

## 1. 配置概述

`modelslim_convert` 量化（权重转换）任务配置，位于 YAML 根节点。

| 项目 | 内容 |
|------|------|
| 配置类 | `ModelslimConvertQuantConfig` |
| 源码 | [quant_config.py](../../../../../msmodelslim/core/quant_service/modelslim_convert/quant_config.py) |

执行入口是 `msmodelslim quant --config`，完整命令与操作步骤见《[msmodelslim quant 命令行](../../cli/msmodelslim_quant.md#76-权重转换不需要-model_type)》与《[权重转换使用指南](../../../knowledge_base/ptq/convert/usage_weight_conversion.md)》。

## 2. 参数列表

<h3 id="2-1-modelslim-convert">2.1 ModelslimConvertQuantConfig</h3>

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `apiversion` | `string` | 必选 | 无 | `modelslim_convert` | 权重转换任务类型，固定为 `modelslim_convert`。 | 无 |
| `spec` | `object` | 必选 | 无 | — | `modelslim_convert` 服务的 spec 结构，声明线性层转换规则（`linears`）、保存格式（`save`）与并行（`parallel`）。 | 本页 <a href="#2-2-modelslim-convert-spec">§2.2</a> |

**配置约束**

- YAML 须写 `apiversion: modelslim_convert`。
- `spec` 为必选；日常至少配置 `linears` 与 `save`。

<h3 id="2-2-modelslim-convert-spec">2.2 ModelslimConvertServiceConfig</h3>

`modelslim_convert` 服务的 spec 结构。

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `preprocess` | `list[object]` | 可选 | `[]` | — | 预处理步骤列表，每项 `type` 为 `rename` 或 `convert`。 | 本页 <a href="#2-3-preprocessconfig">§2.3</a> |
| `linears` | `list[object]` | 可选 | `[]` | — | 线性层转换规则列表。 | 本页 <a href="#2-8-linear-convert-config">§2.8</a> |
| `save` | `list[object]` | 可选 | `[]` | — | 保存格式配置列表，取首个生效。 | 本页 <a href="#2-9-save-config">§2.9</a> |
| `parallel` | `object` | 可选 | 见嵌套配置默认值 | — | CPU 并行配置。NPU 多卡用命令行 `--device_id`。 | 本页 <a href="#2-10-parallel-spec-config">§2.10</a> |
| `defaults` | `object` | 可选 | 见嵌套配置默认值 | — | 保底默认值配置。配置了 `save` 时直接按 `save` 执行，不走此处的默认值。 | 本页 <a href="#2-11-convert-defaults">§2.11</a> |

**配置约束**

- 日常至少配置 `linears` 与 `save`。配置了 `save` 时以 `save[0].type` 为准，`defaults.dst_format` 会被忽略。

<h3 id="2-3-preprocessconfig">2.3 PreprocessConfig</h3>

**派生类**

- `RenamePreprocessConfig`（`type: rename`） — `modelslim_convert` 预处理步骤之一：批量重命名权重张量。 本页 <a href="#2-4-rename">§2.4</a>
- `ConvertPreprocessConfig`（`type: convert`） — `modelslim_convert` 预处理步骤之一：对匹配的线性层做权重变换（拆分/合并等）。 本页 <a href="#2-6-convert">§2.6</a>

<h4 id="2-4-rename">2.4 RenamePreprocessConfig</h4>

`modelslim_convert` 预处理步骤之一：批量重命名权重张量。

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 可选 | `rename` | `rename` | 预处理类型，固定为 `rename`。 | 无 |
| `patterns` | `list[object]` | 可选 | `[]` | — | 重命名规则列表，逐条应用到匹配的权重名。 | 本页 <a href="#2-5-rename-pattern">§2.5</a> |

**配置约束**

- 无。

<h3 id="2-5-rename-pattern">2.5 RenamePattern</h3>

权重张量名重命名规则：把匹配 `from` 的权重名改写为 `to`。

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `from` | `string` | 必选 | 无 | — | 源权重名模式，支持通配符；匹配到的权重名将被改写。 | 无 |
| `to` | `string` | 必选 | 无 | — | 改写后的目标权重名模式。 | 无 |

**配置约束**

- 无。

<h4 id="2-6-convert">2.6 ConvertPreprocessConfig</h4>

`modelslim_convert` 预处理步骤之一：对匹配的线性层做权重变换（拆分/合并等）。

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 可选 | `convert` | `convert` | 预处理类型，固定为 `convert`。 | 无 |
| `source` | `list[string]` | 可选 | `[]` | — | 源权重名模式列表（待变换的线性层）。 | 无 |
| `target` | `list[string]` | 可选 | `[]` | — | 目标权重名模式列表（变换结果）。 | 无 |
| `ops` | `list[object]` | 可选 | `[]` | — | 权重变换算子列表，如 `chunk`、`merge`。 | 本页 <a href="#2-7-convert-op-config">§2.7</a> |

**配置约束**

- 无。

<h3 id="2-7-convert-op-config">2.7 ConvertOpConfig</h3>

`convert` 预处理步骤中的权重算子，如拆分/合并 fused 的 gate/up 投影。

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 必选 | 无 | — | 算子类型：`chunk`（拆分 fused gate/up）、`merge`（合并 gate/up）或其他映射算子。 | 无 |
| `dim` | `int / null` | 可选 | `null` | — | 拆分/合并维度：不指定时按算子类型自动推断，`chunk` 为 1，`merge` 为 0。 | 无 |
| `projections` | `list[string] / null` | 可选 | `null` | — | `chunk` 拆出的投影名列表：不指定时自动推断为 `gate_proj`、`up_proj`。 | 无 |

**配置约束**

- `type: chunk` 用于拆分 fused 的 `gate_up_proj`。专家数从模型 `config.json` 的 `num_experts` 读取。
- 权重已经是独立的 `gate_proj` / `up_proj` 时不要再配 `chunk`，只改 `linears.match`。

<h3 id="2-8-linear-convert-config">2.8 LinearConvertConfig</h3>

指定匹配的线性层要转到哪种目标格式。

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `match` | `list[string]` | 可选 | `[]` | — | 匹配的线性层名称模式列表。 | 无 |
| `target` | `string` | 必选 | 无 | `FLOAT`、`W8A8_MXFP8` | 转换目标。常用 `FLOAT`、`W8A8_MXFP8`。 | 无 |

**配置约束**

- `W8A8_MXFP8` 配 `ascend_v1`，`FLOAT` 配 `huggingface`。

<h3 id="2-9-save-config">2.9 SaveConfig</h3>

`modelslim_convert` 的保存格式配置。

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 可选 | `ascend_v1` | — | 保存格式：`ascend_v1`（昇腾 AscendV1）、`huggingface`（HF safetensors）。 | 无 |
| `part_file_size` | `int` | 可选 | `4` | — | 分片文件大小，单位 GB；0 表示不分片。 | 无 |

**配置约束**

- `W8A8_MXFP8` 必须用 `ascend_v1`；`FLOAT` 用 `huggingface`。

<h3 id="2-10-parallel-spec-config">2.10 ParallelSpecConfig</h3>

`modelslim_convert` 的并行执行配置。

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `workers` | `int` | 可选 | `1` | — | CPU 并行 worker 数。 | 无 |
| `max_group_size` | `int / null` | 可选 | `null` | — | 单个依赖组的最大任务数，超过则拆成多个子组分散到不同进程/卡；不设置表示不拆分。 | 无 |

**配置约束**

- `workers` 用于 CPU。NPU 用 `--device npu --device_id ...`。
- `max_group_size`：MoE 等含有大量专家任务的模型，可设置为约「每层专家任务数 / 卡数」（如 8 卡设为 96），防止大组单进程/单卡长尾等待。

<h3 id="2-11-convert-defaults">2.11 ConvertDefaults</h3>

未显式声明对应配置时的全局保底默认值。

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `src_format` | `string` | 可选 | `auto` | — | 源权重格式；工具按 checkpoint 权重 key 与 dtype 自动推断，日常无需配置。 | 无 |
| `dst_format` | `string` | 可选 | `ascendv1` | — | 保底落盘格式。**与 `spec.save` 互斥**：只要配置了 `spec.save`，就直接使用 `save[0].type`，不走本字段；未配置 `spec.save` 时才回退到此默认值。 | 无 |
| `dst_ir` | `string / null` | 可选 | `null` | `FLOAT`、`W8A8_MXFP8` 等；或 null | 保底目标格式。各层目标格式直接由 `linears[].target` 显式指定，日常无需配置此项。 | 无 |

**配置约束与互斥说明**

- **与 `spec.save` 互斥/覆盖**：`spec.save` 优先级高于 `defaults.dst_format`。只要在 YAML 中配置了 `spec.save`（如 `type: ascend_v1` 或 `type: huggingface`），系统直接按 `save` 落盘，**完全不走 `defaults.dst_format` 默认值**。
- **与 `linears.target` 的关系**：线性层转换目标直接由各条规则的 `linears[].target` 指定，不走 `defaults.dst_ir`。
- **无需配置**：日常转换推荐显式配置 `linears` 和 `save`，**无需配置 `defaults`**。

## 3. 完整配置参考

```yaml
apiversion: modelslim_convert
spec:
  linears: []
  save: []
  parallel:
    workers: 1
    max_group_size: null
```

可落地的最小命令：

```bash
msmodelslim quant \
  --model_path "${MODEL_PATH}" \
  --save_path "${SAVE_PATH}" \
  --config "${CONFIG_PATH}"
```

多卡 NPU：

```bash
msmodelslim quant \
  --model_path "${MODEL_PATH}" \
  --save_path "${SAVE_PATH}" \
  --config "${CONFIG_PATH}" \
  --device npu \
  --device_id 0 1 2 3
```
