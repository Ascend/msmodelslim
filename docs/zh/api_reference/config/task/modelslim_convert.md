<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.core.quant_service.modelslim_convert.quant_config.ModelslimConvertQuantConfig -->
# modelslim_convert 配置说明

## 1. 配置概述

`modelslim_convert` 量化（权重转换）任务配置，位于 YAML 根节点。

| 项目 | 内容 |
|------|------|
| 配置类 | `ModelslimConvertQuantConfig` |
| 源码 | [quant_config.py](../../../../../msmodelslim/core/quant_service/modelslim_convert/quant_config.py) |

## 2. 参数列表

<h3 id="2-1-modelslim-convert">2.1 ModelslimConvertQuantConfig</h3>

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `apiversion` | `string` | 可选 | Unknown（代码占位；YAML 中须按任务类型显式指定） | `modelslim_v1`、`multimodal_vlm_modelslim_v1`、`multimodal_sd_modelslim_v1`、`modelslim_convert` | API 版本（任务类型），决定 spec 的结构：`modelslim_v1`、`multimodal_vlm_modelslim_v1`、`multimodal_sd_modelslim_v1`、`modelslim_convert`；YAML 中必须显式指定，默认值 `Unknown` 仅为代码内部占位，不可直接使用。 | 无 |
| `spec` | `object` | 必选 | 无 | — | `modelslim_convert` 服务的 spec 结构。<br><br>声明权重名重命名/变换（`preprocess`）、线性层转换规则（`linears`）、<br>保存格式（`save`）、并行执行（`parallel`）与默认值（`defaults`）。 | 本页 <a href="#2-2-modelslim-convert-spec">§2.2</a> |

**配置约束**

- 无。

<h3 id="2-2-modelslim-convert-spec">2.2 ModelslimConvertServiceConfig</h3>

`modelslim_convert` 服务的 spec 结构。

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `preprocess` | `list[object]` | 可选 | `[]` | — | 预处理步骤列表，每项 `type` 为 `rename` 或 `convert`。 | 本页 <a href="#2-3-preprocessconfig">§2.3</a> |
| `linears` | `list[object]` | 可选 | `[]` | — | 线性层转换规则列表。 | 本页 <a href="#2-8-linear-convert-config">§2.8</a> |
| `save` | `list[object]` | 可选 | `[]` | — | 保存格式配置列表，取首个生效。 | 本页 <a href="#2-9-save-config">§2.9</a> |
| `parallel` | `object` | 可选 | 见嵌套配置默认值 | — | 并行执行配置。 | 本页 <a href="#2-10-parallel-spec-config">§2.10</a> |
| `defaults` | `object` | 可选 | 见嵌套配置默认值 | — | 字段缺省时的全局默认值。 | 本页 <a href="#2-11-convert-defaults">§2.11</a> |

**配置约束**

- 无。

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

- 无。

<h3 id="2-8-linear-convert-config">2.8 LinearConvertConfig</h3>

指定匹配的线性层转换到目标 IR 的规则。

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `match` | `list[string]` | 可选 | `[]` | — | 匹配的线性层名称模式列表。 | 无 |
| `target` | `string` | 必选 | 无 | `FLOAT`、`FP8_BLOCK`、`W8A8_MXFP8`、`W4A4_MXFP4`、`W4A8_MXFP8`、`INT4_PACKED`、`NVFP4_MODELOPT`、`HIFP4`、`UNKNOWN` | 转换目标 IR 类型，如 `W8A8_MXFP8`、`INT4_PACKED` 等。 | 无 |
| `route` | `list[string] / string` | 可选 | `auto` | `FLOAT`、`FP8_BLOCK`、`W8A8_MXFP8`、`W4A4_MXFP4`、`W4A8_MXFP8`、`INT4_PACKED`、`NVFP4_MODELOPT`、`HIFP4`、`UNKNOWN`；`auto` | 转换路径：显式 IR 列表（首元素为源 IR），或 `auto` 由虚拟树按权重 dtype 推断。 | 无 |

**配置约束**

- 无。

<h3 id="2-9-save-config">2.9 SaveConfig</h3>

`modelslim_convert` 的保存格式配置。

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 可选 | `ascend_v1` | — | 保存格式：`ascend_v1`（昇腾，与 `ConvertDefaults.dst_format` 的 `ascendv1` 等价）；`compressed_tensors`（HF 兼容 safetensors）；`huggingface`/`hf` 是 `compressed_tensors` 的别名。 | 无 |
| `part_file_size` | `int` | 可选 | `4` | — | 分片文件大小，单位 GB；0 表示不分片。 | 无 |

**配置约束**

- 无。

<h3 id="2-10-parallel-spec-config">2.10 ParallelSpecConfig</h3>

`modelslim_convert` 的并行执行配置。

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `workers` | `int` | 可选 | `1` | — | 并行 worker 数：1 表示单进程组内线程（可配 NPU）；大于1 表示组间多进程 + 组内线程（CPU）。 | 无 |
| `max_group_size` | `int / null` | 可选 | `null` | — | 单个依赖组的最大任务数，超过则拆成多个子组分散到不同进程；不设置表示不拆分。 | 无 |
| `worker_device` | `string` | 可选 | `cpu` | — | worker 运行设备：`cpu` 或 `npu`。 | 无 |
| `npu_max_workers` | `int` | 可选 | `1` | — | 仅 `workers=1` 且 `worker_device=npu` 时生效，限制组内并发以防显存溢出。 | 无 |

**配置约束**

- 无。

<h3 id="2-11-convert-defaults">2.11 ConvertDefaults</h3>

转换规则未显式声明字段时的全局默认值。

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `src_format` | `string` | 可选 | `auto` | — | 源权重格式；`auto` 由模型适配器/权重目录自动推断。 | 无 |
| `dst_format` | `string` | 可选 | `ascendv1` | — | 目标保存格式：`ascendv1`（昇腾，与 `SaveConfig.type` 的 `ascend_v1` 等价）；`compressed_tensors`（HF 兼容 safetensors）；`huggingface`/`hf` 是 `compressed_tensors` 的别名。 | 无 |
| `dst_ir` | `string / null` | 可选 | `null` | `FLOAT`、`FP8_BLOCK`、`W8A8_MXFP8`、`W4A4_MXFP4`、`W4A8_MXFP8`、`INT4_PACKED`、`NVFP4_MODELOPT`、`HIFP4`、`UNKNOWN` | 目标 IR 类型；不设置时由目标格式决定。 | 无 |

**配置约束**

- 无。

## 3. 完整配置参考

```yaml
apiversion: modelslim_convert
spec:
  preprocess: []
  linears: []
  save: []
  parallel:
    workers: 1
    max_group_size: null
    worker_device: cpu
    npu_max_workers: 1
  defaults:
    src_format: auto
    dst_format: ascendv1
    dst_ir: null
```
