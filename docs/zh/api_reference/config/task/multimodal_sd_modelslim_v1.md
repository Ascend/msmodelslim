<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.core.quant_service.multimodal_sd_v1.quant_config.MultimodalSDModelslimV1QuantConfig -->
# multimodal_sd_modelslim_v1 配置说明

## 1. 配置概述

`multimodal_sd_modelslim_v1` 量化任务配置，位于 YAML 根节点。

| 项目 | 内容 |
|------|------|
| 配置类 | `MultimodalSDModelslimV1QuantConfig` |
| 源码 | [quant_config.py](../../../../../msmodelslim/core/quant_service/multimodal_sd_v1/quant_config.py) |

## 2. 参数列表

<h3 id="2-1-multimodal-sd-modelslim-v1">2.1 MultimodalSDModelslimV1QuantConfig</h3>

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `apiversion` | `string` | 可选 | Unknown（代码占位；YAML 中须按任务类型显式指定） | `modelslim_v1`、`multimodal_vlm_modelslim_v1`、`multimodal_sd_modelslim_v1`、`modelslim_convert` | API 版本（任务类型），决定 spec 的结构：`modelslim_v1`、`multimodal_vlm_modelslim_v1`、`multimodal_sd_modelslim_v1`、`modelslim_convert`；YAML 中必须显式指定，默认值 `Unknown` 仅为代码内部占位，不可直接使用。 | 无 |
| `spec` | `object` | 必选 | 无 | — | `multimodal_sd_modelslim_v1` 服务的 spec 结构。<br><br>面向多模态生成（SD）模型：在通用字段之外支持按专家（`per_expert`）覆盖处理器链，<br>并提供多模态专用 `multimodal_sd_config`。 | 本页 <a href="#2-2-multimodal-sd-modelslim-v1-spec">§2.2</a> |

**配置约束**

- 无。

<h3 id="2-2-multimodal-sd-modelslim-v1-spec">2.2 MultimodalSDServiceConfig</h3>

`multimodal_sd_modelslim_v1` 服务的 spec 结构。

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `runner` | `string` | 可选 | `layer_wise` | `auto`、`model_wise`、`layer_wise`、`dp_layer_wise` | 流水线执行方式：`layer_wise` 逐层计算（默认）、`auto` 按设备数量自动选择、`model_wise` 整模型计算、`dp_layer_wise` 数据并行逐层计算。 | 无 |
| `prior` | `list[object]` | 可选 | `[]` | — | 前置阶段列表，每阶段含 process 与 dataset | 本页 <a href="#2-3-prior-stage-config">§2.3</a> |
| `process` | `list[object]` | 可选 | `[]` | — | 量化处理器链，按顺序执行；每个元素是 `type` 分派的处理器配置。 | 本页 <a href="#2-4-autoprocessorconfig">§2.4</a> |
| `per_expert` | `object / null` | 可选 | `null` | — | 按专家覆盖 process；值为 Processor 列表。某专家在此出现则整链替换，否则回退 process | 本页 <a href="#2-4-autoprocessorconfig">§2.4</a> |
| `save` | `list[object]` | 可选 | `[]` | — | 保存格式列表，每个元素是 `type` 分派的保存格式配置。 | 本页 <a href="#2-5-quantformatconfig">§2.5</a> |
| `dataset` | `string` | 可选 | `mix_calib.jsonl` | — | 校准数据集名称（`lab_calib` 下的文件名）或数据集路径。 | 无 |
| `multimodal_sd_config` | `object` | 可选 | 由工厂函数生成 | — | 多模态生成模型的专用配置，可为字典或 `MultimodalSDConfig` 实例，含 `dump_config` 与 `inference_config`。 | 本页 <a href="#2-6-multimodal-sd-config">§2.6</a> |

**配置约束**

- 无。

<h3 id="2-3-prior-stage-config">2.3 PriorStageConfig</h3>

前置阶段配置：仅 process + dataset，用于如 adapt_rotation stage1 等先验阶段。

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `process` | `list[object]` | 可选 | `[]` | — | 该阶段处理器列表 | 本页 <a href="#2-4-autoprocessorconfig">§2.4</a> |
| `dataset` | `string / null` | 可选 | `null` | — | 该阶段数据集名称，不提供则使用 spec.dataset | 无 |

**配置约束**

- 无。

<h3 id="2-4-autoprocessorconfig">2.4 AutoProcessorConfig</h3>

**派生类**

- `AdaptRotationProcessorConfig`（`type: adapt_rotation`） — 自适应旋转（adapt_rotation）处理器配置。 《[adapt_rotation 配置说明](../processor/adapt_rotation.md)》
- `AutoroundProcessorConfig`（`type: autoround_quant`） — autoround 量化处理器配置。 《[autoround_quant 配置说明](../processor/autoround_quant.md)》
- `AWQProcessorConfig`（`type: awq`） — AWQ（Activation-aware Weight Quantization）处理器配置。 《[awq 配置说明](../processor/awq.md)》
- `BinaryAnalysisProcessorConfig`（`type: binary_analysis`） — 二值（有/无量化）敏感性分析处理器配置。 《[binary_analysis 配置说明](../processor/binary_analysis.md)》
- `BinaryOperatorLayerWiseProcessorConfig`（`type: binary_operator_layer_wise`） — 逐层敏感度分析处理器配置（对比逐块浮点与量化输出）。 《[binary_operator_layer_wise 配置说明](../processor/binary_operator_layer_wise.md)》
- `BinaryOperatorModelWiseProcessorConfig`（`type: binary_operator_model_wise`） — 模型级敏感性分析配置（对比模型最终输出，使用 MSE 指标） 《[binary_operator_model_wise 配置说明](../processor/binary_operator_model_wise.md)》
- `DynamicCacheProcessorConfig`（`type: dynamic_cache`） — KV cache 量化处理器配置。 《[dynamic_cache 配置说明](../processor/dynamic_cache.md)》
- `FA3QuantProcessorConfig`（`type: fa3_quant`） — FA3（FlashAttention-3）量化处理器配置。 《[fa3_quant 配置说明](../processor/fa3_quant.md)》
- `FlatQuantProcessorConfig`（`type: flatquant`） — FlatQuant处理器配置：定义量化训练参数、策略、混合精度等 《[flatquant 配置说明](../processor/flatquant.md)》
- `FlexAWQSSZProcessorConfig`（`type: flex_awq_ssz`） — FlexAWQSSZ 平滑+AWQ 处理器配置。 《[flex_awq_ssz 配置说明](../processor/flex_awq_ssz.md)》
- `FlexSmoothQuantProcessorConfig`（`type: flex_smooth_quant`） — FlexSmoothQuant 平滑量化处理器配置。 《[flex_smooth_quant 配置说明](../processor/flex_smooth_quant.md)》
- `FloatSparseProcessorConfig`（`type: float_sparse`） — 浮点稀疏处理器配置。 《[float_sparse 配置说明](../processor/float_sparse.md)》
- `GroupProcessorConfig`（`type: group`） — 处理器合并器配置。 《[group 配置说明](../processor/group.md)》
- `IterSmoothProcessorConfig`（`type: iter_smooth`） — 迭代平滑（IterativeSmooth）处理器配置。 《[iter_smooth 配置说明](../processor/iter_smooth.md)》
- `KVSmoothProcessorConfig`（`type: kv_smooth`） — KV cache 平滑处理器配置。 《[kv_smooth 配置说明](../processor/kv_smooth.md)》
- `LinearProcessorConfig`（`type: linear_quant`） — 线性层（Linear）量化处理器配置。 《[linear_quant 配置说明](../processor/linear_quant.md)》
- `LoadProcessorConfig`（`type: load`） — 模块加载/卸载处理器配置。 《[load 配置说明](../processor/load.md)》
- `OASQProcessorConfig`（`type: oasq`） — OASQ（Outlier-Aware Smooth Quantization）处理器配置。 《[oasq 配置说明](../processor/oasq.md)》
- `OnlineQuaRotProcessorConfig`（`type: online_quarot`） — 在线 QuaRot 旋转处理器配置。 《[online_quarot 配置说明](../processor/online_quarot.md)》
- `QuaRotProcessorConfig`（`type: quarot`） — QuaRot（离线旋转）处理器配置。 《[quarot 配置说明](../processor/quarot.md)》
- `QuantSaveProcessorConfig`（`type: saver`） — 统一保存处理器配置。 《[saver 配置说明](../processor/saver.md)》
- `SmoothQuantProcessorConfig`（`type: smooth_quant`） — SmoothQuant 平滑量化处理器配置。 《[smooth_quant 配置说明](../processor/smooth_quant.md)》
- `SVDResidualProcessorConfig`（`type: svd_res`） — SVD 残差（低秩补偿）处理器配置。 《[svd_res 配置说明](../processor/svd_res.md)》
- `TrainableLinearQuantProcessorConfig`（`type: trainable_linear_quant`） — 可训练线性量化（TLQ）处理器配置。 《[trainable_linear_quant 配置说明](../processor/trainable_linear_quant.md)》
- `UnaryAnalysisProcessorConfig`（`type: unary_analysis`） — 一元（无量化）敏感性分析处理器配置。 《[unary_analysis 配置说明](../processor/unary_analysis.md)》

<h3 id="2-5-quantformatconfig">2.5 QuantFormatConfig</h3>

**派生类**

- `AscendV1QuantFormatConfig`（`type: ascendv1_saver`） — AscendV1 保存格式配置，导出昇腾落盘格式的权重文件。 《[ascendv1_saver 配置说明](../format/ascendv1_saver.md)》
- `CompressedTensorsQuantFormatConfig`（`type: compressed_tensors`） — compressed_tensors 保存格式配置，导出 safetensors 权重与 config.json。 《[compressed_tensors 配置说明](../format/compressed_tensors.md)》
- `MindIEQuantFormatConfig`（`type: mindie_format_saver`） — MindIE 保存格式配置，导出 MindIE 落盘格式的权重文件。 《[mindie_format_saver 配置说明](../format/mindie_format_saver.md)》

<h3 id="2-6-multimodal-sd-config">2.6 MultimodalSDConfig</h3>

多模态生成（SD）模型的专用配置，含 dump 与推理参数。

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `dump_config` | `object` | 必选 | 无 | — | 浮点 dump 配置，必选。 | 本页 <a href="#2-7-dump-config">§2.7</a> |
| `inference_config` | `object / null` | 可选 | `null` | — | 推理参数字典，经模型适配器的 InferenceConfig 类校验；与迁移期字段 `model_config` 互斥。 | 无 |

**配置约束**

- 校验 inference_config 与迁移期字段 model_config 互斥：两者同时提供时报错。

<h3 id="2-7-dump-config">2.7 DumpConfig</h3>

多模态生成（SD）模型的浮点 dump 配置。

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `enable_dump` | `bool` | 可选 | `true` | — | 是否在量化前对模型做浮点 dump（导出 pth 校准数据）；`false` 时跳过 pth 加载与浮点 dump。 | 无 |
| `capture_mode` | `string` | 可选 | `args` | `args` | dump 捕获模式，当前仅支持 `args`。 | 无 |
| `dump_data_dir` | `string` | 可选 | `` | — | 浮点 dump 数据的输出目录；为空时回退到 save_path。 | 无 |

**配置约束**

- 无。

## 3. 完整配置参考

```yaml
apiversion: multimodal_sd_modelslim_v1
spec:
  runner: layer_wise
  prior: []
  process: []
  save:
  - type: ascendv1_saver
  dataset: mix_calib.jsonl
  multimodal_sd_config:
    dump_config:
      enable_dump: true
      capture_mode: args
      dump_data_dir: ''
```
