<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.core.quant_service.modelslim_v1.quant_config.ModelslimV1QuantConfig -->
# modelslim_v1 配置说明

## 1. 配置概述

`modelslim_v1` 量化任务配置，位于 YAML 根节点。

| 项目 | 内容 |
|------|------|
| 配置类 | `ModelslimV1QuantConfig` |
| 源码 | [quant_config.py](../../../../../msmodelslim/core/quant_service/modelslim_v1/quant_config.py) |

## 2. 参数列表

<h3 id="2-1-modelslim-v1">2.1 ModelslimV1QuantConfig</h3>

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `apiversion` | `string` | 可选 | `modelslim_v1` | `modelslim_v1` | — | 无 |
| `spec` | `object` | 必选 | 无 | — | `modelslim_v1` 服务的 spec 结构，声明量化流水线、保存格式与校准数据。 | 本页 <a href="#2-2-modelslim-v1-spec">§2.2</a> |

**配置约束**

- 无。

<h3 id="2-2-modelslim-v1-spec">2.2 ModelslimV1ServiceConfig</h3>

`modelslim_v1` 服务的 spec 结构，声明量化流水线、保存格式与校准数据。

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `runner` | `string` | 可选 | `auto` | `auto`、`model_wise`、`layer_wise`、`dp_layer_wise` | 流水线执行方式：`auto` 按设备数量自动选择（单设备 `layer_wise`，多设备 `dp_layer_wise`）、`model_wise` 整模型计算、`layer_wise` 逐层计算、`dp_layer_wise` 数据并行逐层计算。 | 无 |
| `prior` | `list[object]` | 可选 | `[]` | — | 前置阶段列表，每阶段含 process 与 dataset | 本页 <a href="#2-2-1-prior-stage-config">§2.2.1</a> |
| `process` | `list[object]` | 可选 | `[]` | — | 量化处理器链，按顺序执行；每个元素是 `type` 分派的处理器配置，如 `linear_quant`、`awq`、`smooth_quant` 等。 | 本页 <a href="#2-2-2-autoprocessorconfig">§2.2.2</a> |
| `save` | `list[object]` | 可选 | `[]` | — | 保存格式列表，每个元素是 `type` 分派的保存格式配置，如 `ascendv1_saver`、`compressed_tensors`、`mindie_format_saver`。 | 本页 <a href="#2-2-3-quantformatconfig">§2.2.3</a> |
| `dataset` | `string` | 可选 | `mix_calib.jsonl` | — | 校准数据集名称（`lab_calib` 下的文件名）或数据集路径，用于量化的参数估计与敏感性校准。 | 无 |

**配置约束**

- 无。

<h4 id="2-2-1-prior-stage-config">2.2.1 PriorStageConfig</h4>

前置阶段配置：仅 process + dataset，用于如 adapt_rotation stage1 等先验阶段。

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `process` | `list[object]` | 可选 | `[]` | — | 该阶段处理器列表 | 本页 <a href="#2-2-2-autoprocessorconfig">§2.2.2</a> |
| `dataset` | `string / null` | 可选 | `null` | — | 该阶段数据集名称，不提供则使用 spec.dataset | 无 |

**配置约束**

- 无。

<h4 id="2-2-2-autoprocessorconfig">2.2.2 《<a href="../processor/linear_quant.md">AutoProcessorConfig</a>》</h4>

**派生类**

| 配置类 | `type` | 说明 | 文档 |
|--------|----------|------|------|
| `AdaptRotationProcessorConfig` | `adapt_rotation` | 自适应旋转（adapt_rotation）处理器配置。 | 《[adapt_rotation 配置说明](../processor/adapt_rotation.md)》 |
| `AutoroundProcessorConfig` | `autoround_quant` | autoround 量化处理器配置。 | 《[autoround_quant 配置说明](../processor/autoround_quant.md)》 |
| `AWQProcessorConfig` | `awq` | AWQ（Activation-aware Weight Quantization）处理器配置。 | 《[awq 配置说明](../processor/awq.md)》 |
| `BinaryAnalysisProcessorConfig` | `binary_analysis` | 二值（有/无量化）敏感度分析处理器配置。 | 《[binary_analysis 配置说明](../processor/binary_analysis.md)》 |
| `BinaryOperatorLayerWiseProcessorConfig` | `binary_operator_layer_wise` | 逐层敏感度分析处理器配置（对比逐块浮点与量化输出）。 | 《[binary_operator_layer_wise 配置说明](../processor/binary_operator_layer_wise.md)》 |
| `BinaryOperatorModelWiseProcessorConfig` | `binary_operator_model_wise` | 模型级敏感度分析配置（对比模型最终输出，使用 MSE 指标） | 《[binary_operator_model_wise 配置说明](../processor/binary_operator_model_wise.md)》 |
| `DynamicCacheProcessorConfig` | `dynamic_cache` | KV cache 量化处理器配置。 | 《[dynamic_cache 配置说明](../processor/dynamic_cache.md)》 |
| `FA3QuantProcessorConfig` | `fa3_quant` | FA3（FlashAttention-3）量化处理器配置。 | 《[fa3_quant 配置说明](../processor/fa3_quant.md)》 |
| `FlatQuantProcessorConfig` | `flatquant` | FlatQuant处理器配置：定义量化训练参数、策略、混合精度等 | 《[flatquant 配置说明](../processor/flatquant.md)》 |
| `FlexAWQSSZProcessorConfig` | `flex_awq_ssz` | Flex AWQ SSZ 离群值抑制处理器配置。 | 《[flex_awq_ssz 配置说明](../processor/flex_awq_ssz.md)》 |
| `FlexSmoothQuantProcessorConfig` | `flex_smooth_quant` | FlexSmoothQuant 平滑量化处理器配置。 | 《[flex_smooth_quant 配置说明](../processor/flex_smooth_quant.md)》 |
| `FloatSparseProcessorConfig` | `float_sparse` | 浮点稀疏处理器配置。 | 《[float_sparse 配置说明](../processor/float_sparse.md)》 |
| `GroupProcessorConfig` | `group` | 处理器合并器配置。 | 《[group 配置说明](../processor/group.md)》 |
| `IterSmoothProcessorConfig` | `iter_smooth` | 迭代平滑（IterativeSmooth）处理器配置。 | 《[iter_smooth 配置说明](../processor/iter_smooth.md)》 |
| `KVSmoothProcessorConfig` | `kv_smooth` | KV cache 平滑处理器配置。 | 《[kv_smooth 配置说明](../processor/kv_smooth.md)》 |
| `LinearProcessorConfig` | `linear_quant` | 线性层（Linear）量化处理器配置。 | 《[linear_quant 配置说明](../processor/linear_quant.md)》 |
| `LoadProcessorConfig` | `load` | 模块加载/卸载处理器配置。 | 《[load 配置说明](../processor/load.md)》 |
| `OASQProcessorConfig` | `oasq` | OASQ（Outlier-Aware Smooth Quantization）处理器配置。 | 《[oasq 配置说明](../processor/oasq.md)》 |
| `OnlineQuaRotProcessorConfig` | `online_quarot` | 在线 QuaRot 旋转处理器配置。 | 《[online_quarot 配置说明](../processor/online_quarot.md)》 |
| `QuaRotProcessorConfig` | `quarot` | QuaRot（离线旋转）处理器配置。 | 《[quarot 配置说明](../processor/quarot.md)》 |
| `QuantSaveProcessorConfig` | `saver` | 统一保存处理器配置。 | 《[saver 配置说明](../processor/saver.md)》 |
| `SmoothQuantProcessorConfig` | `smooth_quant` | SmoothQuant 平滑量化处理器配置。 | 《[smooth_quant 配置说明](../processor/smooth_quant.md)》 |
| `SVDResidualProcessorConfig` | `svd_res` | SVD 残差（低秩补偿）处理器配置。 | 《[svd_res 配置说明](../processor/svd_res.md)》 |
| `TrainableLinearQuantProcessorConfig` | `trainable_linear_quant` | 可训练线性量化（TLQ）处理器配置。 | 《[trainable_linear_quant 配置说明](../processor/trainable_linear_quant.md)》 |
| `UnaryAnalysisProcessorConfig` | `unary_analysis` | 一元（无量化）敏感度分析处理器配置。 | 《[unary_analysis 配置说明](../processor/unary_analysis.md)》 |

<h4 id="2-2-3-quantformatconfig">2.2.3 《<a href="../format/ascendv1_saver.md">QuantFormatConfig</a>》</h4>

**派生类**

| 配置类 | `type` | 说明 | 文档 |
|--------|----------|------|------|
| `AscendV1QuantFormatConfig` | `ascendv1_saver` | AscendV1 保存格式配置，导出昇腾落盘格式的权重文件。 | 《[ascendv1_saver 配置说明](../format/ascendv1_saver.md)》 |
| `CompressedTensorsQuantFormatConfig` | `compressed_tensors` | compressed_tensors 保存格式配置，导出 safetensors 权重与 config.json。 | 《[compressed_tensors 配置说明](../format/compressed_tensors.md)》 |
| `MindIEQuantFormatConfig` | `mindie_format_saver` | MindIE 保存格式配置，导出 MindIE 落盘格式的权重文件。 | 《[mindie_format_saver 配置说明](../format/mindie_format_saver.md)》 |

## 3. 完整配置参考

```yaml
apiversion: modelslim_v1
spec:
  runner: auto
  prior: []
  process: []
  save: []
  dataset: mix_calib.jsonl
```
