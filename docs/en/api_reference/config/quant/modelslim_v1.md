<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.core.quant_service.modelslim_v1.quant_config.ModelslimV1QuantConfig -->
# modelslim_v1 Configuration Reference

## 1. Configuration Overview

`modelslim_v1` quantization task configuration, located at the YAML root.

| Item | Content |
|------|------|
| Config Class | `ModelslimV1QuantConfig` |
| Source | [quant_config.py](../../../../../msmodelslim/core/quant_service/modelslim_v1/quant_config.py) |

`modelslim_v1` is the new-generation quantization processing framework introduced by the quantization tool and is currently evolving quickly. Compared with `modelslim_v0`, `modelslim_v1` has the following advantages:

- Algorithms are implemented independently, and configurations can be combined freely.
- Layer-wise quantization is supported, greatly reducing resource consumption.
- It does not depend on a specific version of CANN.

## 2. Parameter List

<h3 id="2-1-modelslim-v1">2.1 ModelslimV1QuantConfig</h3>

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `apiversion` | `string` | Optional | `modelslim_v1` | `modelslim_v1` | — | None |
| `spec` | `object` | Required | None | — | The spec structure of the `modelslim_v1` service; declares the quantization pipeline, save format, and calibration data. | This page <a href="#2-2-modelslim-v1-spec">§2.2</a> |

**Configuration Constraints**

- None.

<h3 id="2-2-modelslim-v1-spec">2.2 ModelslimV1ServiceConfig</h3>

The spec structure of the `modelslim_v1` service; declares the quantization pipeline, save format, and calibration data.

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `runner` | `string` | Optional | `auto` | `auto`, `model_wise`, `layer_wise`, `dp_layer_wise` | Pipeline execution mode: `auto` selects automatically based on the device count (single device `layer_wise`, multiple devices `dp_layer_wise`), `model_wise` computes the whole model at once, `layer_wise` computes layer by layer, and `dp_layer_wise` computes layer by layer with data parallelism. | None |
| `prior` | `list[object]` | Optional | `[]` | — | List of prior stages; each stage contains process and dataset | This page <a href="#2-2-1-prior-stage-config">§2.2.1</a> |
| `process` | `list[object]` | Optional | `[]` | — | Quantization processor chain, executed in order; each element is a processor configuration dispatched by `type`, such as `linear_quant`, `awq`, `smooth_quant`, and more. | This page <a href="#2-2-2-autoprocessorconfig">§2.2.2</a> |
| `save` | `list[object]` | Optional | `[]` | — | Save format list; each element is a save format configuration dispatched by `type`, such as `ascendv1_saver`, `compressed_tensors`, `mindie_format_saver`. | This page <a href="#2-2-3-quantformatconfig">§2.2.3</a> |
| `dataset` | `string` | Optional | `mix_calib.jsonl` | — | Calibration dataset name (the file name under `lab_calib`) or a dataset path, used for parameter estimation and sensitivity calibration during quantization. | None |

**Configuration Constraints**

- None.

<h4 id="2-2-1-prior-stage-config">2.2.1 PriorStageConfig</h4>

Prior-stage configuration: process + dataset only, used for prior stages such as adapt_rotation stage1.

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `process` | `list[object]` | Optional | `[]` | — | List of processors for this stage | This page <a href="#2-2-2-autoprocessorconfig">§2.2.2</a> |
| `dataset` | `string / null` | Optional | `null` | — | Dataset name for this stage; if not provided, spec.dataset is used | None |

**Configuration Constraints**

- None.

<h4 id="2-2-2-autoprocessorconfig">2.2.2 <a href="../processor/linear_quant.md">AutoProcessorConfig</a></h4>

**Derived Classes**

| Config Class | `type` | Description | Document |
|--------|----------|------|------|
| `AdaptRotationProcessorConfig` | `adapt_rotation` | Adaptive rotation (adapt_rotation) processor configuration. | [adapt_rotation Configuration Reference](../processor/adapt_rotation.md) |
| `AutoroundProcessorConfig` | `autoround_quant` | AutoRound quantization processor configuration. | [autoround_quant Configuration Reference](../processor/autoround_quant.md) |
| `AWQProcessorConfig` | `awq` | AWQ (Activation-aware Weight Quantization) processor configuration. | [awq Configuration Reference](../processor/awq.md) |
| `BinaryAnalysisProcessorConfig` | `binary_analysis` | Binary (quantized/non-quantized) sensitivity analysis processor configuration. | [binary_analysis Configuration Reference](../processor/binary_analysis.md) |
| `BinaryOperatorLayerWiseProcessorConfig` | `binary_operator_layer_wise` | Layer-wise sensitivity analysis processor configuration (compares per-block floating-point and quantized outputs). | [binary_operator_layer_wise Configuration Reference](../processor/binary_operator_layer_wise.md) |
| `BinaryOperatorModelWiseProcessorConfig` | `binary_operator_model_wise` | Model-level sensitivity analysis configuration (compares final model outputs, using the MSE metric) | [binary_operator_model_wise Configuration Reference](../processor/binary_operator_model_wise.md) |
| `DynamicCacheProcessorConfig` | `dynamic_cache` | KV cache quantization processor configuration. | [dynamic_cache Configuration Reference](../processor/dynamic_cache.md) |
| `FA3QuantProcessorConfig` | `fa3_quant` | FA3 (FlashAttention-3) quantization processor configuration. | [fa3_quant Configuration Reference](../processor/fa3_quant.md) |
| `FlatQuantProcessorConfig` | `flatquant` | FlatQuant processor configuration: defines quantization training parameters, strategy, mixed precision, and more | [flatquant Configuration Reference](../processor/flatquant.md) |
| `FlexAWQSSZProcessorConfig` | `flex_awq_ssz` | Flex AWQ SSZ outlier suppression processor configuration. | [flex_awq_ssz Configuration Reference](../processor/flex_awq_ssz.md) |
| `FlexSmoothQuantProcessorConfig` | `flex_smooth_quant` | FlexSmoothQuant smooth quantization processor configuration. | [flex_smooth_quant Configuration Reference](../processor/flex_smooth_quant.md) |
| `FloatSparseProcessorConfig` | `float_sparse` | Floating-point sparsity processor configuration. | [float_sparse Configuration Reference](../processor/float_sparse.md) |
| `GroupProcessorConfig` | `group` | Processor combiner configuration. | [group Configuration Reference](../processor/group.md) |
| `IterSmoothProcessorConfig` | `iter_smooth` | Iterative smooth (IterativeSmooth) processor configuration. | [iter_smooth Configuration Reference](../processor/iter_smooth.md) |
| `KVSmoothProcessorConfig` | `kv_smooth` | KV cache smoothing processor configuration. | [kv_smooth Configuration Reference](../processor/kv_smooth.md) |
| `LinearProcessorConfig` | `linear_quant` | Linear layer quantization processor configuration. | [linear_quant Configuration Reference](../processor/linear_quant.md) |
| `LoadProcessorConfig` | `load` | Module load/unload processor configuration. | [load Configuration Reference](../processor/load.md) |
| `OASQProcessorConfig` | `oasq` | OASQ (Outlier-Aware Smooth Quantization) processor configuration. | [oasq Configuration Reference](../processor/oasq.md) |
| `OnlineQuaRotProcessorConfig` | `online_quarot` | Online QuaRot rotation processor configuration. | [online_quarot Configuration Reference](../processor/online_quarot.md) |
| `QuaRotProcessorConfig` | `quarot` | QuaRot (offline rotation) processor configuration. | [quarot Configuration Reference](../processor/quarot.md) |
| `QuantSaveProcessorConfig` | `saver` | Unified save processor configuration. | [saver Configuration Reference](../processor/saver.md) |
| `SmoothQuantProcessorConfig` | `smooth_quant` | SmoothQuant smooth quantization processor configuration. | [smooth_quant Configuration Reference](../processor/smooth_quant.md) |
| `SVDResidualProcessorConfig` | `svd_res` | SVD residual (low-rank compensation) processor configuration. | [svd_res Configuration Reference](../processor/svd_res.md) |
| `TrainableLinearQuantProcessorConfig` | `trainable_linear_quant` | Trainable linear quantization (TLQ) processor configuration. | [trainable_linear_quant Configuration Reference](../processor/trainable_linear_quant.md) |
| `UnaryAnalysisProcessorConfig` | `unary_analysis` | Unary (non-quantized) sensitivity analysis processor configuration. | [unary_analysis Configuration Reference](../processor/unary_analysis.md) |

<h4 id="2-2-3-quantformatconfig">2.2.3 <a href="../format/ascendv1_saver.md">QuantFormatConfig</a></h4>

**Derived Classes**

| Config Class | `type` | Description | Document |
|--------|----------|------|------|
| `AscendV1QuantFormatConfig` | `ascendv1_saver` | AscendV1 save format configuration; exports weight files in the Ascend on-disk format. | [ascendv1_saver Configuration Reference](../format/ascendv1_saver.md) |
| `CompressedTensorsQuantFormatConfig` | `compressed_tensors` | compressed_tensors save format configuration; exports safetensors weights and config.json. | [compressed_tensors Configuration Reference](../format/compressed_tensors.md) |
| `MindIEQuantFormatConfig` | `mindie_format_saver` | MindIE save format configuration; exports weight files in the MindIE on-disk format. | [mindie_format_saver Configuration Reference](../format/mindie_format_saver.md) |

## 3. Complete Configuration Reference

```yaml
apiversion: modelslim_v1
spec:
  runner: auto
  prior: []
  process: []
  save: []
  dataset: mix_calib.jsonl
```
