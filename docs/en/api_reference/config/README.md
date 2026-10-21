# Configuration Reference Navigation

This directory is the field-level reference for msModelSlim configuration (YAML). It is organized as: choose a protocol first, then look up the `type`. For the usage workflow and end-to-end steps, see [One-Click Quantization Usage Guide](../../user_guide/usage_one_click_quantization.md); for the task protocol overview, see [Quantization Task Configuration Reference Navigation](quant/README.md); for the auto-tuning protocol overview, see [Auto Tuning Configuration Reference Navigation](tuning/README.md).

## 1. How to Use This Documentation Set

1. **Choose the protocol**: the `apiversion` at the YAML root determines the structure of `spec`. Select the protocol page by task scenario (for an overview, see [Quantization Task Configuration Reference Navigation](quant/README.md)). Field values and the available `process` / `save` subtypes differ between protocols; the corresponding task page is authoritative.
2. **Look up processors**: each item of `spec.process[]` is dispatched by `type`. Use the `type` to find the corresponding page in the [processors](#3-processors-specprocess-dispatched-by-type) table below; `spec.prior[].process` references the same processors.
3. **Look up save formats**: each item of `spec.save[]` is dispatched by `type`. Use the [save formats](#4-save-formats-specsave-dispatched-by-type) table below to find the corresponding page; the VLM / multimodal generation protocols accept only the AscendV1 and MindIE types.
4. **Look up tuning**: for the top-level auto-tuning structure, see [Auto Tuning Configuration Reference Navigation](tuning/README.md); for the fields of `strategy` and `evaluation`, see the [auto tuning](#5-auto-tuning) table below.

> This page, [Quantization Task Configuration Reference Navigation](quant/README.md), and [Auto Tuning Configuration Reference Navigation](tuning/README.md) are hand-written navigation pages; the remaining configuration pages are generated from source Pydantic annotations by `skills/docs-management/scripts/gen_config_api_docs.py`. To change a field description, edit the source annotations such as `Field(description=)` and regenerate; do not edit pages that carry a `generated-by` marker directly.

## 2. Task Configuration (Selected by `apiversion`)

To look up fields by protocol, see [Quantization Task Configuration Reference Navigation](quant/README.md). The table below is a quick entry point:

| Configuration | `apiversion` | Description |
|------|--------------|------|
| [modelslim_v1](quant/modelslim_v1.md) | `modelslim_v1` | Default protocol for large language model quantization, including `runner`/`process`/`save`/`dataset`. |
| [multimodal_vlm_modelslim_v1](quant/multimodal_vlm_modelslim_v1.md) | `multimodal_vlm_modelslim_v1` | Quantization protocol for multimodal understanding (VLM) models. |
| [multimodal_sd_modelslim_v1](quant/multimodal_sd_modelslim_v1.md) | `multimodal_sd_modelslim_v1` | Quantization protocol for multimodal generation (DiT/SD) models, including dump and inference parameters. |
| [modelslim_convert](quant/modelslim_convert.md) | `modelslim_convert` | Pure weight conversion protocol, used for weight renaming, structure replacement, and format conversion. |
| [practice_config](quant/practice_config.md) | `modelslim_v1` (inside `task`) | Best-practice configuration: `metadata` + `task`, referenced by auto-tuning strategies. |

## 3. Processors (`spec.process[]` Dispatched by `type`)

| `type` | Configuration | Purpose |
|--------|------|------|
| `adapt_rotation` | [adapt_rotation](processor/adapt_rotation.md) | Adaptive rotation; suppresses outliers. |
| `autoround_quant` | [autoround_quant](processor/autoround_quant.md) | AutoRound quantization training. |
| `awq` | [awq](processor/awq.md) | AWQ activation-aware weight quantization. |
| `binary_analysis` | [binary_analysis](processor/binary_analysis.md) | Binary (quantized/non-quantized) sensitivity analysis. |
| `binary_operator_layer_wise` | [binary_operator_layer_wise](processor/binary_operator_layer_wise.md) | Layer-wise sensitivity analysis. |
| `binary_operator_model_wise` | [binary_operator_model_wise](processor/binary_operator_model_wise.md) | Model-level sensitivity analysis. |
| `dynamic_cache` | [dynamic_cache](processor/dynamic_cache.md) | KV cache dynamic quantization. |
| `fa3_quant` | [fa3_quant](processor/fa3_quant.md) | FA3 attention quantization. |
| `flatquant` | [flatquant](processor/flatquant.md) | FlatQuant quantization training. |
| `flex_awq_ssz` | [flex_awq_ssz](processor/flex_awq_ssz.md) | FlexAWQSSZ smooth quantization. |
| `flex_smooth_quant` | [flex_smooth_quant](processor/flex_smooth_quant.md) | FlexSmoothQuant smooth quantization. |
| `float_sparse` | [float_sparse](processor/float_sparse.md) | Floating-point sparsity. |
| `group` | [group](processor/group.md) | Combiner processor; shares the inference process. |
| `iter_smooth` | [iter_smooth](processor/iter_smooth.md) | Iterative smoothing. |
| `kv_smooth` | [kv_smooth](processor/kv_smooth.md) | KV cache smoothing. |
| `linear_quant` | [linear_quant](processor/linear_quant.md) | Linear layer quantization, including `qconfig` (activation/weight). |
| `load` | [load](processor/load.md) | Module load/unload. |
| `oasq` | [oasq](processor/oasq.md) | OASQ outlier-aware smooth quantization. |
| `online_quarot` | [online_quarot](processor/online_quarot.md) | Online QuaRot rotation. |
| `quarot` | [quarot](processor/quarot.md) | Offline QuaRot rotation. |
| `saver` | [saver](processor/saver.md) | Unified save processor. |
| `smooth_quant` | [smooth_quant](processor/smooth_quant.md) | SmoothQuant smooth quantization. |
| `svd_res` | [svd_res](processor/svd_res.md) | SVD low-rank residual compensation. |
| `trainable_linear_quant` | [trainable_linear_quant](processor/trainable_linear_quant.md) | Trainable linear quantization (TLQ). |
| `unary_analysis` | [unary_analysis](processor/unary_analysis.md) | Unary (non-quantized) sensitivity analysis. |

## 4. Save Formats (`spec.save[]` Dispatched by `type`)

| `type` | Configuration | Applicable Protocol |
|--------|------|----------|
| `ascendv1_saver` | [ascendv1_saver](format/ascendv1_saver.md) | All protocols (`modelslim_v1`, `multimodal_*`, `modelslim_convert`). |
| `compressed_tensors` | [compressed_tensors](format/compressed_tensors.md) | Only `modelslim_v1` and `modelslim_convert`; the VLM / multimodal generation protocols do not accept it. |
| `mindie_format_saver` | [mindie_format_saver](format/mindie_format_saver.md) | All protocols. |

## 5. Auto Tuning

For the top-level protocol and field overview, see [Auto Tuning Configuration Reference Navigation](tuning/README.md). The following are field references for each strategy and evaluation service:

| Configuration | Description |
|------|------|
| [tuning_plan](tuning/tuning_plan.md) | Tuning plan root configuration (`strategy` + `evaluation`). |
| [strategy_standing_high](tuning/strategy_standing_high.md) | The `standing_high` strategy. |
| [strategy_standing_high_with_experience](tuning/strategy_standing_high_with_experience.md) | The `standing_high_with_experience` strategy. |
| [strategy_binary_fallback](tuning/strategy_binary_fallback.md) | The `binary_fallback` strategy. |
| [evaluation_service_oriented](tuning/evaluation_service_oriented.md) | AISBench evaluation service and pre-check configuration. |
