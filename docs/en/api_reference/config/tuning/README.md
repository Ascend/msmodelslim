# Auto Tuning Configuration Reference Navigation

This directory is the field-level reference for msModelSlim auto-tuning configuration (YAML). It is organized as: write the tuning plan first, then look up strategies and evaluation services by `type`. For the usage workflow and end-to-end steps, see [Auto Tuning Usage Guide](../../../user_guide/usage_auto_precision_tuning.md); for command-line parameters, see [msmodelslim tune Command-Line API Reference](../../cli/msmodelslim_tune.md); for strategy selection, see [Auto Tuning Strategy Overview](../../../knowledge_base/tuning_strategies/README.md). For the fields of quantization task YAML, see [Quantization Task Configuration Reference Navigation](../quant/README.md); for the master tables of processors and save formats, see [Configuration Reference Navigation](../README.md).

## 1. How to Use This Documentation Set

1. **Determine the top-level structure**: the root of a tuning YAML is not `apiversion` but the two required fields of [tuning_plan](tuning_plan.md): `strategy` and `evaluation`.
2. **Look up strategies**: `strategy` is dispatched by `type`. Use the [tuning strategies](#3-tuning-strategies-strategy-dispatched-by-type) table below to find the corresponding page.
3. **Look up evaluation services**: `evaluation` is dispatched by `type`. Use the [evaluation services](#4-evaluation-services-evaluation-dispatched-by-type) table below to find the corresponding page. Currently only `service_oriented` is available.
4. **Look up embedded quantization configuration**: the `template` of `standing_high` and `binary_fallback` references a best-practice or `modelslim_v1` task configuration; for protocol pages, see [Quantization Task Configuration Reference Navigation](../quant/README.md); for processor `type` values, see [Configuration Reference Navigation](../README.md). `standing_high_with_experience` does not require a complete hand-written quantization YAML; it is generated from `quant_type` and `structure_configs`.

> Except for this page, every page in this directory is generated from source Pydantic annotations by `skills/docs-management/scripts/gen_config_api_docs.py`. To change a field description, edit the source annotations such as `Field(description=)` and regenerate; do not edit pages that carry a `generated-by` marker directly.

## 2. Tuning Plan

| Configuration | Description |
|------|------|
| [tuning_plan](tuning_plan.md) | Tuning plan root configuration, containing only `strategy` and `evaluation`. |

## 3. Tuning Strategies (`strategy` Dispatched by `type`)

| `type` | Configuration | Purpose |
|--------|------|------|
| `standing_high` | [standing_high](strategy_standing_high.md) | Standing high: run sensitive layer analysis first, then roll back progressively and try different outlier suppression chains; a quantization template must be provided. |
| `standing_high_with_experience` | [standing_high_with_experience](strategy_standing_high_with_experience.md) | Experience-based Standing high: generates a quantization plan from the quantization type and structure configuration, with no need to hand-write a complete template. |
| `binary_fallback` | [binary_fallback](strategy_binary_fallback.md) | Binary fallback: searches for the shortest fallback prefix over a complete `PracticeConfig` template. |

`standing_high.template` points to [modelslim_v1 Configuration Reference](../quant/modelslim_v1.md). `binary_fallback.template` points to [PracticeConfig Configuration Reference](../quant/practice_config.md), and `apiversion` must be `modelslim_v1`.

## 4. Evaluation Services (`evaluation` Dispatched by `type`)

| `type` | Configuration | Purpose |
|--------|------|------|
| `service_oriented` | [evaluation_service_oriented](evaluation_service_oriented.md) | Service-oriented evaluation: accuracy expectations, AISBench evaluation, and the vLLM-Ascend inference engine. |
