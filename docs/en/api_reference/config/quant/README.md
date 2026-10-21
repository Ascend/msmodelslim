# Quantization Task Configuration Reference Navigation

This directory is the field-level reference for msModelSlim quantization task configuration (YAML). It is organized as: choose an `apiversion` first, then expand `spec`. For the usage workflow and end-to-end steps, see [One-Click Quantization Usage Guide](../../../user_guide/usage_one_click_quantization.md); for the master tables of processors and save formats, see [Configuration Reference Navigation](../README.md); for auto tuning, see [Auto Tuning Configuration Reference Navigation](../tuning/README.md).

## 1. How to Use This Documentation Set

1. **Choose the protocol**: the `apiversion` at the YAML root determines the structure of `spec`. Select the corresponding page by task scenario from the [task configuration](#2-task-configuration-selected-by-apiversion) table below. The available `process` / `save` subtypes differ between protocols; the corresponding task page is authoritative.
2. **Look up task fields**: open the parameter list of the corresponding protocol page. The root block is the task configuration (such as `ModelslimV1QuantConfig`), and the nested blocks are `spec` and the expanded service configurations.
3. **Look up processors / save formats**: `spec.process[]` and `spec.save[]` are dispatched by `type`. For fields, see the processor and save format tables in [Configuration Reference Navigation](../README.md), or open the corresponding page directly under `../processor/` or `../format/`.
4. **Look up the best-practice wrapper**: if the one-click quantization / tuning template carries `metadata`, see [practice_config](practice_config.md); its inner `task` still corresponds to a task configuration of some `apiversion`.

> This page is hand-written navigation; the other pages in this directory are generated from source Pydantic annotations by `skills/docs-management/scripts/gen_config_api_docs.py` (or `gen_quant_config_docs.py`). To change a field description, edit the source annotations such as `Field(description=)` and regenerate; do not edit pages that carry a `generated-by` marker directly.

## 2. Task Configuration (Selected by `apiversion`)

| Configuration | `apiversion` | Description |
|------|--------------|------|
| [modelslim_v1](modelslim_v1.md) | `modelslim_v1` | Default protocol for large language model quantization and weight conversion, including `runner`/`process`/`save`/`dataset`. |
| [multimodal_vlm_modelslim_v1](multimodal_vlm_modelslim_v1.md) | `multimodal_vlm_modelslim_v1` | Quantization protocol for multimodal understanding (VLM) models. |
| [multimodal_sd_modelslim_v1](multimodal_sd_modelslim_v1.md) | `multimodal_sd_modelslim_v1` | Quantization protocol for multimodal generation (DiT/SD) models, including dump and inference parameters. |
| [modelslim_convert](modelslim_convert.md) | `modelslim_convert` | Pure weight conversion protocol, used for weight renaming, structure replacement, and format conversion. |
| [practice_config](practice_config.md) | `modelslim_v1` (inside `task`) | Best-practice configuration: `metadata` + `task`, referenced by auto-tuning strategies. |
