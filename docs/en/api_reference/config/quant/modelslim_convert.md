<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.core.quant_service.modelslim_convert.quant_config.ModelslimConvertQuantConfig -->
# modelslim_convert Configuration Reference

## 1. Configuration Overview

`modelslim_convert` quantization (weight conversion) task configuration, located at the YAML root.

| Item | Content |
|------|------|
| Config Class | `ModelslimConvertQuantConfig` |
| Source | [quant_config.py](../../../../../msmodelslim/core/quant_service/modelslim_convert/quant_config.py) |

## 2. Parameter List

<h3 id="2-1-modelslim-convert">2.1 ModelslimConvertQuantConfig</h3>

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `apiversion` | `string` | Optional | `modelslim_convert` | `modelslim_convert` | Weight conversion task type, fixed to `modelslim_convert`. | None |
| `spec` | `object` | Required | None | — | The spec structure of the `modelslim_convert` service.<br><br>Declares weight name renaming/transformation (`preprocess`), linear layer conversion rules (`linears`),<br>the save format (`save`), parallel execution (`parallel`), and defaults (`defaults`). | This page <a href="#2-2-modelslim-convert-spec">§2.2</a> |

**Configuration Constraints**

- None.

<h3 id="2-2-modelslim-convert-spec">2.2 ModelslimConvertServiceConfig</h3>

The spec structure of the `modelslim_convert` service.

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `preprocess` | `list[object]` | Optional | `[]` | — | List of preprocessing steps; each item has `type` `rename` or `convert`. | This page <a href="#2-2-1-preprocessconfig">§2.2.1</a> |
| `linears` | `list[object]` | Optional | `[]` | — | List of linear layer conversion rules. | This page <a href="#2-2-6-linear-convert-config">§2.2.6</a> |
| `save` | `list[object]` | Optional | `[]` | — | List of save format configurations; the first one takes effect. | This page <a href="#2-2-7-save-config">§2.2.7</a> |
| `parallel` | `object` | Optional | See the nested configuration default | — | CPU parallelism configuration. For NPU multi-card, use the command line `--device_id`. | This page <a href="#2-2-8-parallel-spec-config">§2.2.8</a> |
| `defaults` | `object` | Optional | See the nested configuration default | — | Global default values used when fields are omitted. | This page <a href="#2-2-9-convert-defaults">§2.2.9</a> |

**Configuration Constraints**

- None.

<h4 id="2-2-1-preprocessconfig">2.2.1 PreprocessConfig</h4>

**Derived Classes**

| Config Class | `type` | Description | Document |
|--------|----------|------|------|
| `RenamePreprocessConfig` | `rename` | One of the `modelslim_convert` preprocessing steps: batch-renames weight tensors. | This page <a href="#2-2-2-rename">§2.2.2</a> |
| `ConvertPreprocessConfig` | `convert` | One of the `modelslim_convert` preprocessing steps: applies weight transformations (split/merge, and so on) to matching linear layers. | This page <a href="#2-2-4-convert">§2.2.4</a> |

<h4 id="2-2-2-rename">2.2.2 RenamePreprocessConfig</h4>

One of the `modelslim_convert` preprocessing steps: batch-renames weight tensors.

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | Optional | `rename` | `rename` | Preprocessing type, fixed to `rename`. | None |
| `patterns` | `list[object]` | Optional | `[]` | — | List of renaming rules, applied one by one to matching weight names. | This page <a href="#2-2-3-rename-pattern">§2.2.3</a> |

**Configuration Constraints**

- None.

<h4 id="2-2-3-rename-pattern">2.2.3 RenamePattern</h4>

Weight tensor renaming rule: rewrites weight names matching `from` to `to`.

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `from` | `string` | Required | None | — | Source weight name pattern; wildcards are supported. Matching weight names are rewritten. | None |
| `to` | `string` | Required | None | — | Target weight name pattern after rewriting. | None |

**Configuration Constraints**

- None.

<h4 id="2-2-4-convert">2.2.4 ConvertPreprocessConfig</h4>

One of the `modelslim_convert` preprocessing steps: applies weight transformations (split/merge, and so on) to matching linear layers.

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | Optional | `convert` | `convert` | Preprocessing type, fixed to `convert`. | None |
| `source` | `list[string]` | Optional | `[]` | — | List of source weight name patterns (the linear layers to transform). | None |
| `target` | `list[string]` | Optional | `[]` | — | List of target weight name patterns (the transformation result). | None |
| `ops` | `list[object]` | Optional | `[]` | — | List of weight transformation operators, such as `chunk` and `merge`. | This page <a href="#2-2-5-convert-op-config">§2.2.5</a> |

**Configuration Constraints**

- None.

<h4 id="2-2-5-convert-op-config">2.2.5 ConvertOpConfig</h4>

Weight operators in the `convert` preprocessing step, such as splitting/merging fused gate/up projections.

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | Required | None | — | Operator type: `chunk` (splits fused gate/up), `merge` (merges gate/up), or another mapping operator. | None |
| `dim` | `int / null` | Optional | `null` | — | Split/merge dimension: when unspecified it is inferred from the operator type, being 1 for `chunk` and 0 for `merge`. | None |
| `projections` | `list[string] / null` | Optional | `null` | — | Names of the projections produced by `chunk`: when unspecified they are inferred as `gate_proj` and `up_proj`. | None |

**Configuration Constraints**

- None.

<h4 id="2-2-6-linear-convert-config">2.2.6 LinearConvertConfig</h4>

Specifies the rule for converting matching linear layers to the target IR.

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `match` | `list[string]` | Optional | `[]` | — | List of matching linear layer name patterns. | None |
| `target` | `string` | Required | None | `FLOAT`, `FP8_BLOCK`, `W8A8_MXFP8`, `W4A4_MXFP4`, `W4A8_MXFP8`, `INT4_PACKED`, `NVFP4_MODELOPT`, `HIFP4`, `UNKNOWN` | Target IR type for conversion, such as `W8A8_MXFP8` or `INT4_PACKED`. | None |
| `route` | `list[string] / string` | Optional | `auto` | `FLOAT`, `FP8_BLOCK`, `W8A8_MXFP8`, `W4A4_MXFP4`, `W4A8_MXFP8`, `INT4_PACKED`, `NVFP4_MODELOPT`, `HIFP4`, `UNKNOWN`; `auto` | Conversion route: an explicit IR list (the first element is the source IR), or `auto` to let the virtual tree infer it from the weight dtype. | None |

**Configuration Constraints**

- None.

<h4 id="2-2-7-save-config">2.2.7 SaveConfig</h4>

Save format configuration of `modelslim_convert`.

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | Optional | `ascend_v1` | — | Save format: `ascend_v1` (Ascend; equivalent to `ascendv1` in `ConvertDefaults.dst_format`); `compressed_tensors` (HF-compatible safetensors); `huggingface`/`hf` are aliases of `compressed_tensors`. | None |
| `part_file_size` | `int` | Optional | `4` | — | Shard file size in GB; 0 means no sharding. | None |

**Configuration Constraints**

- None.

<h4 id="2-2-8-parallel-spec-config">2.2.8 ParallelSpecConfig</h4>

Parallel execution configuration of `modelslim_convert`. The device (NPU/CPU and card index) is determined by the CLI `--device` / `--device_id`; the YAML contains no device field.

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `cpu_workers` | `int` | Optional | `8` | — | Number of parallel workers for CPU conversion: 1 means a single process; greater than 1 means multiple processes across groups plus threads within a group. Takes effect only with `--device cpu`; NPU conversion always uses one subprocess per card (determined by `--device_id`) and does not need this setting. | None |
| `max_group_size` | `int / null` | Optional | `null` | — | Maximum number of tasks in a single dependency group; beyond this the group is split into several sub-groups spread across processes. Unset means no splitting. | None |

**Configuration Constraints**

- `cpu_workers` takes effect only with `--device cpu`. For NPU use `--device npu --device_id ...`.
- `max_group_size`: for models with many expert tasks, such as MoE, set it to roughly "expert tasks per layer / number of cards" (for example 96 for 8 cards) to avoid long-tail waiting on a single process/card for a large group.

<h4 id="2-2-9-convert-defaults">2.2.9 ConvertDefaults</h4>

Global default values used when conversion rules do not explicitly declare fields.

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `src_format` | `string` | Optional | `auto` | `auto` | Source weight format: `auto` infers it from the model adapter or the weight directory. | None |
| `dst_format` | `string` | Optional | `ascendv1` | `ascendv1`, `ascendv1_saver`, `compressed_tensors`, `huggingface`, `hf` | Target save format: `ascendv1` (Ascend; equivalent to `ascend_v1` in `SaveConfig.type`); `compressed_tensors` (HF-compatible safetensors); `huggingface`/`hf` are aliases of `compressed_tensors`. | None |
| `dst_ir` | `string / null` | Optional | `null` | `FLOAT`, `FP8_BLOCK`, `W8A8_MXFP8`, `W4A4_MXFP4`, `W4A8_MXFP8`, `INT4_PACKED`, `NVFP4_MODELOPT`, `HIFP4`, `UNKNOWN`, or null | Target IR type; when unset it is determined by the target format. | None |

**Configuration Constraints**

- None.

## 3. Complete Configuration Reference

```yaml
apiversion: modelslim_convert
spec:
  linears:
  - match:
    - model.language_model.layers.*.mlp.gate_proj
    - model.language_model.layers.*.mlp.up_proj
    - model.language_model.layers.*.mlp.down_proj
    - model.language_model.layers.*.mlp.shared_experts.gate_proj
    - model.language_model.layers.*.mlp.shared_experts.up_proj
    - model.language_model.layers.*.mlp.shared_experts.down_proj
    - model.language_model.layers.*.mlp.experts.*.gate_proj
    - model.language_model.layers.*.mlp.experts.*.up_proj
    - model.language_model.layers.*.mlp.experts.*.down_proj
    - model.language_model.layers.*.self_attn.q_a_proj
    - model.language_model.layers.*.self_attn.q_b_proj
    - model.language_model.layers.*.self_attn.kv_a_proj_with_mqa
    target: W8A8_MXFP8
    route: auto
  save:
  - type: ascend_v1
    part_file_size: 4
  parallel:
    cpu_workers: 8
    max_group_size: null
  defaults:
    src_format: auto
    dst_format: ascendv1
    dst_ir: null
```
