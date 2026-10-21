# AscendV1 Quantization Format Terminology Entry

> **Entry Category**: [Quantization Format](../README.md)<br>
> **English Name**: AscendV1 Quantization Format<br>
> **Application Domain**: Ascend NPU inference, large language model quantization and compression, multimodal understanding model quantization and compression<br>
> **msModelSlim Implementation**: [`msmodelslim/format/ascendV1_format/`](../../../../../msmodelslim/format/ascendV1_format/)

---

## 1. Overview

AscendV1 is the [Quantization Format Terminology Entry](../README.md) of msModelSlim for Ascend NPU inference. Inference frameworks (vLLM Ascend, SGLang, MindIE) identify the quantization type of each tensor through `quant_model_description.json` and load the corresponding parameters from `quant_model_weights*.safetensors`. It solves the problem of unified write-to-disk and loading of quantized weights on the Ascend side. Its core characteristics are covering multiple quantization mode enumerations and aligning with the vLLM Ascend, SGLang, and MindIE loading paths.

For configuration, execution, and deployment steps see the *[AscendV1 Usage Guide](ascendv1_usage.md)*. For the principles and formulas of each quantization mode see *[Quantization Modes](../../quantization_mode/README.md)* and the entry links in the support tables below.

---

## 2. Term Description

### 2.1 Principles

**Core Idea**

AscendV1 is essentially a set of **write-to-disk conventions for quantized models on the Ascend inference side**: it does not perform calibration or fake-quantization computation; instead, at the end of the one-click quantization pipeline, it converts the already-quantized QIR modules into two types of deliverables that inference frameworks can load directly - the **Quantization description file** `quant_model_description.json` and the **Quantization weight file** `quant_model_weights*.safetensors`. The former serves as a tensor-level index for vLLM Ascend, SGLang, and MindIE to identify each tensor name and its quantization mode (such as `W8A8` and `W4A4_MXFP4`); the latter stores the corresponding int8 / FP8 / scale / zero-point values under the same key names. The inference side first reads the description file to determine the loading and operator path, then fetches data from safetensors by key name; the two sets of key names correspond one-to-one, and neither may be missing.

Therefore, users can understand AscendV1 as: **the Ascend standard export package of msModelSlim quantization results** - the algorithms handle quantization computation and parameter generation, while AscendV1 handles weight write-to-disk, metadata description, and alignment with the vLLM Ascend, SGLang, and MindIE loading paths.

**Key Properties**

- Supports distributed export and weight sharding (`part_file_size`).
- Can optionally export the QuaRot rotation matrix.
- Expresses the ability to carry multiple quantization modes through the enumeration values in the description file.

---

## 3. <span id="export-artifacts">Export Artifacts (Deliverables)</span>

### 3.1 Directory and File Description

After running one-click quantization with the AscendV1 write-to-disk format specified (the saver `type` is `ascendv1_saver`), the following files are generated under the specified `save_path` directory:

```bash
├── config.json                            # Original model configuration file
├── generation_config.json                 # Original generation configuration file
├── quant_model_description.json           # Quantization description file
├── quant_model_weights*.safetensors       # Quantization weights: unsharded is quant_model_weights.safetensors; sharded is such as quant_model_weights-00001-of-00003.safetensors ... plus quant_model_weights.safetensors.index.json
├── tokenizer_config.json                  # Original tokenizer configuration file
├── tokenizer.json                         # Original tokenizer vocabulary
├── {model_type}_best_practice.yaml        # Quantization configuration protocol
├── vocab.json                             # Original vocabulary mapping file (some models)
└── optional/                              # Optional export directory (generated when some algorithms are enabled)
    └── quarot.safetensors                 # QuaRot global rotation matrix (generated when export_extra_info is enabled)
```

| File Name | Description |
|--------|------|
| `config.json` | The configuration file of the original model, containing key parameters such as the model architecture, the number of layers, and the hidden dimension |
| `generation_config.json` | The generation configuration file of the original model, containing inference-related parameters such as the sampling strategy and the maximum generation length |
| `quant_model_description.json` | **Quantization description file**, recording the quantization type and metadata of each weight tensor; each tensor key corresponds to one quantization type identifier, and all parameters of the same Linear layer (weight, scale, and so on) share the same type identifier |
| `quant_model_weights*.safetensors` | **Quantization weight file**; unsharded is `quant_model_weights.safetensors`; sharded is in forms such as `quant_model_weights-00001-of-0000N.safetensors`, indexed by `quant_model_weights.safetensors.index.json` |
| `tokenizer_config.json` | The configuration file of the original tokenizer, containing information such as special tokens and the vocabulary size |
| `tokenizer.json` | The vocabulary file of the original tokenizer, defining the mapping between tokens and IDs |
| `{model_type}_best_practice.yaml` | **Quantization configuration protocol file**, recording the complete configuration used in this quantization; see the *[modelslim_v1 Configuration Description](../../../api_reference/config/quant/modelslim_v1.md)* |
| `vocab.json` | The original vocabulary mapping file; some models (such as GPT-style models) include this file |
| `optional/quarot.safetensors` | **Optional export**: the QuaRot global rotation matrix (generated only when QuaRot is used and `export_extra_info: true` is set); see Optional Export below |

### 3.2 quant_model_description.json

**File Structure Example**

```json
{
  "model_quant_type": "W8A8",
  "version": "1.0.0",
  "group_size": 128,
  "kv_quant_type": "KV8",
  "model.layers.0.self_attn.qkv_proj.weight": "W8A8",
  "model.layers.0.self_attn.o_proj.weight": "W8A8",
  "model.layers.0.mlp.gate_proj.weight": "W8A8",
  "model.layers.0.mlp.up_proj.weight": "W8A8",
  "model.layers.0.mlp.down_proj.weight": "W8A8",
  "metadata": {},
  "optional": {}
}
```

> In the example above the `*.weight` field names are determined by the model itself.

**<span id="global-metadata">Global Metadata Fields</span>**

| Field Name | Type | Description |
|--------|------|------|
| `model_quant_type` | string | Overall quantization type of the model (for mixed quantization, the one with the highest priority; see [Mixed Quantization Priority](#quant-type-priority)) |
| `version` | string | Format version; currently `"1.0.0"` |
| `group_size` | int | The group size for group quantization |
| `kv_quant_type` / `kv_cache_type` | string | KV Cache quantization type |
| `fa_quant_type` | string | Flash Attention quantization type |
| `reduce_quant_type` | string | Communication quantization type |
| `metadata` | object | Extended metadata (such as the QuaRot online rotation description) |
| `optional` | object | Optional exported artifacts (such as the QuaRot global rotation matrix path) |

The remaining key-value pairs are `{tensor name}: {quantization type}`, for example `"model.layers.0.self_attn.q_proj.weight": "W8A8"`.

**<span id="quant-type-priority">`model_quant_type` Priority in Mixed Quantization</span>**

The further back in the list, the higher the priority (lower bit widths take precedence; within the same bit width, `W8A8` takes precedence over `W8A8_DYNAMIC` / `W8A8_MIX`). `W4A8_DYNAMIC` does not participate in the selection. Priority from low to high:

1. `FLOAT`
2. `W16A16S`
3. `W8A16`
4. `W8A8_DYNAMIC`
5. `W8A8_MIX`
6. `W8A8`
7. `WFP8AFP8_DYNAMIC`
8. `W8A8_MXFP8`
9. `W4A8_MXFP`
10. `W4A4_DYNAMIC`
11. `W4A4_MXFP4`
12. `W4A4_MXFP4_DUALSCALE`
13. `W4A4_MXFP4_SVD` (a suffix marker appended after the underlying quantization type when SVDQuant residuals are exported; not a separate deliverable subsection)

### 3.3 <span id="optional-quarot">Optional Export: QuaRot-Related Files</span>

When the pipeline enables QuaRot and `export_extra_info: true` is configured, AscendV1 can additionally write the rotation matrix file and register its path in the description file. The algorithm itself is covered by the corresponding algorithm entry; only the **format-side** write-to-disk conventions are described here.

```bash
optional/
└── quarot.safetensors
```

| Key Name | Data Type | Description |
|------|----------|------|
| `global_rotation` | float32 | Global rotation matrix |

When `online` is enabled, `quant_model_description.json` contains `metadata.quarot` (a layer list and the online rotation description), as shown below:

```json
{
  "metadata": {
    "quarot": {
      "heads_rotation": {
        "layers": ["model.layers.0.self_attn.o_proj", "model.layers.1.self_attn.o_proj"]
      },
      "kronecker_rotation": {
        "layers": ["model.layers.0.mlp.down_proj"]
      }
    }
  }
}
```

When `export_extra_info` is enabled, it contains:

```jsonc
{
  "optional": {
    "quarot": {
      "rotation_map": {
        "global_rotation": "optional/quarot.safetensors"
      }
    }
  }
}
```

The inference framework loads the matrix file according to `optional.quarot.rotation_map`.

---

## 4. <span id="engine-support">Inference Engine Support</span>

All format enumerations in the table **can be written to disk by AscendV1**; the table describes whether the artifact can be loaded and deployed by the target inference engine. The reference basis is the notes of the *[Large Model Support Matrix](../../model/README.md)* and the lab_practice validation labels; specific model x mode x engine combinations are subject to the support matrix and official best practices, and are constrained by CANN / engine versions and hardware generations.

| Format Enumeration | vLLM Ascend | SGLang | MindIE | Description |
| --- | --- | --- | --- | --- |
| `FLOAT` | √ | √ | √ | Unquantized tensors, loaded together with the model |
| `W8A8` | √ | √ | √ | One of the common entry points for Atlas A2 products / Atlas A3 products / Ascend 950PR&950DT products |
| `W8A8_DYNAMIC` | √ | √ | √ | One of the common entry points for Atlas A2 products / Atlas A3 products / Ascend 950PR&950DT products |
| `W4A8_DYNAMIC` | √ | √ | √ | Commonly used when memory is tighter |
| `W4A4_DYNAMIC` | √ | √ | √ | A lower-bit INT path |
| `W8A8_MXFP8` | √ | √ | √ | **Recommended for Ascend 950PR&950DT products**; usually unavailable for Atlas A2 products / Atlas A3 products |
| `W4A8_MXFP` | √ | √ | √ | **Recommended for Ascend 950PR&950DT products** |
| `W4A4_MXFP4` | √ | √ | √ | **Recommended for Ascend 950PR&950DT products** |
| `W4A4_MXFP4_DUALSCALE` | √ | √ | √ | **Recommended for Ascend 950PR&950DT products** |
| `WFP8AFP8_DYNAMIC` | √ | — | — | vLLM Ascend only; **recommended for Ascend 950PR&950DT products** |
| `W8A16` | — | — | √ | MindIE only |
| `W8A8_MIX` | — | — | √ | PD-Mix; MindIE only |
| `W16A16S` | — | — | √ | Sparse quantization; MindIE only |
| `C8` | — | — | √ | KV Cache; MindIE only (including combinations such as w8a8c8 / w4a8c8) |
| `FAQuant` | — | — | √ | FA3 and so on; MindIE only |

> **Legend**: `√` means the engine has a loadable path or validated practice; `—` means it is currently not a recommended deployment path for that engine. When selecting, first narrow the candidate enumerations by engine, then check the AscendV1 deliverable fields in [Quantization Mode Support](#mode-support) below.

---

## 5. <span id="mode-support">Quantization Mode Support</span>

> **Deliverable notes**: In the table, "Deliverable: quantization description JSON" corresponds to `quant_model_description.json`; "Deliverable: quantization weight file" corresponds to `quant_model_weights*.safetensors`. For the mode principles see the *[Quantization Modes](../../quantization_mode/README.md)* entry; this table does not elaborate the dequantization formulas or NPU operators. The deliverables of this entry are described in two columns: the description JSON key values and the safetensors tensor fields.

| Format Enumeration | Whether AscendV1 Supports Write-to-Disk | Quantization Mode Entry | Deliverable: Quantization Description JSON | Deliverable: Quantization Weight Files |
| --- | --- | --- | --- | --- |
| `FLOAT` | Supported | [Quantization Mode Overview](../../quantization_mode/README.md#modes) | [FLOAT Description Keys](#desc-float) | [FLOAT Weight Tensors](#st-float) |
| `W16A16S` | Supported | [Floating-Point Sparse Quantization](../../quantization_mode/linear_layer_quantization/term_float_sparse.md) | [W16A16S Description Keys](#desc-w16a16s) | [W16A16S Weight Tensors](#st-w16a16s) |
| `W8A8` | Supported | [W8A8 Static Quantization](../../quantization_mode/linear_layer_quantization/term_w8a8_static.md) | [W8A8 Description Keys](#desc-w8a8) | [W8A8 Weight Tensors](#st-w8a8) |
| `W8A8_DYNAMIC` | Supported | [W8A8 Dynamic Quantization](../../quantization_mode/linear_layer_quantization/term_w8a8_dynamic.md) | [W8A8_DYNAMIC Description Keys](#desc-w8a8-dynamic) | [W8A8_DYNAMIC Weight Tensors](#st-w8a8-dynamic) |
| `W8A8_MIX` | Supported | [W8A8 PD-Mix](../../quantization_mode/linear_layer_quantization/term_w8a8_pdmix.md) | [W8A8_MIX Description Keys](#desc-w8a8-mix) | [W8A8_MIX Weight Tensors](#st-w8a8-mix) |
| `W8A16` | Supported | [W8A16 Static Quantization](../../quantization_mode/linear_layer_quantization/term_w8a16_static.md) | [W8A16 Description Keys](#desc-w8a16) | [W8A16 Weight Tensors](#st-w8a16) |
| `W4A4_DYNAMIC` | Supported | [W4A4 Dynamic Quantization](../../quantization_mode/linear_layer_quantization/term_w4a4_dynamic.md) | [W4A4_DYNAMIC Description Keys](#desc-w4a4-dynamic) | [W4A4_DYNAMIC Weight Tensors](#st-w4a4-dynamic) |
| `W4A8_DYNAMIC` | Supported | [W4A8 Dynamic Quantization](../../quantization_mode/linear_layer_quantization/term_w4a8_dynamic.md) | [W4A8_DYNAMIC Description Keys](#desc-w4a8-dynamic) | [W4A8_DYNAMIC Weight Tensors](#st-w4a8-dynamic) |
| `WFP8AFP8_DYNAMIC` | Supported | [W8A8 FP8 Dynamic Quantization](../../quantization_mode/linear_layer_quantization/term_w8a8_fp8_dynamic.md) | [WFP8AFP8_DYNAMIC Description Keys](#desc-wfp8afp8-dynamic) | [WFP8AFP8_DYNAMIC Weight Tensors](#st-wfp8afp8-dynamic) |
| `W8A8_MXFP8` | Supported | [W8A8 MX Dynamic Quantization](../../quantization_mode/linear_layer_quantization/term_w8a8_mx_dynamic.md) | [W8A8_MXFP8 Description Keys](#desc-mxfp) | [W8A8_MXFP8 Weight Tensors](#st-mxfp) |
| `W4A8_MXFP` | Supported | [W4A8 MX Dynamic Quantization](../../quantization_mode/linear_layer_quantization/term_w4a8_mx_dynamic.md) | [W4A8_MXFP Description Keys](#desc-mxfp) | [W4A8_MXFP Weight Tensors](#st-mxfp) |
| `W4A4_MXFP4` | Supported | [W4A4 MX Dynamic Quantization](../../quantization_mode/linear_layer_quantization/term_w4a4_mx_dynamic.md) | [W4A4_MXFP4 Description Keys](#desc-mxfp) | [W4A4_MXFP4 Weight Tensors](#st-mxfp) |
| `W4A4_MXFP4_DUALSCALE` | Supported | [W4A4 MX Dual Scale](../../quantization_mode/linear_layer_quantization/term_w4a4_mx_dualscale.md) | [W4A4_MXFP4_DUALSCALE Description Keys](#desc-mxfp-dualscale) | [W4A4_MXFP4_DUALSCALE Weight Tensors](#st-mxfp-dualscale) |
| `C8` | Supported | [KVCache-PerChannel](../../quantization_mode/kv_cache_quantization/term_kv_cache_perchannel.md) | [C8 Description Keys](#desc-c8) | [C8 Weight Tensors](#st-c8) |
| `FAQuant` | Supported | [FA Quantization](../../quantization_mode/fa_quantization/README.md) | [FAQuant Description Keys](#desc-faquant) | [FAQuant Weight Tensors](#st-faquant) |

---

## 6. Deliverable Formats of Each Quantization Mode

Conventions:

- `{prefix}` is the module prefix (for example `model.layers.0.self_attn.q_proj`).
- **Description JSON**: the file `quant_model_description.json`; keys are full tensor names and values are quantization type strings (consistent with the enumerations); for global fields see [Global Metadata Fields](#global-metadata) above.
- **safetensors**: the file `quant_model_weights*.safetensors` (may be sharded); keys are `{prefix}.<param>` and store the actual numeric tensors.

### 6.1 FLOAT

**<span id="desc-float">quant_model_description.json</span>**

| Description Key | Value | Description |
| --- | --- | --- |
| `{prefix}.weight` | `"FLOAT"` | Unquantized weight |
| `{prefix}.bias` | `"FLOAT"` | Bias (if present) |

The overall `model_quant_type` can be `"FLOAT"` when only FLOAT is present (for mixed quantization it is selected according to the [priority](#quant-type-priority)).

**<span id="st-float">`quant_model_weights*.safetensors`</span>**

| Tensor Name | Data Type | Description |
| --- | --- | --- |
| `{prefix}.weight` | float16 / bfloat16 | Original floating-point weight |
| `{prefix}.bias` | float16 / bfloat16 | Bias (optional) |

### 6.2 W16A16S

**<span id="desc-w16a16s">quant_model_description.json</span>**

| Description Key | Value | Description |
| --- | --- | --- |
| `{prefix}.weight` | `"W16A16S"` | Sparse weight |
| `{prefix}.scale` | `"W16A16S"` | Sparse scaling factor |

**<span id="st-w16a16s">`quant_model_weights*.safetensors`</span>**

| Tensor Name | Data Type | Description |
| --- | --- | --- |
| `{prefix}.weight` | float16 / bfloat16 | Weight after sparse processing |
| `{prefix}.scale` | float16 / bfloat16 | Scaling factor |

### 6.3 W8A8

**<span id="desc-w8a8">quant_model_description.json</span>**

The following keys under the same Linear layer share the type `"W8A8"` (`bias` can be marked `"FLOAT"` if it remains floating point):

| Description Key | Value | Description |
| --- | --- | --- |
| `{prefix}.weight` | `"W8A8"` | Quantized weight |
| `{prefix}.quant_bias` | `"W8A8"` | Quantization bias |
| `{prefix}.input_scale` | `"W8A8"` | Activation scale |
| `{prefix}.input_offset` | `"W8A8"` | Activation zero-point |
| `{prefix}.deq_scale` | `"W8A8"` | Combined dequantization scale |
| `{prefix}.bias` | `"FLOAT"` or `"W8A8"` | Original floating-point bias (optional) |

**<span id="st-w8a8">`quant_model_weights*.safetensors`</span>**

| Tensor Name | Data Type | Description |
| --- | --- | --- |
| `{prefix}.weight` | int8 | Quantized weight |
| `{prefix}.quant_bias` | int32 | Quantization bias |
| `{prefix}.input_scale` | float32 | Activation quantization scale |
| `{prefix}.input_offset` | float32 | Activation quantization zero-point |
| `{prefix}.deq_scale` | int64 / float32 | Combined dequantization scale (for bfloat16 models it is usually float32; otherwise it is commonly stored as the int64 type per the operator convention) |
| `{prefix}.bias` | float32 | Original floating-point bias (optional) |

### 6.4 W8A8_DYNAMIC

**<span id="desc-w8a8-dynamic">quant_model_description.json</span>**

| Description Key | Value | Description |
| --- | --- | --- |
| `{prefix}.weight` | `"W8A8_DYNAMIC"` | Quantized weight |
| `{prefix}.weight_scale` | `"W8A8_DYNAMIC"` | Weight quantization scale |
| `{prefix}.weight_offset` | `"W8A8_DYNAMIC"` | Weight quantization zero-point |
| `{prefix}.bias` | `"FLOAT"` or `"W8A8_DYNAMIC"` | Bias (optional) |

Activation dynamic parameters are not written to disk, so the description has no static activation keys such as `input_scale` / `input_offset`.

**<span id="st-w8a8-dynamic">`quant_model_weights*.safetensors`</span>**

| Tensor Name | Data Type | Description |
| --- | --- | --- |
| `{prefix}.weight` | int8 | Quantized weight |
| `{prefix}.weight_scale` | float32 | Weight quantization scale |
| `{prefix}.weight_offset` | float32 | Weight quantization zero-point (0 when symmetric) |
| `{prefix}.bias` | float32 | Original floating-point bias (optional) |

Activation quantization parameters are computed dynamically at inference time and are **not written** to the weight file.

### 6.5 W8A8_MIX

**<span id="desc-w8a8-mix">quant_model_description.json</span>**

| Description Key | Value | Description |
| --- | --- | --- |
| `{prefix}.weight` | `"W8A8_MIX"` | Quantized weight |
| `{prefix}.quant_bias` | `"W8A8_MIX"` | Quantization bias |
| `{prefix}.input_scale` | `"W8A8_MIX"` | Activation scale |
| `{prefix}.input_offset` | `"W8A8_MIX"` | Activation zero-point |
| `{prefix}.deq_scale` | `"W8A8_MIX"` | Combined dequantization scale |
| `{prefix}.weight_scale` | `"W8A8_MIX"` | Weight quantization scale |
| `{prefix}.weight_offset` | `"W8A8_MIX"` | Weight quantization zero-point |
| `{prefix}.bias` | `"FLOAT"` or `"W8A8_MIX"` | Bias (optional) |

**<span id="st-w8a8-mix">`quant_model_weights*.safetensors`</span>**

The union of the W8A8 static activation-related fields and the W8A8_DYNAMIC weight quantization fields:

| Tensor Name | Data Type | Description |
| --- | --- | --- |
| `{prefix}.weight` | int8 | Quantized weight |
| `{prefix}.quant_bias` | int32 | Quantization bias |
| `{prefix}.input_scale` | float32 | Activation quantization scale |
| `{prefix}.input_offset` | float32 | Activation quantization zero-point |
| `{prefix}.deq_scale` | int64 / float32 | Combined dequantization scale |
| `{prefix}.weight_scale` | float32 | Weight quantization scale |
| `{prefix}.weight_offset` | float32 | Weight quantization zero-point |
| `{prefix}.bias` | float32 | Original floating-point bias (optional) |

### 6.6 W8A16

**<span id="desc-w8a16">quant_model_description.json</span>**

| Description Key | Value | Description |
| --- | --- | --- |
| `{prefix}.weight` | `"W8A16"` | Quantized weight |
| `{prefix}.weight_scale` | `"W8A16"` | Weight quantization scale |
| `{prefix}.weight_offset` | `"W8A16"` | Weight quantization zero-point |
| `{prefix}.bias` | `"FLOAT"` or `"W8A16"` | Bias (optional) |

**<span id="st-w8a16">`quant_model_weights*.safetensors`</span>**

| Tensor Name | Data Type | Description |
| --- | --- | --- |
| `{prefix}.weight` | int8 | Quantized weight |
| `{prefix}.weight_scale` | float32 | Weight quantization scale |
| `{prefix}.weight_offset` | float32 | Weight quantization zero-point |
| `{prefix}.bias` | float32 | Original floating-point bias (optional) |

### 6.7 W4A4_DYNAMIC

**<span id="desc-w4a4-dynamic">quant_model_description.json</span>**

| Description Key | Value | Description |
| --- | --- | --- |
| `{prefix}.weight` | `"W4A4_DYNAMIC"` | Quantized weight (packed) |
| `{prefix}.weight_scale` | `"W4A4_DYNAMIC"` | Weight quantization scale |
| `{prefix}.weight_offset` | `"W4A4_DYNAMIC"` | Weight quantization zero-point |
| `{prefix}.bias` | `"FLOAT"` or `"W4A4_DYNAMIC"` | Bias (optional) |

**<span id="st-w4a4-dynamic">`quant_model_weights*.safetensors`</span>**

| Tensor Name | Data Type | Description |
| --- | --- | --- |
| `{prefix}.weight` | int8 | int4 packed storage |
| `{prefix}.weight_scale` | float32 | Weight quantization scale |
| `{prefix}.weight_offset` | float32 | Weight quantization zero-point |
| `{prefix}.bias` | float32 | Original floating-point bias (optional) |

Activation quantization parameters are computed dynamically at inference time and are not written to the weight file.

### 6.8 W4A8_DYNAMIC

**<span id="desc-w4a8-dynamic">quant_model_description.json</span>**

| Description Key | Value | Description |
| --- | --- | --- |
| `{prefix}.weight` | `"W4A8_DYNAMIC"` | Quantized weight (packed) |
| `{prefix}.weight_scale` | `"W4A8_DYNAMIC"` | Weight quantization scale |
| `{prefix}.weight_offset` | `"W4A8_DYNAMIC"` | Weight quantization zero-point |
| `{prefix}.scale_bias` | `"W4A8_DYNAMIC"` | Extra adjustment factor for dequantization |
| `{prefix}.bias` | `"FLOAT"` or `"W4A8_DYNAMIC"` | Bias (optional) |

**<span id="st-w4a8-dynamic">`quant_model_weights*.safetensors`</span>**

| Tensor Name | Data Type | Description |
| --- | --- | --- |
| `{prefix}.weight` | int8 | int4 packed storage |
| `{prefix}.weight_scale` | float32 | Weight quantization scale |
| `{prefix}.weight_offset` | float32 | Weight quantization zero-point |
| `{prefix}.scale_bias` | float32 | Extra adjustment factor for dequantization |
| `{prefix}.bias` | float32 | Original floating-point bias (optional) |

### 6.9 WFP8AFP8_DYNAMIC

**<span id="desc-wfp8afp8-dynamic">quant_model_description.json</span>**

| Description Key | Value | Description |
| --- | --- | --- |
| `{prefix}.weight` | `"WFP8AFP8_DYNAMIC"` | FP8 weight |
| `{prefix}.weight_scale` | `"WFP8AFP8_DYNAMIC"` | Weight quantization scale |
| `{prefix}.weight_offset` | `"WFP8AFP8_DYNAMIC"` | Weight quantization zero-point |
| `{prefix}.bias` | `"FLOAT"` or `"WFP8AFP8_DYNAMIC"` | Bias (optional) |

**<span id="st-wfp8afp8-dynamic">`quant_model_weights*.safetensors`</span>**

| Tensor Name | Data Type | Description |
| --- | --- | --- |
| `{prefix}.weight` | float8_e4m3fn | FP8 weight |
| `{prefix}.weight_scale` | float32 | Weight quantization scale |
| `{prefix}.weight_offset` | float32 | Weight quantization zero-point |
| `{prefix}.bias` | float32 | Original floating-point bias (optional) |

### 6.10 W8A8_MXFP8 / W4A8_MXFP / W4A4_MXFP4

**<span id="desc-mxfp">quant_model_description.json</span>**

The description key values are `"W8A8_MXFP8"` / `"W4A8_MXFP"` / `"W4A4_MXFP4"` respectively (consistent with the specific enumeration):

| Description Key | Value | Description |
| --- | --- | --- |
| `{prefix}.weight` | Corresponding MXFP enumeration | Quantized weight |
| `{prefix}.weight_scale` | Corresponding MXFP enumeration | block-wise scale |
| `{prefix}.bias` | `"FLOAT"` or the corresponding enumeration | Bias (optional) |

**<span id="st-mxfp">`quant_model_weights*.safetensors`</span>**

| Tensor Name | Data Type | Description |
| --- | --- | --- |
| `{prefix}.weight` | float8_e4m3fn or uint8 (packed fp4) | Quantized weight |
| `{prefix}.weight_scale` | uint8 | block-wise scale (commonly stored **with a +127 offset** during export, ranging from 0 to 255) |
| `{prefix}.bias` | float32 | Original floating-point bias (optional) |

### 6.11 W4A4_MXFP4_DUALSCALE

**<span id="desc-mxfp-dualscale">quant_model_description.json</span>**

Building on the [MXFP description fields](#desc-mxfp), the values are all `"W4A4_MXFP4_DUALSCALE"`, with the following additions:

| Description Key | Value | Description |
| --- | --- | --- |
| `{prefix}.weight_dual_scale` | `"W4A4_MXFP4_DUALSCALE"` | Second scale path |

**<span id="st-mxfp-dualscale">`quant_model_weights*.safetensors`</span>**

Building on the [MXFP weight fields](#st-mxfp), it additionally contains:

| Tensor Name | Data Type | Description |
| --- | --- | --- |
| `{prefix}.weight_dual_scale` | float32 | Second scale path |

### 6.12 C8

**<span id="desc-c8">quant_model_description.json</span>**

| Description Key / Global Field | Value | Description |
| --- | --- | --- |
| `{prefix}.kv_cache_scale` | `"C8"` | KV Cache scale |
| `{prefix}.kv_cache_offset` | `"C8"` | KV Cache zero-point |
| `kv_quant_type` / `kv_cache_type` | such as `"C8"` / `"KV8"` | Global KV quantization type (written when consistent with the export implementation) |

**<span id="st-c8">`quant_model_weights*.safetensors`</span>**

| Tensor Name | Data Type | Description |
| --- | --- | --- |
| `{prefix}.kv_cache_scale` | float32 / float16 | KV Cache quantization scale |
| `{prefix}.kv_cache_offset` | float32 / float16 | KV Cache quantization zero-point |

The specific `{prefix}` depends on the naming of the attention KV-related modules.

### 6.13 FAQuant

> **Naming conventions**: The `{prefix}` in this section refers to the full name of the **attention module** (such as `model.layers.0.self_attn`), which differs from the Linear `{prefix}` above (such as `...self_attn.q_proj`). The Q/K/V branches of FA quantization are written to disk as the `fa_q` / `fa_k` / `fa_v` sub-keys rather than a single `{prefix}.scale` / `{prefix}.offset`. For the holistic modes of each FA variant see *[FA Quantization](../../quantization_mode/fa_quantization/README.md)*.

**<span id="desc-faquant">quant_model_description.json</span>**

| Description Key / Global Field | Value | Description |
| --- | --- | --- |
| `{prefix}.fa_q.scale` | `"FAQuant"` | Q activation quantization scale (written when this branch uses static quantization) |
| `{prefix}.fa_q.offset` | `"FAQuant"` | Q activation quantization zero-point (written when this branch uses static quantization) |
| `{prefix}.fa_k.scale` | `"FAQuant"` | K activation quantization scale (written when this branch uses static quantization) |
| `{prefix}.fa_k.offset` | `"FAQuant"` | K activation quantization zero-point (written when this branch uses static quantization) |
| `{prefix}.fa_v.scale` | `"FAQuant"` | V activation quantization scale (written when this branch uses static quantization) |
| `{prefix}.fa_v.offset` | `"FAQuant"` | V activation quantization zero-point (written when this branch uses static quantization) |
| `{prefix}.quant_type` | string | The FA quantization type string of this attention layer; for the values see the rules below, for example `INT8`, `FP8_DYNAMIC`, `Q_INT8_DYNAMIC_KV_INT8`, and `QK_MXFP8_DYNAMIC_V_MXFP8_PER_CHANNEL` |
| `fa_quant_type` | string | Global FA quantization type (written when FA quantization is enabled, such as `"FAQuant"`) |

**`{prefix}.quant_type` Value Rules**:

| Rule | Description | Example |
| --- | --- | --- |
| Data type name | The type string uses the data type name: FP8 (E4M3) is `FP8`, INT8 is `INT8`, MXFP4 is `MXFP4`, and MXFP8 is `MXFP8` | — |
| Dynamic marker | Dynamic (per-token/per-block) appends `_DYNAMIC` after the type name; static does not | per-token dynamic INT8 is written as `INT8_DYNAMIC` |
| per-channel marker | per-channel static appends `_PER_CHANNEL` after the type name to distinguish it from per-head static | per-channel static MXFP8 is written as `MXFP8_PER_CHANNEL` |
| Prefix rule | Branches using the same quantization method are merged, and prefixes are concatenated in Q/K/V/P order; the prefix is omitted when all three QKV branches are identical | Q and K using the same quantization method are written as `QK_...`; the prefix is omitted when QKV are identical |
| Multi-group concatenation | Groups with different quantization methods are connected with `_` | `Q_INT8_DYNAMIC_KV_INT8`, `QK_MXFP8_DYNAMIC_V_MXFP8_PER_CHANNEL` |

The correspondence between validated schemes and `{prefix}.quant_type`:

| `{prefix}.quant_type` | Corresponding Quantization Mode |
| --- | --- |
| `INT8` | [FA INT8 PerHead Quantization](../../quantization_mode/fa_quantization/term_fa_int8_perhead.md) |
| `INT8_DYNAMIC` | [FA INT8 Dynamic Quantization](../../quantization_mode/fa_quantization/term_fa_int8_dynamic.md) |
| `FP8_DYNAMIC` | [FA FP8 Dynamic Quantization](../../quantization_mode/fa_quantization/term_fa_fp8_dynamic.md) |
| `MXFP4_DYNAMIC` | [FA MXFP4 Dynamic Quantization](../../quantization_mode/fa_quantization/term_fa_mxfp4_dynamic.md) |
| `Q_INT8_DYNAMIC_KV_INT8` | [FA Q-INT8 Dynamic K/V-INT8 Static Quantization](../../quantization_mode/fa_quantization/term_fa_q_int8_dynamic_kv_int8.md) |
| `Q_FP8_DYNAMIC_KV_FP8` | [FA Q-FP8 Dynamic K/V-FP8 Static Quantization](../../quantization_mode/fa_quantization/term_fa_q_fp8_dynamic_kv_fp8.md) |
| `QK_MXFP8_DYNAMIC_V_MXFP8_PER_CHANNEL` | [FA QK-MXFP8 Dynamic / V-MXFP8 PerChannel Static Quantization](../../quantization_mode/fa_quantization/term_fa_qk_mxfp8_dynamic_v_mxfp8_perchannel.md) |

Static branches write `scale` / `offset` to disk: per-head static writes a float32 scale plus a zero-valued zero-point - int8 zero for INT8 schemes and float32 zero for FP8 schemes. per-channel static (MXFP8) encodes the E8M0 shared exponent as uint8 (exp + 127) and writes it as the scale, together with a uint8 zero-valued zero-point. The activations of dynamic branches (per-token / per-block) do not write `scale` / `offset` to disk; only the `{prefix}.quant_type` above is written. For the algorithm configuration see *[FA3 Quant](../../quantization_algorithms/fa3_quant/term_fa3_quant.md)*.

**<span id="st-faquant">`quant_model_weights*.safetensors`</span>**

| Tensor Name | Data Type | Description |
| --- | --- | --- |
| `{prefix}.fa_q.scale` | float32 or uint8 | Static scale of Q: float32 for per-head, uint8 for per-channel (E8M0 exponent exp + 127) |
| `{prefix}.fa_q.offset` | int8, float32, or uint8 | Static zero-point of Q: int8 (INT8) or float32 (FP8) zero for per-head, uint8 zero for per-channel |
| `{prefix}.fa_k.scale` | float32 or uint8 | Static scale of K: float32 for per-head, uint8 for per-channel (E8M0 exponent exp + 127) |
| `{prefix}.fa_k.offset` | int8, float32, or uint8 | Static zero-point of K: int8 (INT8) or float32 (FP8) zero for per-head, uint8 zero for per-channel |
| `{prefix}.fa_v.scale` | float32 or uint8 | Static scale of V: float32 for per-head, uint8 for per-channel (E8M0 exponent exp + 127) |
| `{prefix}.fa_v.offset` | int8, float32, or uint8 | Static zero-point of V: int8 (INT8) or float32 (FP8) zero for per-head, uint8 zero for per-channel |

---

## 7. Applicable Scenarios and Limitations

### 7.1 Applicable Scenarios

- Deploying LLM / multimodal understanding quantized weights with vLLM Ascend, SGLang, and MindIE on the Ascend side.
- Writing to disk with multiple quantization type enumerations carried in the same description file.

### 7.2 Usage Limitations

- Not applicable to the generic vLLM path aimed only at the HF `quantization_config` / compressed-tensors.
- Whether a specific quantization mode is available depends on CANN, the inference framework version, and the model best practice.

---

## 8. Related Workflows

- *[AscendV1 Usage Guide](ascendv1_usage.md)*: confirm mode support, configure, and run.
- *[One-Click Quantization Usage Guide](../../../user_guide/usage_one_click_quantization.md)*: commands and configuration protocol.
- *[Quantization Format Integration Guide](../iformat_integration_guide.md)*: a reference for developing a new format.

---

## 9. Related Terms

- [Quantization Format Terminology Entry](../README.md): the parent concept; the directory this entry belongs to.
- [compressed-tensors Quantization Format Terminology Entry](../compressed_tensors/term_compressed_tensors.md): other; a peer write-to-disk protocol that also belongs to quantization formats.
- [MindIE SD Quantization Format Terminology Entry](../mindie_sd/term_mindie_sd.md): other; a peer write-to-disk protocol that also belongs to quantization formats.
- [Quantization Modes](../../quantization_mode/README.md): a companion term; the deliverable enumerations of this format correspond to the quantization modes.
