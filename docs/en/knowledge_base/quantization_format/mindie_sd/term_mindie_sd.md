# MindIE SD Quantization Format Terminology Entry

> **Entry Category**: [Quantization Format](../README.md)<br>
> **English Name**: MindIE SD Quantization Format<br>
> **Application Domain**: Multimodal generation model quantization and compression, MindIE SD deployment<br>
> **msModelSlim Implementation**: [`msmodelslim/format/mindie_format/`](../../../../../msmodelslim/format/mindie_format/)

---

## 1. Overview

MindIE SD is the [Quantization Format Terminology Entry](../README.md) of msModelSlim for **multimodal generation** scenarios, consumed by MindIE SD. It is enabled in one-click quantization through the saver type `mindie_format_saver`. It solves the problem of aligning the quantized weights of multimodal generation models such as diffusion / DiT models with the MindIE SD loading conventions. Its core feature is that it works together with the `multimodal_sd_modelslim_v1` configuration protocol and writes to disk as `quant_model_description*.json` + `quant_model_weight*.safetensors`.

For configuration, execution, and deployment steps see the *[MindIE SD Usage Guide](mindie_sd_usage.md)*. For the principles and formulas of each quantization mode see *[Quantization Modes](../../quantization_mode/README.md)* and the entry links in the support tables below.

---

## 2. Term Description

### 2.1 Principles

**Core Idea**

MindIE SD is essentially a set of **write-to-disk conventions for quantized models intended for multimodal generation and consumed by the MindIE SD inference engine**: it does not perform calibration or fake-quantization computation; instead, at the end of the `multimodal_sd_modelslim_v1` quantization pipeline, it converts the already-quantized modules into two types of deliverables that the MindIE SD inference engine can load directly - the **Quantization description file** `quant_model_description*.json` and the **Quantization weight file** `quant_model_weight*.safetensors` (note that the weight file name uses the singular `weight`, unlike AscendV1's `quant_model_weights`). The former serves as a tensor-level index that lets the MindIE SD inference engine identify each tensor name and its quantization mode; the latter stores the corresponding quantization parameters under the same key names. At inference time the description file is read first, and the values are then fetched from safetensors by key name.

Therefore, users can understand MindIE SD as: **the MindIE SD inference engine standard export package of multimodal generation quantization results** - the algorithms and the multimodal adapter handle quantization and calibration orchestration, while MindIE SD handles the write-to-disk and description aligned with the MindIE SD inference engine loading path.

**Key Properties**

- Targets multimodal generation (such as Wan2.2) rather than the generic LLM AscendV1 default path.
- In YAML, `type` is fixed to `mindie_format_saver`.
- Supports sharding by `part_file_size`; `0` means no sharding.
- Usually appears together with `multimodal_sd_config` (dump / inference_config).
- The enumeration values in the description file express the ability to carry multiple quantization modes; modes whose handler is not implemented must switch to *[AscendV1](../ascendv1/term_ascendv1.md)*.

---

## 3. <span id="export-artifacts">Export Artifacts (Deliverables)</span>

### 3.1 Directory and File Description

After running one-click quantization with the MindIE SD write-to-disk format specified (the saver `type` is `mindie_format_saver`), the deliverables are located under `save_path` (in distributed scenarios they may be placed in `rank_*` subdirectories). A generic single-model directory example is as follows:

```bash
├── quant_model_description*.json              # Quantization weight description (with or without the quantization type suffix)
├── quant_model_weight*.safetensors            # Quantization weight (singular weight; may be sharded / may carry the quantization type suffix)
├── quant_model_weight*.safetensors.index.json # Index generated when sharding (optional)
├── *.json / *.py                              # Configuration and code copied from the source model (excluding index.json)
└── (optional) calib_data_*.pth                # Dump calibration data (when enable_dump is enabled, the directory is determined by dump_data_dir)
```

For dual noise expert structures such as Wan2.2, the description file and the quantization weight are usually placed in the `low_noise_model/` and `high_noise_model/` subdirectories respectively, for example:

```bash
├── low_noise_model/
│   ├── quant_model_description*.json
│   └── quant_model_weight*.safetensors
└── high_noise_model/
    ├── quant_model_description*.json
    └── quant_model_weight*.safetensors
```

| File Name | Description |
| --- | --- |
| `quant_model_description*.json` | **Quantization weight description file**, recording the tensor quantization type and metadata; it may also carry a `{quant_type}` suffix |
| `quant_model_weight*.safetensors` | **Quantization weight file** (singular `weight`, unlike AscendV1's plural `weights`); it may be sharded when large and indexed through index.json |
| `*.json` / `*.py` | Configuration and code files copied from the source model; `index.json` is **not copied**; permissions are set according to the tool conventions |
| `calib_data_*.pth` | **Optional**: calibration dump data; see Optional Export below |

In `quant_model_description*.json`, each tensor key corresponds to one quantization type identifier; the related parameters of the same Linear layer usually share the same type identifier.

### 3.2 quant_model_description.json

**File Structure Example**

```json
{
  "model_quant_type": "W8A8",
  "group_size": 32,
  "blocks.0.self_attn.q.weight": "W8A8",
  "blocks.0.self_attn.q.input_scale": "W8A8",
  "blocks.0.self_attn.q.input_offset": "W8A8",
  "blocks.0.self_attn.q.deq_scale": "W8A8",
  "blocks.0.self_attn.q.quant_bias": "W8A8",
  "blocks.0.self_attn.q.bias": "FLOAT"
}
```

> Tensor key names are determined by the model adapter (multimodal generation models commonly use the `blocks.*` prefix rather than the `model.layers.*` of LLMs); when FA3 and so on are enabled, fields such as `fa_quant_type` and layer-level `quant_type` may also appear.

**<span id="global-metadata">Global Metadata Fields</span>**

| Field Name | Type | Description |
| ------------------ | ------ | ------------------------------------------------ |
| `model_quant_type` | string | Overall quantization type of the model (written by the handler, such as `"W8A8"` and `"W8A8_MXFP8"`) |
| `group_size` | int | The group / block related size; the default is commonly `32` in MXFP and similar scenarios |
| `fa_quant_type` | string | Flash Attention related quantization type (written when FA3 and so on are enabled) |
| `{tensor name}` | string | The quantization type enumeration that each tensor belongs to |

For the complete multimodal configuration protocol see the *[multimodal_sd_modelslim_v1 Configuration Description](../../../api_reference/config/quant/multimodal_sd_modelslim_v1.md)*.

### 3.3 <span id="optional-dump">Optional Export: Calibration Dump Data</span>

When `multimodal_sd_config.dump_config.enable_dump` is true, the calibration pth can be written under `dump_data_dir` (if it is empty, `save_path` is used), for example:

```bash
calib_data_<task_config>_low_noise_model.pth
calib_data_<task_config>_high_noise_model.pth
```

For the meaning of the fields see [DumpConfig](../../../api_reference/config/quant/multimodal_sd_modelslim_v1.md#2-2-5-dump-config). This directory is auxiliary data for the quantization process and is **not necessarily** delivered together with the MindIE deployment directory.

---

## 4. <span id="engine-support">Inference Engine Support</span>

All format enumerations in the table **can be written to disk by MindIE SD**; the table describes whether the artifact can be loaded by the **MindIE multimodal generation** path. The reference basis is the *[Large Model Support Matrix](../../model/README.md)* and the lab_practice validation labels; specific model x mode x MindIE SD version combinations are subject to the support matrix and official best practices.

| Format Enumeration | MindIE SD | Description |
| --- | --- | --- |
| `FLOAT` | √ | Unquantized tensors, loaded together with the model |
| `W8A8` | √ | Static W8A8 |
| `W8A8_DYNAMIC` | √ | Dynamic W8A8 |
| `W8A8_MXFP8` | √ | MXFP8; commonly used in generation best practices |
| `W4A4_MXFP4` | √ | MXFP4 |
| `W4A4_MXFP4_DUALSCALE` | √ | MXFP4 Dual Scale |
| `FAQuant` | √ | FA3 and so on; often combined with linear layer quantization |

> **Legend**: `√` means the engine has a loadable path or validated practice. For the generic LLM paths vLLM Ascend / SGLang / MindIE, use *[AscendV1](../ascendv1/term_ascendv1.md)* instead; after selecting, check the deliverable fields in [Quantization Mode Support](#mode-support) below.

---

## 5. <span id="mode-support">Quantization Mode Support</span>

> **Deliverable notes**: In the table, "Deliverable: quantization description JSON" corresponds to `quant_model_description*.json`; "Deliverable: quantization safetensors" corresponds to `quant_model_weight*.safetensors`. For the mode principles see the *[Quantization Modes](../../quantization_mode/README.md)* entry (this entry does not elaborate the dequantization formulas or NPU operators). Specific model and `quant_type` combinations are subject to the *[Large Model Support Matrix](../../model/README.md)* and lab_practice / example.

| Format Enumeration | Whether MindIE SD Supports Write-to-Disk | Quantization Mode Entry | Deliverable: Quantization Description JSON | Deliverable: Quantization safetensors |
| ---------------------- | ---------------- | ------ | ------------------------------------------------ | ----------------------------------------------- |
| `FLOAT` | Supported | [Quantization Mode Overview](../../quantization_mode/README.md#modes) | [FLOAT Description Keys](#desc-float) | [FLOAT Weight Tensors](#st-float) |
| `W8A8` | Supported | [W8A8 Static Quantization](../../quantization_mode/linear_layer_quantization/term_w8a8_static.md) | [W8A8 Description Keys](#desc-w8a8) | [W8A8 Weight Tensors](#st-w8a8) |
| `W8A8_DYNAMIC` | Supported | [W8A8 Dynamic Quantization](../../quantization_mode/linear_layer_quantization/term_w8a8_dynamic.md) | [W8A8_DYNAMIC Description Keys](#desc-w8a8-dynamic) | [W8A8_DYNAMIC Weight Tensors](#st-w8a8-dynamic) |
| `W8A8_MXFP8` | Supported | [W8A8 MX Dynamic Quantization](../../quantization_mode/linear_layer_quantization/term_w8a8_mx_dynamic.md) | [W8A8_MXFP8 Description Keys](#desc-w8a8-mxfp8) | [W8A8_MXFP8 Weight Tensors](#st-w8a8-mxfp8) |
| `W4A4_MXFP4` | Supported | [W4A4 MX Dynamic Quantization](../../quantization_mode/linear_layer_quantization/term_w4a4_mx_dynamic.md) | [W4A4_MXFP4 Description Keys](#desc-w4a4-mxfp4) | [W4A4_MXFP4 Weight Tensors](#st-w4a4-mxfp4) |
| `W4A4_MXFP4_DUALSCALE` | Supported | [W4A4 MX Dual Scale](../../quantization_mode/linear_layer_quantization/term_w4a4_mx_dualscale.md) | [W4A4_MXFP4_DUALSCALE Description Keys](#desc-mxfp-dualscale) | [W4A4_MXFP4_DUALSCALE Weight Tensors](#st-mxfp-dualscale) |
| `FAQuant` | Supported (FA3 and so on) | [FA INT8 PerHead Quantization](../../quantization_mode/fa_quantization/term_fa_int8_perhead.md) | [FAQuant Description Keys](#desc-faquant) | [FAQuant Weight Tensors](#st-faquant) |

---

## 6. Deliverable Formats of Each Quantization Mode

Conventions:

- `{prefix}` is the module prefix (determined by the multimodal adapter naming).
- **Description JSON**: the file `quant_model_description*.json`; keys are full tensor names and values are quantization type strings; for global fields see [Global Metadata Fields](#global-metadata) above.
- **safetensors**: the file `quant_model_weight*.safetensors` (may be sharded); keys are `{prefix}.<param>` and store the actual numeric tensors.

### 6.1 FLOAT

**<span id="desc-float">quant_model_description.json</span>**

| Description Key | Value | Description |
| ---------------------- | --------- | ----------------------------------------------- |
| `{prefix}.weight` and other parameter names | `"FLOAT"` | Unquantized parameters; written out by `on_float_module` according to `named_parameters` |

**<span id="st-float">`quant_model_weight*.safetensors`</span>**

| Tensor Name | Data Type | Description |
| ------------ | ------ | ----------------------------------- |
| `{prefix}.*` | Same as the source parameter | Floating-point parameters are written to disk as-is; online rotation matrices and so on may also be written with the `"FLOAT"` tag |

### 6.2 W8A8

**<span id="desc-w8a8">quant_model_description.json</span>**

| Description Key | Value | Description |
| ----------------------- | --------- | ------------- |
| `{prefix}.weight` | `"W8A8"` | Quantized weight |
| `{prefix}.quant_bias` | `"W8A8"` | Quantization bias |
| `{prefix}.input_scale` | `"W8A8"` | Activation scale |
| `{prefix}.input_offset` | `"W8A8"` | Activation zero-point |
| `{prefix}.deq_scale` | `"W8A8"` | Combined dequantization scale |
| `{prefix}.bias` | `"FLOAT"` | Original floating-point bias (optional) |

**<span id="st-w8a8">`quant_model_weight*.safetensors`</span>**

| Tensor Name | Data Type | Description |
| ----------------------- | ------- | --------------- |
| `{prefix}.weight` | int8 | Quantized weight |
| `{prefix}.quant_bias` | int32 | Quantization bias |
| `{prefix}.input_scale` | float32 | Activation quantization scale |
| `{prefix}.input_offset` | float32 | Activation quantization zero-point |
| `{prefix}.deq_scale` | float32 | Combined dequantization scale |
| `{prefix}.bias` | float32 | Original floating-point bias (optional) |

### 6.3 W8A8_DYNAMIC

**<span id="desc-w8a8-dynamic">quant_model_description.json</span>**

| Description Key | Value | Description |
| ------------------------ | ---------------- | --------------- |
| `{prefix}.weight` | `"W8A8_DYNAMIC"` | Quantized weight |
| `{prefix}.weight_scale` | `"W8A8_DYNAMIC"` | Weight quantization scale |
| `{prefix}.weight_offset` | `"W8A8_DYNAMIC"` | Weight quantization zero-point |
| `{prefix}.bias` | `"FLOAT"` | Bias (optional) |

Activation dynamic parameters are not written to disk.

**<span id="st-w8a8-dynamic">`quant_model_weight*.safetensors`</span>**

| Tensor Name | Data Type | Description |
| ------------------------ | ------- | ----------------------- |
| `{prefix}.weight` | int8 | Quantized weight |
| `{prefix}.weight_scale` | float32 | Weight quantization scale |
| `{prefix}.weight_offset` | float32 | Weight quantization zero-point (0 when symmetric) |
| `{prefix}.bias` | float32 | Original floating-point bias (optional) |

### 6.4 W8A8_MXFP8

**<span id="desc-w8a8-mxfp8">quant_model_description.json</span>**

| Description Key | Value | Description |
| ----------------------- | -------------- | ---------------- |
| `{prefix}.weight` | `"W8A8_MXFP8"` | MXFP8 weight |
| `{prefix}.weight_scale` | `"W8A8_MXFP8"` | block-wise scale |
| `{prefix}.bias` | `"FLOAT"` | Bias (optional) |

**<span id="st-w8a8-mxfp8">`quant_model_weight*.safetensors`</span>**

| Tensor Name | Data Type | Description |
| ----------------------- | ------------- | -------------------------------------- |
| `{prefix}.weight` | float8_e4m3fn | MXFP8 weight |
| `{prefix}.weight_scale` | uint8 | block-wise scale (commonly stored **with a +127 offset** during export) |
| `{prefix}.bias` | float32 | Original floating-point bias (optional) |

### 6.5 W4A4_MXFP4

**<span id="desc-w4a4-mxfp4">quant_model_description.json</span>**

| Description Key | Value | Description |
| ----------------------- | -------------- | ---------------- |
| `{prefix}.weight` | `"W4A4_MXFP4"` | Packed FP4 weight |
| `{prefix}.weight_scale` | `"W4A4_MXFP4"` | block-wise scale |
| `{prefix}.bias` | `"FLOAT"` | Bias (optional) |

**<span id="st-w4a4-mxfp4">`quant_model_weight*.safetensors`</span>**

| Tensor Name | Data Type | Description |
| ----------------------- | ------- | -------------------------------- |
| `{prefix}.weight` | uint8 | FP4 packed storage |
| `{prefix}.weight_scale` | uint8 | block-wise scale (commonly **with a +127 offset**) |
| `{prefix}.bias` | float32 | Original floating-point bias (optional) |

### 6.6 W4A4_MXFP4_DUALSCALE

**<span id="desc-mxfp-dualscale">quant_model_description.json</span>**

Building on the [W4A4_MXFP4 description keys](#desc-w4a4-mxfp4), the values are all `"W4A4_MXFP4_DUALSCALE"`, with the following additions:

| Description Key | Value | Description |
| ---------------------------- | ------------------------ | --------- |
| `{prefix}.weight_dual_scale` | `"W4A4_MXFP4_DUALSCALE"` | Second scale path |

**<span id="st-mxfp-dualscale">`quant_model_weight*.safetensors`</span>**

| Tensor Name | Data Type | Description |
| ---------------------------- | ------- | ------------------------- |
| `{prefix}.weight` | uint8 | FP4 packed storage |
| `{prefix}.weight_scale` | uint8 | First scale path (commonly **with a +127 offset**) |
| `{prefix}.weight_dual_scale` | float32 | Second scale path |
| `{prefix}.bias` | float32 | Original floating-point bias (optional) |

### 6.7 FAQuant

**<span id="desc-faquant">quant_model_description.json</span>**

| Description Key / Global Field | Value | Description |
| --------------------------------- | ----------- | ------------------ |
| `{prefix}.scale`                  | `"FAQuant"` | FA per-head scale  |
| `{prefix}.offset`                 | `"FAQuant"` | FA per-head offset |
| Global `fa_quant_type`, layer-level `quant_type` | string | Written when FA3 and so on are enabled; the value of the layer-level `quant_type` depends on the quantization method of each Q/K/V branch; for the holistic modes see *[FA Quantization](../../quantization_mode/fa_quantization/README.md)* |

**<span id="st-faquant">`quant_model_weight*.safetensors`</span>**

| Tensor Name | Data Type | Description |
| ----------------- | -------------- | -------------------------- |
| `{prefix}.scale` | float32 | FA quantization scale |
| `{prefix}.offset` | int8 or float32 | Depends on the INT8 / FP8 per-head path |

---

## 7. Applicable Scenarios and Limitations

### 7.1 Applicable Scenarios

- Quantizing integrated multimodal generation models such as Wan2.2 and delivering them to MindIE.
- Export scenarios that need to be used together with the `inference_config` / dump configuration of `multimodal_sd_modelslim_v1`.

### 7.2 Usage Limitations

- Does not replace AscendV1 as the default export format for generic LLMs; modes whose handler is not implemented will prompt you to switch to AscendV1.
- Artifact file naming and fields evolve with the model adapter, so they must be verified against the target MindIE version before deployment.

---

## 8. Related Workflows

- *[MindIE SD Usage Guide](mindie_sd_usage.md)*: confirm mode support, configure, and run.
- *[One-Click Quantization Usage Guide](../../../user_guide/usage_one_click_quantization.md)*: details of the multimodal_sd configuration.
- *[Multimodal Generation Model Integration](../../ptq/dit/integration_guide_diffusion_transformer_quantization.md)*: model integration and examples.
- *[Quantization Format Integration Guide](../iformat_integration_guide.md)*: a reference for developing a new format.

---

## 9. Related Terms

- [Quantization Format Terminology Entry](../README.md): the parent concept; the directory this entry belongs to.
- [AscendV1 Quantization Format Terminology Entry](../ascendv1/term_ascendv1.md): other; a peer write-to-disk protocol that also belongs to quantization formats; modes whose handler is not implemented can switch to AscendV1.
- [compressed-tensors Quantization Format Terminology Entry](../compressed_tensors/term_compressed_tensors.md): other; a peer write-to-disk protocol that also belongs to quantization formats.
- [Quantization Modes](../../quantization_mode/README.md): a companion term; the deliverable enumerations of this format correspond to the quantization modes; see [Quantization Mode Support](#mode-support) on this page for details.
