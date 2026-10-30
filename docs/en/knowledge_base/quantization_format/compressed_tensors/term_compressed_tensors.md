# compressed-tensors Quantization Format Terminology Entry

> **Entry Category**: [Quantization Data Format](../README.md)<br>
> **English Name**: compressed_tensors<br>
> **Application Domain**: Quantized weight exchange in the HuggingFace / vLLM ecosystem<br>
> **msModelSlim Implementation**: [`msmodelslim/format/compressed_tensors_format/`](../../../../../msmodelslim/format/compressed_tensors_format/)

---

## 1. Overview

compressed-tensors is the [Quantization Format Terminology Entry](../README.md) compatible with the HuggingFace / vLLM ecosystem, and its field conventions follow the [vllm-project/compressed-tensors](https://github.com/vllm-project/compressed-tensors) specification. msModelSlim has this specification's structure definitions **built in at export time**, so the quantization process **does not require** installing the `compressed-tensors` package. Its core characteristic is writing the scheme into `config.json` -> `quantization_config` and the weights into HF-style safetensors.

For configuration, execution, and deployment steps see the *[compressed-tensors Usage Guide](compressed_tensors_usage.md)*. For the principles and formulas of each quantization mode see *[Quantization Modes](../../quantization_mode/README.md)* and the entry links in the support tables below.

---

## 2. Term Description

### 2.1 Principles

**Core Idea**

compressed-tensors is essentially a set of **write-to-disk conventions for quantized models aimed at the HuggingFace / vLLM ecosystem**: it does not perform calibration or fake-quantization computation; instead, at the end of the one-click quantization pipeline, it converts the already-quantized QIR modules into two types of deliverables that inference frameworks can load directly - the **`quantization_config` injected into `config.json`** and the **HF-style weight file** `model*.safetensors`. The former describes metadata such as `quant_method: "compressed-tensors"` and the scheme (`QuantizationArgs` / `config_groups`) for vLLM and so on to identify the quantization scheme through the HF automatic-detection path; the latter writes tensors such as int8 / scale / zero-point by module prefix. The inference side first reads `quantization_config` to determine the scheme, then fetches data from safetensors by key name.

Therefore, users can understand compressed-tensors as: **the HF-ecosystem standard export package of msModelSlim quantization results** - the algorithms handle the quantization computation, while this format handles the write-to-disk and description aligned with the [vllm-project/compressed-tensors](https://github.com/vllm-project/compressed-tensors) specification.

**Key Properties**

- Compatible with the HF `from_pretrained` / vLLM automatic-detection path.
- Currently linear layer quantization only; `targets` is fixed to `["Linear"]`.
- Distributed export is not supported; `kv_cache_scheme` is always `null`.

---

## 3. <span id="export-artifacts">Export Artifacts (Deliverables)</span>

### 3.1 Directory and File Description

After running one-click quantization with the compressed-tensors write-to-disk format specified (the saver `type` is `compressed_tensors`), the following files are generated under the specified `save_path` directory:

```text
save_directory/
├── config.json                          # Injects the quantization_config field
├── model*.safetensors                   # Unsharded is model.safetensors; sharded is model-*-of-*.safetensors
├── model.safetensors.index.json         # Generated when sharded
└── (HF auxiliary files copied from the source model)
    └── *.json / *.py / *.txt / *.jinja
```

| File Name | Description |
| ------------------------------ | ------------------------------------------------- |
| `config.json` | The original HF configuration, with `quantization_config` (quantization scheme metadata) injected |
| `model.safetensors` | **Quantization weights**; when large they can be sharded as `model-*-of-*.safetensors` |
| `model.safetensors.index.json` | Shard index (generated only when sharded) |
| HF auxiliary files | `*.json` / `*.py` / `*.txt` / `*.jinja` and so on, copied from the source model |

### 3.2 config.json → quantization_config

**File Structure Example**

An example of the `quantization_config` structure is as follows:

```json
{
  "version": "0.13.0",
  "quant_method": "compressed-tensors",
  "sparsity_config": {},
  "transform_config": {},
  "config_groups": {
    "group_0": {
      "targets": ["Linear"],
      "weights": { "num_bits": 8, "type": "int", "strategy": "channel", "symmetric": true, "dynamic": false },
      "input_activations": { "num_bits": 8, "type": "int", "strategy": "tensor", "symmetric": false, "dynamic": false },
      "format": "int-quantized"
    }
  },
  "format": "int-quantized",
  "quantization_status": "compressed",
  "global_compression_ratio": null,
  "ignore": ["re:...(?![.\\w])"],
  "kv_cache_scheme": null
}
```

**<span id="global-metadata">Top-Level Field Description</span>**

| Field | Description |
| -------------------------------------- | ------------------------------------------------------------- |
| `version` | Schema version; fixed to `"0.13.0"` |
| `quant_method` | Fixed to `"compressed-tensors"` |
| `config_groups` | Grouped by unique scheme; the key names are `group_0`, `group_1`, and so on |
| `format` | The root format, such as `int-quantized` and `mixed-precision` |
| `quantization_status` | `"compressed"` at export time |
| `global_compression_ratio` | Global compression ratio (a floating-point number from 0 to 1); an optional informational field that **does not participate in inference loading**; msModelSlim currently does not compute it and it is always `null` |
| `ignore` | A regex list of layer names of the same type that are not quantized |
| `kv_cache_scheme` | The KV Cache quantization scheme; **currently not supported** and always `null` |
| `sparsity_config` / `transform_config` | Empty object placeholders |

**QuantizationScheme (inside config_groups)**

Each `config_groups` entry describes the quantization scheme of a group of layers. **msModelSlim currently supports only linear layer (**`nn.Linear` **/ QIR FakeQuantLinear) quantization**, so `targets` is fixed to `["Linear"]`.

| Field | Description |
| -------------------- | ----------------------------- |
| `targets` | Target module type; **currently fixed to** `["Linear"]` |
| `weights` | Weight quantization parameters (`QuantizationArgs`) |
| `input_activations` | Input activation quantization parameters; `null` for weight-only quantization |
| `output_activations` | Output activation quantization parameters |
| `format` | Layer compression format, such as `int-quantized` |

**QuantizationArgs Parameter Description**

| Parameter | Type | Default | Description |
| ----------------- | --------- | ------- | -------------------------------------------------------------------------------------- |
| `num_bits` | int | `8` | Quantization bit width |
| `type` | enum | `"int"` | Quantization type: `"int"` / `"float"` |
| `symmetric` | bool | `true` | Whether the quantization is symmetric |
| `strategy` | enum | Inferred automatically | Quantization granularity: `tensor` / `channel` / `group` / `block` / `token` / `tensor_group` / `attn_head` |
| `group_size` | int | `null` | The group size of the group strategy |
| `block_structure` | list[int] | `null` | Specific to the block strategy; an integer list of length 2 in the form `[rows, cols]` |
| `dynamic` | bool | `false` | Whether the quantization is dynamic: `false` is static and `true` is dynamic; msModelSlim writes it explicitly from the QIR preset at export time |
| `actorder` | enum | `null` | Activation ordering (`group` / `weight`, and so on) |
| `scale_dtype` | string | `null` | The data type of the scale tensor (a torch dtype string) |
| `zp_dtype` | string | Inferred automatically | The data type of the zero-point tensor; exported as `null` for symmetric quantization |
| `observer` | string | Inferred automatically | The calibration method; the default is `memoryless_minmax` for static quantization and `null` for dynamic quantization |
| `observer_kwargs` | object | `{}` | Extra parameters passed to the observer |

---

## 4. <span id="engine-support">Inference Engine Support</span>

All Presets in the table **can be exported by compressed-tensors**; the table describes whether the artifact can be loaded by the target inference engine through the HF `quantization_config`. Specific model x Preset x engine combinations are subject to the *[Foundation Model Support Matrix](../../model/README.md)* and the official best practices.

| Format Preset | vLLM (HF ecosystem) | Description |
| --- | --- | --- |
| W8A8 Static | √ | The activation has been calibrated offline; `act.scope: per_tensor` |
| W8A8 Dynamic | √ | Dynamic activation; `act.scope: per_token` |

> **Legend**: `√` means the engine has a loadable path or validated practice. When selecting, first confirm that the framework supports `quant_method: "compressed-tensors"`, then check the deliverable fields in [Quantization Mode Support](#mode-support) below. For Ascend private paths use the *[AscendV1](../ascendv1/term_ascendv1.md)* instead.

---

## 5. <span id="mode-support">Quantization Mode Support</span>

> **Deliverable notes**: In the table, "Deliverable: quantization_config" corresponds to the scheme inside `config.json`; "Deliverable: safetensors" corresponds to `model*.safetensors`. For the mode principles see the *[Quantization Modes](../../quantization_mode/README.md)* entry. The deliverables of this entry are described in two columns: `quantization_config` and `model*.safetensors`.

| Format Preset | Whether compressed-tensors Supports Export | Quantization Mode Entry | Deliverable: quantization_config | Deliverable: safetensors |
| ------------ | ------------------------- | ---------------- | ----------------------------------------- | ------------------------------------- |
| W8A8 Static | Supported | [W8A8 Static Quantization](../../quantization_mode/linear_layer_quantization/term_w8a8_static.md) | [W8A8 Static scheme](#desc-w8a8-static) | [W8A8 Static weight tensors](#st-w8a8-static) |
| W8A8 Dynamic | Supported | [W8A8 Dynamic Quantization](../../quantization_mode/linear_layer_quantization/term_w8a8_dynamic.md) | [W8A8 Dynamic scheme](#desc-w8a8-dynamic) | [W8A8 Dynamic weight tensors](#st-w8a8-dynamic) |

---

## 6. Deliverable Formats of Each Quantization Mode

Conventions:

- `{prefix}` is the module prefix (for example `model.layers.0.self_attn.q_proj`).
- **quantization_config**: written into `config.json`; for the top level and `QuantizationArgs` see [Top-Level Field Description](#global-metadata) above; below are the typical schemes of each Preset in `config_groups`.
- **safetensors**: the file `model*.safetensors` (may be sharded); keys are `{prefix}.<param>` and store the actual numeric tensors.

### 6.1 W8A8 Static

**<span id="desc-w8a8-static">config.json → quantization_config</span>**

A typical `config_groups` entry (`weights.dynamic = false`, with a static activation):

| Field Path | Typical Value | Description |
| ---------------------------------------------------------------------------- | ---------------------------------------------- | ------ |
| `targets` | `["Linear"]` | Linear layers only |
| `weights.num_bits` / `type` / `strategy` / `symmetric` / `dynamic`           | `8` / `"int"` / `"channel"` / `true` / `false` | weight quantization   |
| `input_activations.num_bits` / `type` / `strategy` / `symmetric` / `dynamic` | `8` / `"int"` / `"tensor"` / `false` / `false` | Static activation quantization |
| `format` | `"int-quantized"` | Layer compression format |

**<span id="st-w8a8-static">`model*.safetensors`</span>**

| Tensor Name | Data Type | Description |
| --------------------------- | ------- | -------------------------- |
| `{prefix}.weight` | int8 | Quantized weight |
| `{prefix}.weight_scale` | float32 | Weight quantization scale (unsqueezed to 2D) |
| `{prefix}.input_scale`      | float32 | Activation quantization scale                 |
| `{prefix}.input_zero_point` | - | Written only when `input_offset != 0` |
| `{prefix}.bias`             | float32 | Optional                         |

### 6.2 W8A8 Dynamic

**<span id="desc-w8a8-dynamic">config.json → quantization_config</span>**

A typical `config_groups` entry (the activation is dynamic; `input_activations.dynamic = true` at export time):

| Field Path | Typical Value | Description |
| ---------------------------------------------------------------------------- | --------------------------------------------- | ------------------------------- |
| `targets` | `["Linear"]` | Linear layers only |
| `weights.num_bits` / `type` / `strategy` / `symmetric` / `dynamic` | `8` / `"int"` / `"channel"` / `true` / `false` | Weight quantization (per-channel) |
| `input_activations.num_bits` / `type` / `strategy` / `symmetric` / `dynamic` | `8` / `"int"` / `"token"` / `false` / `true` | Dynamic activation; scale / zero-point are **not written to disk** |
| `format` | `"int-quantized"` | Layer compression format |

**<span id="st-w8a8-dynamic">`model*.safetensors`</span>**

| Tensor Name | Data Type | Description |
| ----------------------- | ------- | ------------------------------- |
| `{prefix}.weight` | int8 | Quantized weight |
| `{prefix}.weight_scale` | float32 | Weight quantization scale (unsqueezed to 2D when 1D) |
| `{prefix}.bias`         | float32 | Optional                              |

> The scale / zero-point of the dynamic activation are **not written** to the weight file and are computed dynamically per-token at inference time.

---

## 7. Applicable Scenarios and Limitations

### 7.1 Applicable Scenarios

- Delivering interchangeable quantized weights to HF-ecosystem frameworks such as vLLM.
- Exchanging a `quantization_config` that needs to align with the compressed-tensors specification.

### 7.2 Usage Limitations

- Distributed export is not supported (`support_distributed() = False`).
- KV Cache quantization is not supported yet (`kv_cache_scheme = null`).
- Only the two QIR types W8A8 Static / W8A8 Dynamic have handlers.
- The quantization mode principles and operator descriptions are not elaborated (see *[Quantization Modes](../../quantization_mode/README.md)*).

---

## 8. Related Workflows

- *[compressed-tensors Usage Guide](compressed_tensors_usage.md)*: confirm mode support, configure, and run.
- *[Quantization Format Integration Guide](../iformat_integration_guide.md)*: an IFormat one-shot reference.
- *[One-Click Quantization Usage Guide](../../../user_guide/usage_one_click_quantization.md)*: an overview of the save configuration.

---

## 9. Related Terms

- [Quantization Format Terminology Entry](../README.md): the parent concept; the directory this entry belongs to.
- [AscendV1 Quantization Format Terminology Entry](../ascendv1/term_ascendv1.md): other; a peer write-to-disk protocol that also belongs to quantization formats.
- [MindIE SD Quantization Format Terminology Entry](../mindie_sd/term_mindie_sd.md): other; a peer write-to-disk protocol that also belongs to quantization formats.
- [Quantization Modes](../../quantization_mode/README.md): a companion term; the Presets and deliverable fields of this format correspond to the quantization modes; see [Quantization Mode Support](#mode-support) on this page for details.
