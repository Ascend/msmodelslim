<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.processor.quant.fa3.processor.FA3QuantProcessorConfig -->
# fa3_quant Configuration Reference

## 1. Configuration Overview

FA3 (FlashAttention-3) quantization processor configuration.

| Item | Content |
|------|------|
| Config Class | `FA3QuantProcessorConfig` |
| Source | [processor.py](../../../../../msmodelslim/processor/quant/fa3/processor.py) |

## 2. Parameter List

<h3 id="2-1-fa3-quant">2.1 FA3QuantProcessorConfig</h3>

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | Optional | `fa3_quant` | `fa3_quant` | Processor type, fixed to `fa3_quant`. | None |
| `qconfig` | `object / null` | Optional | `null` | — | Unified quantization configuration; if not provided, the default is `dtype: int8`, `scope: per_head`, `symmetric: true`, `method: minmax` (symmetric quantization). See <a href="#2-2-qconfig">QConfig Configuration Reference</a> | This page <a href="#2-2-qconfig">§2.2</a> |
| `include` | `list[string]` | Optional | `['*']` | — | Module name patterns to include; the default `*` matches all modules | None |
| `exclude` | `list[string]` | Optional | `[]` | — | Module name patterns to exclude; takes precedence over `include` | None |
| `details` | `object / null` | Optional | `null` | — | Q/K/V branch-level quantization configuration (FA3AttentionDetails). See <a href="#2-3-fa3-attention-details">FA3AttentionDetails Configuration Reference</a> | This page <a href="#2-3-fa3-attention-details">§2.3</a> |

**Configuration Constraints**

- Validates that qconfig and details are mutually exclusive: providing both raises an error; providing neither uses the default `{dtype: int8, scope: per_head, symmetric: true, method: minmax}`.

<h3 id="2-2-qconfig">2.2 QConfig</h3>

Describes how a single tensor (weight or activation) is quantized.

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `dtype` | `string` | Required | None | `float`, `int8`, `int4`, `mxfp8`, `mxfp4`, `fp8_e4m3` | Quantization data type, such as `int8`, `int4`, `mxfp8`, `mxfp4`, `fp8_e4m3`; `float` means the tensor is not quantized. | None |
| `scope` | `string` | Required | None | `per_tensor`, `per_channel`, `per_group`, `per_block`, `per_token`, `pd_mix`, `per_head`, `dual_scale`, `per_channel_neg_offset` | Quantization granularity, that is, the range over which scale/zero_point is computed. For different quantization targets, the available `scope` values usually differ:<br>Activation: `per_tensor` (a single scale for the whole tensor), `per_token` (per attention head), `per_block` (per fixed block), `pd_mix` (mixed Prefill/Decode granularity);<br>Weight: `per_channel` (per channel), `per_group` (per group), `per_block` (per fixed block), `dual_scale` (dual scale);<br>FA3 attention: `per_head` (per attention head), `per_token` (per attention head). | None |
| `symmetric` | `bool` | Required | None | — | Whether to use symmetric quantization. Symmetric quantization stores only the scale; asymmetric quantization additionally stores a zero_point. Availability depends on the `dtype`/`scope` combination. | None |
| `method` | `string` | Required | None | — | Quantization parameter estimation algorithm, such as `minmax`, `mse_round`, `histogram`, `ssz`, `none`, `gptq`, and more. Algorithms belong to one of two quantizers by tensor type: the activation quantizer supports `minmax`, `histogram`, and `none` (together with `float`); the weight quantizer supports `minmax`, `mse_round`, `ssz`, `gptq`, and more. The available values also depend on the `dtype`/`scope`/`symmetric` combination; `none` means the parameters are not estimated. | None |
| `ext` | `object` | Optional | `{}` | — | Quantizer extension parameters, determined by `method` and the quantizer implementation (for example `percdamp`/`group_size` for gptq); an empty object means no extension parameters. | None |

**Configuration Constraints**

- None.

<h3 id="2-3-fa3-attention-details">2.3 FA3AttentionDetails</h3>

FA3 attention branch-level quantization configuration; specifies the quantization mode of Q/K/V separately.

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `fa_q` | `object` | Optional | `null` | — | Quantization configuration for the Query branch. See <a href="#2-2-qconfig">QConfig Configuration Reference</a> | This page <a href="#2-2-qconfig">§2.2</a> |
| `fa_k` | `object` | Optional | `null` | — | Quantization configuration for the Key branch. See <a href="#2-2-qconfig">QConfig Configuration Reference</a> | This page <a href="#2-2-qconfig">§2.2</a> |
| `fa_v` | `object` | Optional | `null` | — | Quantization configuration for the Value branch. See <a href="#2-2-qconfig">QConfig Configuration Reference</a> | This page <a href="#2-2-qconfig">§2.2</a> |

**Configuration Constraints**

- None.

## 3. Complete Configuration Reference

```yaml
apiversion: modelslim_v1
spec:
  process:
  - type: fa3_quant
    include:
    - '*'
    exclude: []
```
