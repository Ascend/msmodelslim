<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.processor.anti_outlier.flex_smooth.processor.FlexAWQSSZProcessorConfig -->
# flex_awq_ssz Configuration Reference

## 1. Configuration Overview

Flex AWQ SSZ outlier suppression processor configuration.

| Item | Content |
|------|------|
| Config Class | `FlexAWQSSZProcessorConfig` |
| Source | [processor.py](../../../../../msmodelslim/processor/anti_outlier/flex_smooth/processor.py) |

## 2. Parameter List

<h3 id="2-1-flex-awq-ssz">2.1 FlexAWQSSZProcessorConfig</h3>

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | Optional | `flex_awq_ssz` | `flex_awq_ssz` | Processor type, fixed to `flex_awq_ssz`. | None |
| `alpha` | `float` | Optional | `null` | — | Strength of the activation-to-weight smooth migration (0-1); a higher value migrates more outliers to the weights. | None |
| `beta` | `float` | Optional | `null` | — | Additional smoothing tuning coefficient (0-1), used together with alpha. | None |
| `enable_subgraph_type` | `list[any]` | Optional | `['norm-linear', 'linear-linear', 'ov', 'up-down']` | — | List of subgraph types to smooth; the default is `norm-linear`, `linear-linear`, `ov`, `up-down`. | None |
| `include` | `list[string] / null` | Optional | `null` | — | Module name patterns to include; if unset, all modules match. | None |
| `exclude` | `list[string] / null` | Optional | `null` | — | Module name patterns to exclude; takes precedence over `include`. | None |
| `qconfig` | `object` | Required | None | — | Quantization configuration for activations and weights; see <a href="#2-2-linear-qconfig">LinearQConfig Configuration Reference</a>. | This page <a href="#2-2-linear-qconfig">§2.2</a> |

**Configuration Constraints**

- Model-level pre-check: rejects scenarios where qconfig is missing

<h3 id="2-2-linear-qconfig">2.2 LinearQConfig</h3>

Quantization configuration for linear layers, covering both activation and weight quantization.

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `act` | `object` | Optional | `{'dtype': 'float', 'scope': 'per_tensor', 'symmetric': True, 'method': 'none', 'ext': {}}` | — | Quantization configuration for activations. Implemented by the activation quantizer; `method` supports `minmax`, `histogram`, `none`; `scope` supports `per_tensor`, `per_token`, `per_channel` (minmax only, for int8/fp8_e4m3/mxfp8), and more. Defaults to `float` (activations not quantized), quantizing weights only. | This page <a href="#2-2-1-qconfig">§2.2.1</a> |
| `weight` | `object` | Required | None | — | Quantization configuration for weights; required. Implemented by the weight quantizer; `method` supports `minmax`, `mse_round`, `ssz`, `gptq`, and more; `scope` supports `per_channel`, `per_group`, `per_block`, and more (`per_tensor`/`per_token` are not supported). | This page <a href="#2-2-1-qconfig">§2.2.1</a> |

**Configuration Constraints**

- None.

<h4 id="2-2-1-qconfig">2.2.1 QConfig</h4>

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

## 3. Complete Configuration Reference

```yaml
apiversion: modelslim_v1
spec:
  process:
  - type: flex_awq_ssz
    enable_subgraph_type:
    - norm-linear
    - linear-linear
    - ov
    - up-down
    qconfig:
      act:
        dtype: float
        scope: per_tensor
        symmetric: true
        method: none
        ext: {}
      weight:
        dtype: int8
        scope: per_channel
        symmetric: true
        method: minmax
        ext: {}
```
