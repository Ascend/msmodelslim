<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.processor.quant.autoround.AutoroundProcessorConfig -->
# autoround_quant Configuration Reference

## 1. Configuration Overview

AutoRound quantization processor configuration.

| Item | Content |
|------|------|
| Config Class | `AutoroundProcessorConfig` |
| Source | [autoround.py](../../../../../msmodelslim/processor/quant/autoround.py) |

## 2. Parameter List

<h3 id="2-1-autoround-quant">2.1 AutoroundProcessorConfig</h3>

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | Optional | `autoround_quant` | `autoround_quant` | Processor type, fixed to `autoround_quant`. | None |
| `iters` | `int` | Optional | `10` | — | Number of iterations; must be greater than 0 | None |
| `enable_minmax_tuning` | `bool` | Optional | `true` | — | Whether to enable min-max tuning | None |
| `enable_round_tuning` | `bool` | Optional | `true` | — | Whether to enable rounding tuning | None |
| `strategies` | `list[object]` | Optional | `[]` | — | List of quantization strategy configurations; at least one is required | This page <a href="#2-2-autoround-quant-strategy-config">§2.2</a> |

**Configuration Constraints**

- Validates strategies: must be non-empty; the qconfig of each strategy must satisfy the group_size rule (when scope=per_group, ext.group_size must be a positive integer; when scope≠per_group, group_size must not be present); and it must pass the quantizer layer configuration pre-check.

<h3 id="2-2-autoround-quant-strategy-config">2.2 QuantStrategyConfig</h3>

autoround quantization strategy: applies one set of quantization configurations to matching linear layers.

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `qconfig` | `object` | Required | None | — | Quantization configuration for activations and weights; see <a href="#2-2-1-linear-qconfig">LinearQConfig Configuration Reference</a>. | This page <a href="#2-2-1-linear-qconfig">§2.2.1</a> |
| `include` | `list[string]` | Optional | `['*']` | — | Module name patterns to include; the default `*` matches all modules | None |
| `exclude` | `list[string]` | Optional | `[]` | — | Module name patterns to exclude; takes precedence over `include` | None |

**Configuration Constraints**

- None.

<h4 id="2-2-1-linear-qconfig">2.2.1 LinearQConfig</h4>

Quantization configuration for linear layers, covering both activation and weight quantization.

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `act` | `object` | Optional | `{'dtype': 'float', 'scope': 'per_tensor', 'symmetric': True, 'method': 'none', 'ext': {}}` | — | Quantization configuration for activations. Implemented by the activation quantizer; `method` supports `minmax`, `histogram`, `none`; `scope` supports `per_tensor`, `per_token`, `per_channel` (minmax only, for int8/fp8_e4m3/mxfp8), and more. Defaults to `float` (activations not quantized), quantizing weights only. | This page <a href="#2-2-2-qconfig">§2.2.2</a> |
| `weight` | `object` | Required | None | — | Quantization configuration for weights; required. Implemented by the weight quantizer; `method` supports `minmax`, `mse_round`, `ssz`, `gptq`, and more; `scope` supports `per_channel`, `per_group`, `per_block`, and more (`per_tensor`/`per_token` are not supported). | This page <a href="#2-2-2-qconfig">§2.2.2</a> |

**Configuration Constraints**

- None.

<h4 id="2-2-2-qconfig">2.2.2 QConfig</h4>

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
  - type: autoround_quant
    iters: 10
    enable_minmax_tuning: true
    enable_round_tuning: true
    strategies:
    - qconfig:
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
      include:
      - '*'
      exclude: []
```
