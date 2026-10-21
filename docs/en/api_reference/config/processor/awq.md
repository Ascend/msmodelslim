<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.processor.anti_outlier.awq.processor.AWQProcessorConfig -->
# awq Configuration Reference

## 1. Configuration Overview

AWQ (Activation-aware Weight Quantization) processor configuration.

| Item | Content |
|------|------|
| Config Class | `AWQProcessorConfig` |
| Source | [processor.py](../../../../../msmodelslim/processor/anti_outlier/awq/processor.py) |

## 2. Parameter List

<h3 id="2-1-awq">2.1 AWQProcessorConfig</h3>

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | Optional | `awq` | `awq` | Processor type, fixed to `awq`. | None |
| `weight_qconfig` | `object` | Required | None | — | Weight quantization configuration (QConfig); required. See <a href="#2-2-qconfig">QConfig Configuration Reference</a>. | This page <a href="#2-2-qconfig">§2.2</a> |
| `enable_subgraph_type` | `list[any]` | Optional | `['norm-linear', 'linear-linear', 'ov', 'up-down']` | — | List of subgraph types for AWQ; the default is `norm-linear`, `linear-linear`, `ov`, `up-down`. | None |
| `n_grid` | `int` | Optional | `20` | — | Number of grids for the grid search, used to find the optimal quantization scale/clipping point; must be greater than 0. | None |
| `include` | `list[string] / null` | Optional | `null` | — | Module name patterns to include; if unset, all modules match. | None |
| `exclude` | `list[string] / null` | Optional | `null` | — | Module name patterns to exclude; takes precedence over `include`. | None |

**Configuration Constraints**

- None.

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

## 3. Complete Configuration Reference

```yaml
apiversion: modelslim_v1
spec:
  process:
  - type: awq
    weight_qconfig:
      dtype: int8
      scope: per_channel
      symmetric: true
      method: minmax
      ext: {}
    enable_subgraph_type:
    - norm-linear
    - linear-linear
    - ov
    - up-down
    n_grid: 20
```
