<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.processor.flat_quant.flat_quant.FlatQuantProcessorConfig -->
# flatquant Configuration Reference

## 1. Configuration Overview

FlatQuant processor configuration: defines quantization training parameters, strategy, mixed precision, and more

| Item | Content |
|------|------|
| Config Class | `FlatQuantProcessorConfig` |
| Source | [flat_quant.py](../../../../../msmodelslim/processor/flat_quant/flat_quant.py) |

## 2. Parameter List

<h3 id="2-1-flatquant">2.1 FlatQuantProcessorConfig</h3>

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | Optional | `flatquant` | `flatquant` | Processor type identifier, fixed to 'flatquant'. | None |
| `include` | `list[string]` | Optional | `['*']` | — | Names of modules to include | None |
| `exclude` | `list[string]` | Optional | `[]` | — | Names of modules to exclude | None |
| `strategies` | `list[object]` | Optional | `[]` | — | List of quantization strategy configurations | This page <a href="#2-2-flatquant-quant-strategy-config">§2.2</a> |
| `seed` | `int` | Optional | `0` | — | Random seed, used to reproduce results | None |
| `diag_relu` | `bool` | Optional | `true` | — | Whether to use the diag_relu activation function to implement the transformation matrix | None |
| `amp_dtype` | `string` | Optional | `bfloat16` | — | Mixed-precision type, used to accelerate training | None |
| `a_bits` | `int` | Optional | `4` | — | Activation quantization bit width during calibration training (for example 4bit) | None |
| `a_groupsize` | `int` | Optional | `-1` | — | Activation quantization group size during calibration training (-1 means per-tensor grouping) | None |
| `a_asym` | `bool` | Optional | `false` | — | Whether activation quantization during calibration training is asymmetric | None |
| `a_per_tensor` | `bool` | Optional | `false` | — | Whether activation quantization during calibration training is per-tensor rather than per-channel | None |
| `w_bits` | `int` | Optional | `4` | — | Weight quantization bit width during calibration training (for example 4bit) | None |
| `w_groupsize` | `int` | Optional | `-1` | — | Weight quantization group size during calibration training (-1 means per-tensor grouping) | None |
| `w_asym` | `bool` | Optional | `false` | — | Whether weight quantization during calibration training is asymmetric | None |
| `epochs` | `int` | Optional | `10` | — | Total number of calibration training epochs | None |
| `nsamples` | `int / null` | Optional | `null` | — | Number of samples used for calibration | None |
| `cali_bsz` | `int` | Optional | `4` | — | Batch size for the calibration stage | None |
| `flat_lr` | `float` | Optional | `0.001` | — | Learning rate for FlatQuant quantization training | None |
| `add_diag` | `bool` | Optional | `true` | — | Whether to enable the diagonal scaling matrix for global scaling | None |
| `lwc` | `bool` | Optional | `true` | — | Whether to enable weight calibration (training the weight quantization parameters) | None |
| `lac` | `bool` | Optional | `true` | — | Whether to enable activation calibration (training the activation quantization parameters) | None |
| `diag_init` | `string` | Optional | `one_style` | — | Initialization method of the diagonal scaling matrix; supports sq_style and one_style | None |
| `diag_alpha` | `float` | Optional | `0.3` | — | Diagonal scaling parameter; controls the scaling strength | None |
| `warmup` | `bool` | Optional | `true` | — | Whether to enable the training warmup mechanism to improve stability | None |
| `deactive_amp` | `bool` | Optional | `true` | — | Whether to disable mixed-precision training (for debugging) | None |
| `tran_type` | `string` | Optional | `svd` | — | Transformation matrix implementation: svd means based on SVD decomposition | None |

**Configuration Constraints**

- Validation logic: internal fields with `init=False` must not be assigned explicitly in YAML; assigning them raises an error.

<h3 id="2-2-flatquant-quant-strategy-config">2.2 QuantStrategyConfig</h3>

Quantization strategy configuration: defines quantization parameters and include/exclude module rules

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `qconfig` | `object` | Required | None | — | Quantization configuration parameters | This page <a href="#2-2-1-linear-qconfig">§2.2.1</a> |
| `include` | `list[string]` | Optional | `['*']` | — | Module names to include (wildcard * supported) | None |
| `exclude` | `list[string]` | Optional | `[]` | — | Module names to exclude (takes precedence over include) | None |

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
  - type: flatquant
    include:
    - '*'
    exclude: []
    strategies: []
    seed: 0
    diag_relu: true
    amp_dtype: bfloat16
    a_bits: 4
    a_groupsize: -1
    a_asym: false
    a_per_tensor: false
    w_bits: 4
    w_groupsize: -1
    w_asym: false
    epochs: 10
    cali_bsz: 4
    flat_lr: 0.001
    add_diag: true
    lwc: true
    lac: true
    diag_init: one_style
    diag_alpha: 0.3
    warmup: true
    deactive_amp: true
    tran_type: svd
```
