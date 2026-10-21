<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.processor.trainable_linear_quant.config.processor_config.TrainableLinearQuantProcessorConfig -->
# trainable_linear_quant Configuration Reference

## 1. Configuration Overview

Trainable linear quantization (TLQ) processor configuration.

| Item | Content |
|------|------|
| Config Class | `TrainableLinearQuantProcessorConfig` |
| Source | [processor_config.py](../../../../../msmodelslim/processor/trainable_linear_quant/config/processor_config.py) |

## 2. Parameter List

<h3 id="2-1-trainable-linear-quant">2.1 TrainableLinearQuantProcessorConfig</h3>

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | Optional | `trainable_linear_quant` | `trainable_linear_quant` | Processor type, fixed to `trainable_linear_quant`. | None |
| `operations` | `list[object]` | Optional | `[MinmaxTuneOpConfig(type='minmax_tune', lr=None), RoundTuneOpConfig(type='round_tune', lr=None)]` | At least 1 item | List of OP configurations for the trainable quantization pipeline; each item contains type, and the remaining fields are defined by the individual plugin | This page <a href="#2-2-tlq-op-config">§2.2</a> |
| `strategies` | `list[object]` | Optional | `[]` | 0 items or ≥1 item | List of quantization strategy configurations; when not provided it is an empty list and no quantization strategy is applied; if provided explicitly, at least 1 item is required. | This page <a href="#2-3-tlq-quant-strategy-config">§2.3</a> |
| `train_with_act_quant` | `bool` | Optional | `false` | — | Whether the block-level training forward pass applies fake quantization to activations (through x_kernel); false is consistent with train_with_act_quant=False in autoround; the exported IR is still determined by qconfig.act and is unaffected by this option | None |
| `enable_quanted_input` | `bool` | Optional | `false` | — | Whether to use the quantized forward result of this layer as a bypass input (q_input) propagated to training/quantization of the next layer; does not affect the floating-point teacher: the Runner always propagates teacher outputs between layers via datas | None |
| `train_config` | `object` | Optional | See the nested configuration default | — | Block-level Trainer hyperparameters: iters, gradient_accumulate_steps, select_best, lr (or learning_rate), loss_type; each OP can configure lr separately to override the global value | This page <a href="#2-4-block-train-config">§2.4</a> |

**Configuration Constraints**

- Normalizes operations: accepts a single dict or a list; when not provided or malformed, it falls back to the default minmax_tune + round_tune pipeline.

<h3 id="2-2-tlq-op-config">2.2 TLQOpConfig</h3>

Plugin configuration for trainable TLQ operators (Op).

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | Required | None | — | Operator type; dispatches the concrete TLQ operator (such as `minmax_tune` or `round_tune`). | None |
| `lr` | `float / null` | Optional | `null` | >0.0 or null | Learning rate of the trainable parameters of this Op; when unspecified, train_config.lr is used | None |

**Configuration Constraints**

- None.

**Derived Classes**

| Config Class | `type` | Description | Document |
|--------|----------|------|------|
| `MinmaxTuneOpConfig` | `minmax_tune` | `MinmaxTuneOpConfig` is a nested configuration. | This page <a href="#2-2-1-minmax-tune">§2.2.1</a> |
| `RoundTuneOpConfig` | `round_tune` | `RoundTuneOpConfig` is a nested configuration. | This page <a href="#2-2-2-round-tune">§2.2.2</a> |
| `TrainableSmoothOpConfig` | `trainable_smooth` | `TrainableSmoothOpConfig` is a nested configuration. | This page <a href="#2-2-3-trainable-smooth">§2.2.3</a> |

<h4 id="2-2-1-minmax-tune">2.2.1 MinmaxTuneOpConfig</h4>

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | Optional | `minmax_tune` | `minmax_tune` | Plugin type: minmax_tune | None |
| `lr` | `float / null` | Optional | `null` | >0.0 or null | — | None |

**Configuration Constraints**

- None.

<h4 id="2-2-2-round-tune">2.2.2 RoundTuneOpConfig</h4>

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | Optional | `round_tune` | `round_tune` | Plugin type: round_tune | None |
| `lr` | `float / null` | Optional | `null` | >0.0 or null | — | None |

**Configuration Constraints**

- None.

<h4 id="2-2-3-trainable-smooth">2.2.3 TrainableSmoothOpConfig</h4>

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | Optional | `trainable_smooth` | `trainable_smooth` | Plugin type: trainable_smooth | None |
| `lr` | `float / null` | Optional | `null` | >0.0 or null | — | None |
| `enable_subgraph_type` | `list[string]` | Optional | `['norm-linear', 'linear-linear', 'ov', 'up-down', 'non-fusion']` | — | Enabled Smooth subgraph types; must be a subset of SMOOTH_SUPPORTED_SUBGRAPH_TYPES | None |
| `include` | `list[string] / null` | Optional | `null` | — | Subgraph entry include wildcard | None |
| `exclude` | `list[string] / null` | Optional | `null` | — | Subgraph entry exclude wildcard | None |

**Configuration Constraints**

- None.

<h3 id="2-3-tlq-quant-strategy-config">2.3 QuantStrategyConfig</h3>

trainable_linear_quant quantization strategy: applies one set of trainable quantization configurations to matching linear layers.

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `qconfig` | `object` | Required | None | — | Quantization configuration for activations and weights; see <a href="#2-3-1-linear-qconfig">LinearQConfig Configuration Reference</a>. | This page <a href="#2-3-1-linear-qconfig">§2.3.1</a> |
| `include` | `list[string]` | Optional | `['*']` | — | Module name patterns to include; the default `*` matches all modules | None |
| `exclude` | `list[string]` | Optional | `[]` | — | Module name patterns to exclude; takes precedence over `include` | None |

**Configuration Constraints**

- Validates qconfig: the dtype/method combination must be supported by a corresponding TLQ kernel, otherwise an error is raised.

<h4 id="2-3-1-linear-qconfig">2.3.1 LinearQConfig</h4>

Quantization configuration for linear layers, covering both activation and weight quantization.

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `act` | `object` | Optional | `{'dtype': 'float', 'scope': 'per_tensor', 'symmetric': True, 'method': 'none', 'ext': {}}` | — | Quantization configuration for activations. Implemented by the activation quantizer; `method` supports `minmax`, `histogram`, `none`; `scope` supports `per_tensor`, `per_token`, `per_channel` (minmax only, for int8/fp8_e4m3/mxfp8), and more. Defaults to `float` (activations not quantized), quantizing weights only. | This page <a href="#2-3-2-qconfig">§2.3.2</a> |
| `weight` | `object` | Required | None | — | Quantization configuration for weights; required. Implemented by the weight quantizer; `method` supports `minmax`, `mse_round`, `ssz`, `gptq`, and more; `scope` supports `per_channel`, `per_group`, `per_block`, and more (`per_tensor`/`per_token` are not supported). | This page <a href="#2-3-2-qconfig">§2.3.2</a> |

**Configuration Constraints**

- None.

<h4 id="2-3-2-qconfig">2.3.2 QConfig</h4>

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

<h3 id="2-4-block-train-config">2.4 BlockTrainConfig</h3>

Block-level training (block train) hyperparameter configuration.

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `iters` | `int` | Optional | `50` | ≥0 | Number of block-level training iterations; when 0, the Trainer skips optimization | None |
| `gradient_accumulate_steps` | `int` | Optional | `8` | ≥1 | Number of gradient accumulation steps, used to adjust the effective batch under limited video memory | None |
| `lr` | `float` | Optional | `0.01` | >0.0 | Global base learning rate (alias `learning_rate`) | None |
| `select_best` | `object` | Optional | See the nested configuration default | — | Best-iteration snapshot policy (fields differ by mode: ema / min_loss / last) | This page <a href="#2-4-1-selectbestconfig">§2.4.1</a> |
| `loss_type` | `string` | Optional | `l1` | `l1`, `custom_outlier` | Block-level training loss: l1 (L1Loss reduction=none), custom_outlier (0.3 * full L1 + 0.7 * L1 over the region within 3σ) | None |
| `train_seed` | `int` | Optional | `42` | — | Random seed for block-level training (used for sample shuffling and deterministic operators) | None |

**Configuration Constraints**

- None.

<h4 id="2-4-1-selectbestconfig">2.4.1 SelectBestConfig (Dispatched by <code>mode</code>)</h4>

**Derived Classes**

| Config Class | `mode` | Description | Document |
|--------|----------|------|------|
| `EmaSelectBest` | `ema` | Selects the best by EMA moving average; supports early stop. | This page <a href="#2-4-2-ema-select-best">§2.4.2</a> |
| `MinLossSelectBest` | `min_loss` | Selects the best by the historical minimum loss; supports early stop. | This page <a href="#2-4-3-min-loss-select-best">§2.4.3</a> |
| `LastSelectBest` | `last` | Saves only iteration 0 and the last iteration; no early stop. | This page <a href="#2-4-4-last-select-best">§2.4.4</a> |

<h4 id="2-4-2-ema-select-best">2.4.2 EmaSelectBest</h4>

Selects the best by EMA moving average; supports early stop.

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `mode` | `string` | Optional | `ema` | `ema` | Selection policy: ema | None |
| `ema_beta` | `float` | Optional | `0.7` | >0.0; ≤1.0 | EMA decay coefficient for best_loss | None |
| `ema_window_size` | `int` | Optional | `5` | ≥1 | Moving average window length for mean_loss | None |
| `early_stop_patience` | `int` | Optional | `-1` | ≥-1 | Number of consecutive iterations without a better snapshot before early stopping; -1 disables it | None |

**Configuration Constraints**

- None.

<h4 id="2-4-3-min-loss-select-best">2.4.3 MinLossSelectBest</h4>

Selects the best by the historical minimum loss; supports early stop.

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `mode` | `string` | Optional | `min_loss` | `min_loss` | Selection policy: min_loss | None |
| `early_stop_patience` | `int` | Optional | `-1` | ≥-1 | Number of consecutive iterations without a better snapshot before early stopping; -1 disables it | None |

**Configuration Constraints**

- None.

<h4 id="2-4-4-last-select-best">2.4.4 LastSelectBest</h4>

Saves only iteration 0 and the last iteration; no early stop.

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `mode` | `string` | Optional | `last` | `last` | Selection policy: last | None |

**Configuration Constraints**

- None.

## 3. Complete Configuration Reference

```yaml
apiversion: modelslim_v1
spec:
  process:
  - type: trainable_linear_quant
    operations:
    - type: minmax_tune
      lr: null
    - type: round_tune
      lr: null
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
    train_with_act_quant: false
    enable_quanted_input: false
    train_config:
      iters: 50
      gradient_accumulate_steps: 8
      lr: 0.01
      select_best:
        mode: ema
        ema_beta: 0.7
        ema_window_size: 5
        early_stop_patience: -1
      loss_type: l1
      train_seed: 42
```
