<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.processor.anti_outlier.flex_smooth.processor.FlexSmoothQuantProcessorConfig -->
# flex_smooth_quant Configuration Reference

## 1. Configuration Overview

FlexSmoothQuant smooth quantization processor configuration.

| Item | Content |
|------|------|
| Config Class | `FlexSmoothQuantProcessorConfig` |
| Source | [processor.py](../../../../../msmodelslim/processor/anti_outlier/flex_smooth/processor.py) |

## 2. Parameter List

<h3 id="2-1-flex-smooth-quant">2.1 FlexSmoothQuantProcessorConfig</h3>

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | Optional | `flex_smooth_quant` | `flex_smooth_quant` | Processor type, fixed to `flex_smooth_quant`. | None |
| `alpha` | `float` | Optional | `null` | — | Strength of the activation-to-weight smooth migration (0-1); a higher value migrates more outliers to the weights. | None |
| `beta` | `float` | Optional | `null` | — | Additional smoothing tuning coefficient (0-1), used together with alpha. | None |
| `enable_subgraph_type` | `list[any]` | Optional | `['norm-linear', 'linear-linear', 'ov', 'up-down']` | — | List of subgraph types to smooth; the default is `norm-linear`, `linear-linear`, `ov`, `up-down`. | None |
| `include` | `list[string] / null` | Optional | `null` | — | Module name patterns to include; if unset, all modules match. | None |
| `exclude` | `list[string] / null` | Optional | `null` | — | Module name patterns to exclude; takes precedence over `include`. | None |

**Configuration Constraints**

- None.

## 3. Complete Configuration Reference

```yaml
apiversion: modelslim_v1
spec:
  process:
  - type: flex_smooth_quant
    enable_subgraph_type:
    - norm-linear
    - linear-linear
    - ov
    - up-down
```
