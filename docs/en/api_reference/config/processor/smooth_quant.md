<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.processor.anti_outlier.smooth_quant.processor.SmoothQuantProcessorConfig -->
# smooth_quant Configuration Reference

## 1. Configuration Overview

SmoothQuant smooth quantization processor configuration.

| Item | Content |
|------|------|
| Config Class | `SmoothQuantProcessorConfig` |
| Source | [processor.py](../../../../../msmodelslim/processor/anti_outlier/smooth_quant/processor.py) |

## 2. Parameter List

<h3 id="2-1-smooth-quant">2.1 SmoothQuantProcessorConfig</h3>

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | Optional | `smooth_quant` | `smooth_quant` | Processor type, fixed to `smooth_quant`. | None |
| `alpha` | `float` | Optional | `0.5` | — | Smooth migration strength (0-1); a larger value migrates more activation outliers to the weights. | None |
| `symmetric` | `bool` | Optional | `true` | — | Whether to use symmetric quantization; affects the symmetry of the quantized result after smoothing. | None |
| `include` | `list[string] / null` | Optional | `null` | — | Module name patterns to include; if unset, all modules match. | None |
| `exclude` | `list[string] / null` | Optional | `null` | — | Module name patterns to exclude; takes precedence over `include`. | None |

**Configuration Constraints**

- None.

## 3. Complete Configuration Reference

```yaml
apiversion: modelslim_v1
spec:
  process:
  - type: smooth_quant
    alpha: 0.5
    symmetric: true
```
