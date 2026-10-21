<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.processor.anti_outlier.iter_smooth.processor.IterSmoothProcessorConfig -->
# iter_smooth Configuration Reference

## 1. Configuration Overview

Iterative smooth (IterativeSmooth) processor configuration.

| Item | Content |
|------|------|
| Config Class | `IterSmoothProcessorConfig` |
| Source | [processor.py](../../../../../msmodelslim/processor/anti_outlier/iter_smooth/processor.py) |

## 2. Parameter List

<h3 id="2-1-iter-smooth">2.1 IterSmoothProcessorConfig</h3>

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | Optional | `iter_smooth` | `iter_smooth` | Processor type, fixed to `iter_smooth`. | None |
| `alpha` | `float` | Optional | `0.9` | — | Smooth migration strength (0-1); a larger value migrates more outliers to the weights. | None |
| `scale_min` | `float` | Optional | `1e-05` | — | Lower bound for the smoothing scale factor, preventing numerical underflow. | None |
| `symmetric` | `bool` | Optional | `true` | — | Whether to symmetrically quantize the subsequent weights/activations. | None |
| `enable_subgraph_type` | `list[any]` | Optional | `['norm-linear', 'linear-linear', 'ov', 'up-down']` | — | List of subgraph types for iterative smoothing; the default is `norm-linear`, `linear-linear`, `ov`, `up-down`. | None |
| `include` | `list[string] / null` | Optional | `null` | — | Module name patterns to include; if unset, all modules match. | None |
| `exclude` | `list[string] / null` | Optional | `null` | — | Module name patterns to exclude; takes precedence over `include`. | None |

**Configuration Constraints**

- None.

## 3. Complete Configuration Reference

```yaml
apiversion: modelslim_v1
spec:
  process:
  - type: iter_smooth
    alpha: 0.9
    scale_min: 1.0e-05
    symmetric: true
    enable_subgraph_type:
    - norm-linear
    - linear-linear
    - ov
    - up-down
```
