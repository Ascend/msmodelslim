<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.processor.anti_outlier.oasq.processor.OASQProcessorConfig -->
# oasq Configuration Reference

## 1. Configuration Overview

OASQ (Outlier-Aware Smooth Quantization) processor configuration.

| Item | Content |
|------|------|
| Config Class | `OASQProcessorConfig` |
| Source | [processor.py](../../../../../msmodelslim/processor/anti_outlier/oasq/processor.py) |

## 2. Parameter List

<h3 id="2-1-oasq">2.1 OASQProcessorConfig</h3>

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | Optional | `oasq` | `oasq` | Processor type, fixed to `oasq`. | None |
| `max_iters` | `int / null` | Optional | `null` | — | Maximum number of iterations; if unset, the implementation default is used; must be greater than 0. | None |
| `symmetric` | `bool` | Optional | `true` | — | Whether to symmetrically quantize the subsequent weights/activations. | None |
| `enable_subgraph_type` | `list[any]` | Optional | `['norm-linear', 'linear-linear', 'ov', 'up-down']` | — | List of subgraph types for OASQ; the default is `norm-linear`, `linear-linear`, `ov`, `up-down`. | None |
| `include` | `list[string] / null` | Optional | `null` | — | Module name patterns to include; if unset, all modules match. | None |
| `exclude` | `list[string] / null` | Optional | `null` | — | Module name patterns to exclude; takes precedence over `include`. | None |

**Configuration Constraints**

- Validates max_iters: when set, it must be greater than 0.

## 3. Complete Configuration Reference

```yaml
apiversion: modelslim_v1
spec:
  process:
  - type: oasq
    symmetric: true
    enable_subgraph_type:
    - norm-linear
    - linear-linear
    - ov
    - up-down
```
