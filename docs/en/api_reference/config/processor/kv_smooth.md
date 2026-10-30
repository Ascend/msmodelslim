<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.processor.kv_smooth.processor.KVSmoothProcessorConfig -->
# kv_smooth Configuration Reference

## 1. Configuration Overview

KV cache smoothing processor configuration.

| Item | Content |
|------|------|
| Config Class | `KVSmoothProcessorConfig` |
| Source | [processor.py](../../../../../msmodelslim/processor/kv_smooth/processor.py) |

## 2. Parameter List

<h3 id="2-1-kv-smooth">2.1 KVSmoothProcessorConfig</h3>

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | Optional | `kv_smooth` | `kv_smooth` | Processor type, fixed to `kv_smooth`. | None |
| `smooth_factor` | `float` | Optional | `1.0` | — | KV smoothing factor; must be greater than 0; a larger value means stronger smoothing. | None |
| `include` | `list[string]` | Optional | `['*']` | — | Module name patterns to include; the default `*` matches all modules | None |
| `exclude` | `list[string]` | Optional | `[]` | — | Module name patterns to exclude; takes precedence over `include` | None |

**Configuration Constraints**

- None.

## 3. Complete Configuration Reference

```yaml
apiversion: modelslim_v1
spec:
  process:
  - type: kv_smooth
    smooth_factor: 1.0
    include:
    - '*'
    exclude: []
```
