<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.processor.svd_residual.processor.SVDResidualProcessorConfig -->
# svd_res Configuration Reference

## 1. Configuration Overview

SVD residual (low-rank compensation) processor configuration.

| Item | Content |
|------|------|
| Config Class | `SVDResidualProcessorConfig` |
| Source | [processor.py](../../../../../msmodelslim/processor/svd_residual/processor.py) |

## 2. Parameter List

<h3 id="2-1-svd-res">2.1 SVDResidualProcessorConfig</h3>

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | Optional | `svd_res` | `svd_res` | Processor type, fixed to `svd_res`. | None |
| `rank` | `int` | Optional | `32` | >0 | Rank of the low-rank decomposition; must be greater than 0 | None |
| `include` | `list[string]` | Optional | `['*']` | — | Module name patterns to include; the default `*` matches all modules | None |
| `exclude` | `list[string]` | Optional | `[]` | — | Module name patterns to exclude; takes precedence over `include` | None |

**Configuration Constraints**

- None.

## 3. Complete Configuration Reference

```yaml
apiversion: modelslim_v1
spec:
  process:
  - type: svd_res
    rank: 32
    include:
    - '*'
    exclude: []
```
