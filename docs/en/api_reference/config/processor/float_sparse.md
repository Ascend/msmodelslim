<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.processor.sparse.float_sparse.FloatSparseProcessorConfig -->
# float_sparse Configuration Reference

## 1. Configuration Overview

Floating-point sparsity processor configuration.

| Item | Content |
|------|------|
| Config Class | `FloatSparseProcessorConfig` |
| Source | [float_sparse.py](../../../../../msmodelslim/processor/sparse/float_sparse.py) |

## 2. Parameter List

<h3 id="2-1-float-sparse">2.1 FloatSparseProcessorConfig</h3>

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | Optional | `float_sparse` | `float_sparse` | Processor type, fixed to `float_sparse`. | None |
| `sparse_ratio` | `float` | Optional | `0.3` | — | Sparsity ratio (0-1); the fraction of weights set to zero; a larger value means more sparsity. | None |
| `include` | `list[string]` | Optional | `[]` | — | Module name patterns to include | None |
| `exclude` | `list[string]` | Optional | `[]` | — | Module name patterns to exclude; takes precedence over `include` | None |

**Configuration Constraints**

- None.

## 3. Complete Configuration Reference

```yaml
apiversion: modelslim_v1
spec:
  process:
  - type: float_sparse
    sparse_ratio: 0.3
    include: []
    exclude: []
```
