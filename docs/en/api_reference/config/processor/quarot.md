<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.processor.quarot.offline_quarot.quarot.QuaRotProcessorConfig -->
# quarot Configuration Reference

## 1. Configuration Overview

QuaRot (offline rotation) processor configuration.

| Item | Content |
|------|------|
| Config Class | `QuaRotProcessorConfig` |
| Source | [quarot.py](../../../../../msmodelslim/processor/quarot/offline_quarot/quarot.py) |

## 2. Parameter List

<h3 id="2-1-quarot">2.1 QuaRotProcessorConfig</h3>

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | Optional | `quarot` | `quarot` | Processor type, fixed to `quarot`. | None |
| `online` | `bool` | Optional | `false` | — | Whether to use online rotation (offline by default). | None |
| `block_size` | `int` | Optional | `-1` | — | Rotation block size; -1 means rotating the whole hidden_dim as one block. | None |
| `down_proj_online_layers` | `list[int]` | Optional | `[]` | — | Index list of down_proj layers that need online rotation. | None |
| `max_tp_size` | `int` | Optional | `4` | — | Maximum TP (Tensor Parallelism) degree; must be a power of 2. | None |
| `export_extra_info` | `bool` | Optional | `true` | — | Whether to export the `optional.quarot.global_rotation` rotation information for downstream deployment. | None |

**Configuration Constraints**

- Validates max_tp_size: it must be at least 1 and a power of 2
- Validates block_size: the value must be -1 or a non-negative integer power of 2

## 3. Complete Configuration Reference

```yaml
apiversion: modelslim_v1
spec:
  process:
  - type: quarot
    online: false
    block_size: -1
    down_proj_online_layers: []
    max_tp_size: 4
    export_extra_info: true
```
