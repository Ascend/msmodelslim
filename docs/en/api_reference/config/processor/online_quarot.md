<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.processor.quarot.online_quarot.online_quarot.OnlineQuaRotProcessorConfig -->
# online_quarot Configuration Reference

## 1. Configuration Overview

Online QuaRot rotation processor configuration.

| Item | Content |
|------|------|
| Config Class | `OnlineQuaRotProcessorConfig` |
| Source | [online_quarot.py](../../../../../msmodelslim/processor/quarot/online_quarot/online_quarot.py) |

## 2. Parameter List

<h3 id="2-1-online-quarot">2.1 OnlineQuaRotProcessorConfig</h3>

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | Optional | `online_quarot` | `online_quarot` | Processor type, fixed to `online_quarot`. | None |
| `include` | `list[string] / null` | Optional | `null` | — | Module name patterns to include; if unset, all modules match. | None |
| `exclude` | `list[string] / null` | Optional | `null` | — | Module name patterns to exclude; takes precedence over `include`. | None |
| `block_size` | `int` | Optional | `-1` | — | Rotation block size; -1 means rotating the whole hidden_dim as one block. Can be overridden by RotationConfig. | None |

**Configuration Constraints**

- Validates block_size: the value must be -1 or a non-negative integer power of 2

## 3. Complete Configuration Reference

```yaml
apiversion: modelslim_v1
spec:
  process:
  - type: online_quarot
    block_size: -1
```
