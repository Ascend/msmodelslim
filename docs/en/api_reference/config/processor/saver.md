<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.processor.save.processor.QuantSaveProcessorConfig -->
# saver Configuration Reference

## 1. Configuration Overview

Unified save processor configuration.

| Item | Content |
|------|------|
| Config Class | `QuantSaveProcessorConfig` |
| Source | [processor.py](../../../../../msmodelslim/processor/save/processor.py) |

## 2. Parameter List

<h3 id="2-1-saver">2.1 QuantSaveProcessorConfig</h3>

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | Optional | `saver` | `saver` | Processor type, fixed to `saver`. | None |
| `format` | `object` | Required | None | — | Export format configuration (a single object), injected automatically by the save processor. | [`QuantFormatConfig`](../../../../../msmodelslim/format/base.py) |

**Configuration Constraints**

- None.

## 3. Complete Configuration Reference

```yaml
apiversion: modelslim_v1
spec:
  process:
  - type: saver
    format:
      type: _auto_save
```
