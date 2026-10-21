<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.format.compressed_tensors_format.compressed_tensors.CompressedTensorsQuantFormatConfig -->
# compressed_tensors Configuration Reference

## 1. Configuration Overview

compressed_tensors save format configuration; exports safetensors weights and config.json.

| Item | Content |
|------|------|
| Config Class | `CompressedTensorsQuantFormatConfig` |
| Source | [compressed_tensors.py](../../../../../msmodelslim/format/compressed_tensors_format/compressed_tensors.py) |

## 2. Parameter List

<h3 id="2-1-compressed-tensors">2.1 CompressedTensorsQuantFormatConfig</h3>

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | Optional | `compressed_tensors` | `compressed_tensors` | Save format type, fixed to `compressed_tensors`. | None |
| `part_file_size` | `int` | Optional | `4` | — | Shard file size in GB; 0 means no sharding. | None |

**Configuration Constraints**

- None.

## 3. Complete Configuration Reference

```yaml
apiversion: modelslim_v1
spec:
  save:
  - type: compressed_tensors
    part_file_size: 4
```
