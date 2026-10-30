<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.format.ascendV1_format.ascendV1.AscendV1QuantFormatConfig -->
# ascendv1_saver Configuration Reference

## 1. Configuration Overview

AscendV1 save format configuration; exports weight files in the Ascend on-disk format.

| Item | Content |
|------|------|
| Config Class | `AscendV1QuantFormatConfig` |
| Source | [ascendV1.py](../../../../../msmodelslim/format/ascendV1_format/ascendV1.py) |

## 2. Parameter List

<h3 id="2-1-ascendv1-saver">2.1 AscendV1QuantFormatConfig</h3>

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | Optional | `ascendv1_saver` | `ascendv1_saver` | Save format type, fixed to `ascendv1_saver`. | None |
| `part_file_size` | `int` | Optional | `4` | — | Shard file size in GB; 0 means no sharding. | None |
| `ext` | `object` | Optional | `{}` | — | Save format extension parameters, determined by the implementation; an empty object means no extension parameters. | None |

**Configuration Constraints**

- None.

## 3. Complete Configuration Reference

```yaml
apiversion: modelslim_v1
spec:
  save:
  - type: ascendv1_saver
    part_file_size: 4
    ext: {}
```
