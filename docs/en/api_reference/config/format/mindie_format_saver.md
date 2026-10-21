<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.format.mindie_format.mindie.MindIEQuantFormatConfig -->
# mindie_format_saver Configuration Reference

## 1. Configuration Overview

MindIE save format configuration; exports weight files in the MindIE on-disk format.

| Item | Content |
|------|------|
| Config Class | `MindIEQuantFormatConfig` |
| Source | [mindie.py](../../../../../msmodelslim/format/mindie_format/mindie.py) |

## 2. Parameter List

<h3 id="2-1-mindie-format-saver">2.1 MindIEQuantFormatConfig</h3>

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | Optional | `mindie_format_saver` | `mindie_format_saver` | Save format type, fixed to `mindie_format_saver`. | None |
| `part_file_size` | `int` | Optional | `4` | — | Shard file size in GB; 0 means no sharding. | None |
| `ext` | `object` | Optional | `{}` | — | Save format extension parameters, determined by the implementation; an empty object means no extension parameters. | None |

**Configuration Constraints**

- None.

## 3. Complete Configuration Reference

```yaml
apiversion: multimodal_sd_modelslim_v1
spec:
  save:
  - type: mindie_format_saver
    part_file_size: 4
    ext: {}
  multimodal_sd_config:
    dump_config:
      enable_dump: true
      capture_mode: args
      dump_data_dir: ''
```
