<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.processor.memory.load.LoadProcessorConfig -->
# load Configuration Reference

## 1. Configuration Overview

Module load/unload processor configuration.

| Item | Content |
|------|------|
| Config Class | `LoadProcessorConfig` |
| Source | [load.py](../../../../../msmodelslim/processor/memory/load.py) |

## 2. Parameter List

<h3 id="2-1-load">2.1 LoadProcessorConfig</h3>

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | Optional | `load` | `load` | Processor type, fixed to `load`. | None |
| `device` | `string` | Optional | `cpu` | — | Target device, such as `cpu` or `npu` | None |
| `non_blocking` | `bool` | Optional | `false` | — | Whether to load in a non-blocking manner | None |
| `mode` | `string` | Optional | `load` | `load`, `offload` | Loading mode: `load` loads to the target device, and `offload` offloads to CPU | None |
| `cleanup` | `bool` | Optional | `false` | — | Whether to clear the cache | None |
| `post_offload` | `bool` | Optional | `false` | — | Whether to offload activations after unloading | None |

**Configuration Constraints**

- None.

## 3. Complete Configuration Reference

```yaml
apiversion: modelslim_v1
spec:
  process:
  - type: load
    device: cpu
    non_blocking: false
    mode: load
    cleanup: false
    post_offload: false
```
