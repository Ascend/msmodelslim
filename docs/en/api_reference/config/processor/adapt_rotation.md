<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.processor.adapt_rotation.adapt_rotation.AdaptRotationProcessorConfig -->
# adapt_rotation Configuration Reference

## 1. Configuration Overview

Adaptive rotation (adapt_rotation) processor configuration.

| Item | Content |
|------|------|
| Config Class | `AdaptRotationProcessorConfig` |
| Source | [adapt_rotation.py](../../../../../msmodelslim/processor/adapt_rotation/adapt_rotation.py) |

## 2. Parameter List

<h3 id="2-1-adapt-rotation">2.1 AdaptRotationProcessorConfig</h3>

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | Optional | `adapt_rotation` | `adapt_rotation` | Processor type, fixed to `adapt_rotation`. | None |
| `stage` | `int` | Required | None | `1`, `2` | Rotation adaptation stage: 1 or 2; determines which stage configuration is used. | None |
| `stage_config` | `object` | Required | None | — | Stage configuration object (an internally assembled field; it is required but generated automatically by a before-validator based on `stage`, so users do not configure it in YAML); do not configure this field directly in YAML. Instead, flatten the stage fields (such as steps and quant_dtype) under the processor; see <a href="#2-2-1-adapt-rotation-stage1">AdaptRotationStage1ProcessorConfig Configuration Reference</a> / <a href="#2-2-2-adapt-rotation-stage2">AdaptRotationStage2ProcessorConfig Configuration Reference</a>. | This page <a href="#2-2-processorconfig">§2.2</a> |

**Configuration Constraints**

- Assembles the flat fields into stage_config by stage: only fields of the corresponding stage are allowed; extra fields raise an error.

<h3 id="2-2-processorconfig">2.2 <a href="linear_quant.md">ProcessorConfig</a></h3>

**Derived Classes**

| Config Class | Description | Document |
|--------|------|------|
| `AdaptRotationStage1ProcessorConfig` | Configuration of adapt_rotation stage 1. | This page <a href="#2-2-1-adapt-rotation-stage1">§2.2.1</a> |
| `AdaptRotationStage2ProcessorConfig` | Configuration of adapt_rotation stage 2. | This page <a href="#2-2-2-adapt-rotation-stage2">§2.2.2</a> |

<h4 id="2-2-1-adapt-rotation-stage1">2.2.1 AdaptRotationStage1ProcessorConfig</h4>

Configuration of adapt_rotation stage 1.

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | Optional | `_adapt_rotation_stage1` | `_adapt_rotation_stage1` | Stage configuration type; an internal identifier that does not need to be configured. | None |
| `steps` | `int` | Optional | `20` | — | Number of iterative optimization steps | None |
| `quant_dtype` | `string` | Optional | `int4` | `int4`, `int8` | Quantization bit width; it should match the activation quantization type used downstream (for example int4 for w4a4 and int8 for w8a8) | None |
| `layer_type` | `list[string]` | Optional | `['up_proj']` | At least 1 item | List of layer name substrings whose activations are collected | None |
| `block_size` | `int` | Optional | `-1` | — | Block size; -1 means hidden_dim | None |
| `max_samples` | `int` | Optional | `2048` | — | Maximum number of samples per layer | None |

**Configuration Constraints**

- Validates layer_type: each element must be a non-empty string of length <= 128
- Validates block_size: the value must be -1 or a non-negative integer power of 2

<h4 id="2-2-2-adapt-rotation-stage2">2.2.2 AdaptRotationStage2ProcessorConfig</h4>

Configuration of adapt_rotation stage 2.

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | Optional | `_adapt_rotation_stage2` | `_adapt_rotation_stage2` | Stage configuration type; an internal identifier that does not need to be configured. | None |
| `online` | `bool` | Optional | `false` | — | Whether to enable online rotation | None |
| `block_size` | `int` | Optional | `-1` | — | Block size; -1 means hidden_dim | None |
| `down_proj_online_layers` | `list[int]` | Optional | `[]` | — | Index list of online down_proj layers | None |
| `max_tp_size` | `int` | Optional | `4` | — | Maximum TP parallelism | None |

**Configuration Constraints**

- Validates down_proj_online_layers: each element must be a non-negative integer
- Validates max_tp_size: it must be at least 1 and a power of 2
- Validates block_size: the value must be -1 or a non-negative integer power of 2

## 3. Complete Configuration Reference

```yaml
apiversion: modelslim_v1
spec:
  prior:
  - process:
    - type: adapt_rotation
      stage: 1
      steps: 20
      quant_dtype: int4
      layer_type:
      - up_proj
      block_size: -1
      max_samples: 2048
  process:
  - type: adapt_rotation
    stage: 2
    online: false
    block_size: -1
    max_tp_size: 1
```
