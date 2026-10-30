<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.core.practice.interface.PracticeConfig -->
# PracticeConfig Configuration Reference

## 1. Configuration Overview

Complete best-practice quantization task configuration: metadata + task (task ≡ apiversion + spec).

| Item | Content |
|------|------|
| Config Class | `PracticeConfig` |
| Source | [interface.py](../../../../../msmodelslim/core/practice/interface.py) |

## 2. Parameter List

<h3 id="2-1-practice-config">2.1 PracticeConfig</h3>

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `metadata` | `object` | Optional | See the nested configuration default | — | Quantization configuration metadata (config_id/score/label/verified_*) | This page <a href="#2-2-metadata">§2.2</a> |
| `task` | `object` | Optional | See the nested configuration default | — | Quantization task description (apiversion + spec) | This page <a href="#2-3-basequantconfig">§2.3</a> |

**Configuration Constraints**

- Preprocessing: the old format {apiversion, metadata, spec, ...} becomes {metadata, task:{apiversion, spec, ...}}.
- Strips the aggregated task. prefix so that error paths match the spec. paths in the YAML.

<h3 id="2-2-metadata">2.2 Metadata</h3>

Metadata describing a quantization configuration; the label and scenario fields index it for the best-practice library.

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `config_id` | `string` | Optional | `Unknown` | — | Quantization configuration ID, for example 'Qwen3-32B W8A8' | None |
| `score` | `float` | Optional | `100.0` | — | Quantization configuration score, used for sorting; must be >= 0 | None |
| `label` | `object` | Optional | `{}` | — | Quantization configuration labels, used for filtering, for example: `w_bit` (int, weight quantization bit width), `a_bit` (int, activation quantization bit width), `is_sparse` (bool, whether sparse quantization is used), `kv_cache` (bool, whether KV Cache quantization is enabled), `fa_quant` (bool, whether FA quantization is enabled), `fa_bit` (int, FA quantization bit width) | None |
| `verified_model_types` | `list[string]` | Optional | `[]` | — | List of verified model types, for example ['LLaMa3.1-70B', 'Qwen2.5-72B'] | None |
| `verified_tags` | `object` | Optional | `{}` | — | Verified scenario tags: keys are model types, values are lists of scenario tags (each scenario is a set of tags, such as ['MindIE','Atlas_A2_Inference']) | None |

**Configuration Constraints**

- None.

<h3 id="2-3-basequantconfig">2.3 BaseQuantConfig</h3>

**Derived Classes**

| Config Class | Description | Document |
|--------|------|------|
| `ModelslimV1QuantConfig` | `modelslim_v1` quantization task configuration, located at the YAML root. | [modelslim_v1 Configuration Reference](modelslim_v1.md) |
| `ModelslimConvertQuantConfig` | `modelslim_convert` quantization (weight conversion) task configuration, located at the YAML root. | [modelslim_convert Configuration Reference](modelslim_convert.md) |

## 3. Complete Configuration Reference

```yaml
metadata:
  config_id: Qwen3-32B W8A8
  label:
    w_bit: 8
    a_bit: 8
    is_sparse: false
    kv_cache: false
  verified_model_types:
  - Qwen3-32B
task:
  apiversion: modelslim_v1
  spec:
    process:
    - type: linear_quant
      qconfig:
        act:
          dtype: int8
          scope: per_tensor
          symmetric: false
          method: minmax
        weight:
          dtype: int8
          scope: per_channel
          symmetric: true
          method: minmax
      include:
      - '*'
      exclude: []
    save:
    - type: ascendv1_saver
      part_file_size: 4
    dataset: mix_calib.jsonl
```
