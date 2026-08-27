# Binary Fallback Tuning Algorithm

<!-- md-trans-meta sourceCommit=77830c6c459279f8875999ea1964ec2337a6273a translatedAt=2026-08-19T07:22:08.071Z pushedAt=2026-08-19T07:22:46.857Z -->

## Introduction

Binary Fallback is an auto-tuning strategy that determines the minimum quantization fallback configuration solely through **binary search**. Unlike Standing High, this strategy does **not** iterate over outlier suppression strategies, nor does it perform probing and layer reduction; processors such as outlier suppression are fixed in `template`, and the algorithm only adjusts the list-type fallback field pointed to by `rollback_path`.

Currently, only quantization configurations with `apiversion: modelslim_v1` are supported.

## Principle

1. **Zero-fallback evaluation**: Set the field corresponding to `rollback_path` to `[]` and evaluate it; if the accuracy meets the requirement, the process ends.

2. **Candidate list**:

   - When `rollback_candidates` is configured, the user-provided ordered list is used directly, and sensitive layer analysis is **skipped**.
   - When it is not configured, sensitive layer analysis is run (`quant_modules=["*"]`), and a list of layer names is obtained in descending order of sensitivity.

3. **Binary search**: Search for the **minimum** `k` in the candidate prefix `candidates[:k]` that meets the accuracy requirement.

4. **Termination**:

   - If the minimum `k` is found, output the corresponding `PracticeConfig`.
   - If the requirement is still not met even when `k = len(candidates)`, throw `SpecError`.

## Usage Instructions

In the `strategy` field of the auto-tuning YAML, set `type` to `binary_fallback`. For the complete command, see [Auto-tuning Usage Instructions](../../feature_guide/auto_precision_tuning/usage.md).

### YAML Configuration Example

```yaml
strategy:
  type: binary_fallback
  rollback_path: spec.process.1.exclude
  rollback_candidates: []   # Omit or leave empty to run sensitive layer analysis
  analysis_dataset: mix_calib.jsonl  # Optional; defaults to the value of the template.spec.dataset field
  template:
    apiversion: modelslim_v1
    metadata:
      config_id: qwen3-32b-w8a8-tune
      label:
        w_bit: 8
        a_bit: 8
        is_sparse: false
        kv_cache: false
    spec:
      runner: auto
      process:
        - type: iter_smooth
          alpha: 0.5
        - type: linear_quant
          qconfig:
            act:
              scope: per_tensor
              dtype: int8
              symmetric: false
              method: minmax
            weight:
              scope: per_channel
              dtype: int8
              symmetric: true
              method: minmax
          include: ["*"]
          exclude: []
      save:
        - type: ascendv1_saver
          part_file_size: 4
      dataset: mix_calib.jsonl
```

Specify rollback candidates (skip sensitive layer analysis):

```yaml
rollback_candidates:
  - "*model.layers.2.mlp.down_proj.*"
  - "*model.layers.5.mlp.down_proj.*"
```

### Field Description

| Field | Description |
|------|------|
| `type` | Fixed to `binary_fallback` |
| `template` | The complete best-practice `PracticeConfig` (including `apiversion`, `metadata`, and `spec`); `apiversion` must be `modelslim_v1` |
| `rollback_path` | A dot-separated path that specifies the target field within the template to be operated on by the algorithm. This field must be of **list type** (the algorithm writes the fallback layer names into this list), for example, `spec.process.1.exclude` |
| `rollback_candidates` | Optional ordered fallback candidates; if non-empty, sensitive layer analysis is skipped |
| `analysis_dataset` | Optional; the name of the calibration dataset for sensitive layer analysis. By default, the value of the `spec.dataset` field in the template is used |

### Comparison with Standing High

| Item | Standing High | Binary Fallback |
|----|---------------|-----------------|
| Outlier suppression | Iterates over `anti_outlier_strategies` | Fixed in `template.spec.process` |
| Fallback search | Binary search + probing and layer reduction | Binary search only |
| template | Only the spec fragment + outer metadata | Complete PracticeConfig |
| Fallback write | Hardcoded `linear_quant.exclude` | Specified by `rollback_path` |
| Maximum fallback still fails to meet the target | Returns the current best | `SpecError` |

## Applicable Requirements

**Inference engine**: Same as Standing High, the inference engine must support arbitrary layer fallback in the configuration (for example, vLLM-Ascend single-operator mode).

**Model adaptation**:

| Scenario | Requirement |
|------|------|
| Non-empty `rollback_candidates` configured | No additional model protocol requirement (sensitive layer analysis is skipped) |
| `rollback_candidates` not configured or empty | Must implement **`ModelSlimPipelineInterfaceV1`** (that is, the model protocol of the sensitive layer analysis service `PipelineAnalysisService`, which is the same as `PipelineInterface`); the calibration set is loaded by the analysis service through the `DatasetLoaderInfra` injected by the tuning layer |

This is consistent with the automatic sensitive layer analysis requirements of [Standing High](standing_high.md#application-requirements). For model integration, see [LLM Large Model Integration Guide — Auto-tuning and Sensitive Layer Analysis](../../../development_guide/integrating_models.md#auto-tuning-and-sensitive-layer-analysis).
