# Std Parameter Configuration Workflow Guide

## 1. Scope

This guide is intended for users who need to use the [Std Sensitive Layer Analysis Algorithm](./term_std.md). As a metrics indicator of `msmodelslim analyze linear`, Std is used for coarse screening and ranking of sensitivity at linear-layer granularity.

Applicable scenarios:

- Need to quickly scan the linear layers of the whole model at low cost and obtain a coarse-grained sensitivity ranking as the starting point for subsequent validation;
- Need a sensitivity proxy metric that depends only on the activation distribution and can be interpreted without quantization simulation.

To check whether the model is on the officially pre-validated list, see the [Foundation Model Support Matrix](../../model/README.md); for a quick introduction to basic CLI usage, see the [One-Click Quantization Usage Guide](../../../user_guide/usage_one_click_quantization.md).

## 2. Inputs and Deliverables

| Type | Name | Source or Storage Location | Format or Constraints | Acceptance Criteria |
| --- | --- | --- | --- | --- |
| Input | Floating-point model weight directory | Model download or a local path | HuggingFace format containing `config.json` and `*.safetensors` shards | Loadable by the target Transformers version |
| Input | Calibration dataset | Built-in `lab_calib/` or a user-defined path | Text prompts in JSONL or JSON format; 50 samples recommended | Readable by the current quantization workflow and encodable into forward tensors |
| Input | Analysis parameter constraints | Sensitive layer analysis guide, current-version CLI parameters | Metrics, `patterns`/scope, and candidate counts use values supported by the current version | The same data, metric definition, and analysis scope are used within one comparison round |
| Deliverable | Std analysis configuration plan | Sensitive layer analysis command/configuration or review record | Records the metric, analysis scope, number of candidates, and selection rationale; the result serves as candidate input for subsequent fallback or mixed precision | The same analysis conditions can be reproduced stably, and it explains how the candidates are used for subsequent quantization decisions |

## 3. Process Overview

The overall workflow of Std is as follows:

```mermaid
flowchart LR
    A[Determine the linear-layer scope] --> B[Fix<br/>metrics=std]
    B --> C[Measure the<br/>dynamic range and standard deviation]
    C --> D[Rank by<br/>score]
    D --> E[Select fallback candidates]
```

Key details of each stage are as follows:

- **Prerequisite: model adaptation.** Model adaptation must be completed before analysis. The adapter implements interfaces such as `PipelineInterface` (the basic pipeline interface); see the [LLM Model Quantization Integration Guide](../../ptq/llm/integration_guide_large_language_model_quantization.md) for details. Model adaptation must be completed first before the analysis workflow can run.
- **Determine the linear-layer scope**: use `--patterns` to select the name patterns of the linear modules to be measured. For the first run, it is recommended to cover a scope consistent with the actual quantization plan.
- **Fix metrics=std**: `--metrics std` selects the analysis metric corresponding to this guide, which determines the definition of the score and the meaning of the ranking.
- **Measure the dynamic range and standard deviation**: compute `abs_max / std` on the input activations of linear/convolutional modules; a high score means the largest magnitude stands out more against the typical degree of dispersion, and it is often used to quickly identify layers whose quantization range may be widened by a few extremes.
- **Rank by score**: sort the scores of the individual linear layers to obtain a list of sensitivity candidates; when std is very small, the implementation degenerates to abs_max, so a uniform absolute threshold should not be used across layers or models.
- **Select fallback candidates**: treat the Top-K high-scoring objects as the "fallback/bit-width-increase candidate pool"; whether to actually fall back should still be verified against actual quantization accuracy or task metrics.

## 4. Procedure

### Step 1: Confirm Objectives and Constraints

**Action**: Fix the model to analyze, the formal quantization baseline, the calibration data, and the analysis scope. First confirm whether this analysis is meant to screen linear layers, Decoder layers, Attention structures, or other candidate objects, and align the analysis scope with the subsequent actual quantization scope. If the model, calibration data, quantization baseline, or candidate scope change at the same time, the score ranking becomes hard to attribute, so these conditions should be fixed first before adjusting the metric or the number of candidates.

**Output**: A fixed analysis context: model version, quantization baseline, calibration data, analysis scope, and the fallback/mixed-precision problem to be solved.

### Step 2: Establish the Recommended Configuration

**Action**:

- Analysis scope: `linear`;
- `--metrics`: `std`;
- `--top_k`: start with `15`;
- `--patterns`: start with `"*"` to establish a baseline over all linear layers;

**Output**: A reproducible set of sensitivity analysis baseline parameters used to generate the first candidate ranking.

### Step 3: Select and Tune Parameters

**Action**:

| Configuration Item | Meaning (Rationale) | Recommended Configuration | When to Tune |
| --- | --- | --- | --- |
| Analysis scope `linear` | Computes `abs_max / std` on the input activations of linear/convolutional modules. A high score means the largest magnitude stands out more against the typical degree of dispersion, and it is often used to quickly identify layers whose quantization range may be widened by a few extremes. | Fixed to `linear`. | Fixed to linear; this is a fast statistical proxy rather than a real quantization intervention. |
| `--metrics` | Selects the analysis metric `std` corresponding to this guide; the metric determines the definition of the score and the meaning of the ranking. | Fixed to `std`. | Fixed to `std`. Its score depends on the maximum value and the standard deviation; when std is very small, the implementation degenerates to abs_max, so a uniform absolute threshold should not be used across layers or models. |
| `--patterns` | Restricts the module name patterns to be measured. | `["*"]` for the first run; for formal analysis, align it with the modules actually quantized. | The first analysis should cover a candidate scope consistent with the actual quantization plan; narrow it only after confirming that a certain type of module will not be quantized. A change in scope changes the candidate set, so rankings obtained under different scopes should not be compared directly as if they came from the same experiment. |
| `--calibration_dataset` | Calibration data used to collect activations or floating-point/quantized outputs during the forward pass, in **JSON/JSONL format** (an LLM text-sample file such as `mix_calib.jsonl`; for VLM it is a multimodal directory). The data distribution directly determines the statistics, MSE, or head scores, so it is a core input that affects analysis validity rather than a plain file path. | Prefer a calibration set that is **the same as, or drawn from the same distribution as,** the subsequent formal quantization; 50 samples are recommended. Use the built-in example only for quick verification. | If the ranking changes noticeably across datasets, first check whether the calibration set covers the real sequence lengths, topics, and input forms before considering a larger `top_k`. Formal fallback decisions should be made on representative data whenever possible. |
| `--top_k` | Controls the number of high-scoring candidates finally displayed or exported. Default: `15`. It does not change the underlying score computation and does not make the algorithm analyze more data. | Start from `15`. | Candidates are only a set for subsequent validation: increase the value when more fallback candidates are needed, and decrease it for quick manual review. Do not treat `top_k` as a sensitivity threshold; the absolute score scale differs across models and metrics. |

#### Parameter Combinations and Result Interpretation

- **Fix the calibration set and quantization baseline first, then compare rankings.** If the data, quantization configuration, and analysis scope are changed at the same time, the change in score cannot be attributed.
- **A high score means "more noteworthy under the `std` definition" and is not equivalent to a mandatory fallback**. The Top-K should be treated as a candidate set and then validated with actual quantization accuracy or model task metrics.
- **`top_k` only changes the output size.** What actually determines *who ranks first* is usually the calibration data, the candidate scope, and the statistical definition of the metric itself.

Std has low cost and direct interpretability, making it suitable as the first round of coarse screening; for very heavy-tailed distributions, Kurtosis/Quantile often supplement more shape information.

**Output**: An analysis configuration with clear parameter meanings and tuning directions, where each modification can be mapped to a change in the ranking.

### Step 4: Interpret Results and Converge the Candidate Plan

**Action**:

- Objects with a higher score enter the "fallback/bit-width-increase candidate pool" first, but whether to actually fall back should still be decided together with end-to-end accuracy validation.
- `top_k` is only the number of candidates displayed, not an accuracy threshold; do not mechanically treat a position such as the 15th as the fallback boundary.
- The analysis scope (`patterns`/`quant_modules`) should be as consistent as possible with the actual quantization scope; otherwise the resulting ranking cannot accurately guide the final YAML.

**Output**: A set of sensitive-object candidates validated by actual quantization or task metrics, together with corresponding suggestions for local fallback, bit-width increase, or protection.

## 5. Terminology

| Term | Brief Description | Link |
| --- | --- | --- |
| Std Sensitive Layer Analysis Algorithm | Explains the definition, core principle, key properties, applicable scenarios, and limitations of the algorithm. | [Std Sensitive Layer Analysis Algorithm Terminology Entry](./term_std.md) |

## 6. Related Documents

| API or Document | Brief Description | Link |
| --- | --- | --- |
| Sensitive Layer Analysis Usage Guide | Complete CLI, inputs and outputs, and advanced workflows. | [Sensitive Layer Analysis Usage Guide](../../../user_guide/usage_sensitive_linear_analysis.md) |
| modelslim_v1 Configuration Description | Consult this when you need to explore task-level advanced configurations such as runner, prior, save, and dataset. | [modelslim_v1 Configuration Description](../../../api_reference/config/quant/modelslim_v1.md) |
| Weight Quantization Usage Guide | User guide: quantization command parameters and complete usage instructions. | [Weight Quantization Usage Guide](../../../user_guide/usage_weight_quantization.md) |
