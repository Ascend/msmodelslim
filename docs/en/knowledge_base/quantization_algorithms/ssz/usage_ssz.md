# SSZ Parameter Configuration Workflow Guide

## 1. Scope

This guide is intended for users who need to use the [SSZ Weight Quantization Algorithm](./term_ssz.md). As a weight quantization method of the `linear_quant` processor, SSZ improves quantization accuracy in scenarios with uneven weight distributions by iteratively searching for the optimal scale/offset.

Applicable scenarios:

- W4 and other low-bit weight quantization accuracy does not meet the target and a better quantization scale needs to be found by iterative search;
- The weight quantization method needs to be replaced within the `linear_quant` pipeline to validate the benefit of low-bit quantization with minimal changes.

To check whether the model is on the officially pre-validated list, see the [Foundation Model Support Matrix](../../model/README.md); for a quick introduction to basic CLI usage, see the [One-Click Quantization Usage Guide](../../../user_guide/usage_one_click_quantization.md).

## 2. Inputs and Deliverables

| Type | Name | Source or Storage Location | Format or Constraints | Acceptance Criteria |
| --- | --- | --- | --- | --- |
| Input | Floating-point model weight directory | Model download or a local path | HuggingFace format containing `config.json` and `*.safetensors` shards | Loadable by the target Transformers version |
| Input | Calibration dataset | Built-in `lab_calib/` or a user-defined path | Text prompts in JSONL or JSON format; 50 samples recommended | Readable by the current quantization workflow and covering the main input forms |
| Input | Configuration constraints and practical baseline | This algorithm's configuration description, the target model's existing `lab_practice` recipe (if any) | Field names, supported combinations, scope, and model adaptation are consistent with the current version | The recommended starting point is traceable to the current configuration definition or a verified practice |
| Deliverable | SSZ parameter configuration plan | User quantization YAML or task configuration | Parameter values are valid and the scope of effect is explicit; the selection rationale of key parameters is explained | Can serve as algorithm configuration input for the subsequent quantization workflow and can reproduce the baseline selection in this guide |

## 3. Process Overview

The overall workflow of SSZ is as follows:

```mermaid
flowchart LR
    A[Determine the weight bit width] --> B[MinMax<br/>initialization]
    B --> C[Iteratively optimize<br/>scale/offset]
    C --> D[Quantize weights per channel]
    D --> E[Validate low-bit accuracy]
```

Key details of each stage are as follows:

- **Prerequisite: model adaptation.** Model adaptation must be completed before quantization. The adapter implements interfaces such as `PipelineInterface` (the basic pipeline interface); see the [LLM Model Quantization Integration Guide](../../ptq/llm/integration_guide_large_language_model_quantization.md) for details. Model adaptation must be completed first before the quantization workflow can run.
- **Determine the weight bit width**: first fix the target bit width (such as INT4 or INT8), then confirm the `dtype + scope + symmetric + method` combination supported by the current version; the bit width is decided by the product and backend targets and is not a minor tuning item.
- **MinMax initialization**: SSZ takes the MinMax quantization result as its initial point, obtaining the first round of quantization parameters as the starting point for iterative optimization.
- **Iteratively optimize scale/offset**: starting from the initial quantization parameters, alternately solve the least-squares scale with the current integer codes and then requantize; it can converge early when the error no longer improves.
- **Quantize weights per channel**: the current SSZ weight quantizer only registers `per_channel`, and each weight channel optimizes its scale independently (the offset is 0 when symmetric).
- **Validate low-bit accuracy**: tune only one variable per round and compare directly with the baseline; SSZ is mainly for low-bit weight scenarios, so for ordinary W8, first compare whether MinMax already meets the accuracy requirement.

## 4. Procedure

### Step 1: Confirm Objectives and Constraints

**Action**: First fix the target model, the final quantization format/bit width, the target objects, and the evaluation baseline, then confirm the parameter combinations supported by the current version. If the target model already has a `lab_practice` recipe, use that recipe as the practical baseline first; when there is no model-specific recipe, use the general recommended starting point given in this guide. If the algorithm relies on calibration statistics or optimization data, also fix representative data before tuning, to avoid mistaking a change in the data distribution for a parameter gain.

**Output**: An explicit configuration objective: target bit width/format, processing scope, calibration conditions, evaluation baseline, and deployment constraints.

### Step 2: Establish the Recommended Configuration

**Action**:

The following configuration establishes the **first comparable baseline**. If the target model already has a `lab_practice` recipe, prefer the verified recipe, then use this section to understand why each parameter is chosen this way.

```yaml
spec:
  process:
    - type: "linear_quant"
      qconfig:
        weight:
          scope: "per_channel"
          dtype: "int4"
          symmetric: true
          method: "ssz"
```

**Output**: A reproducible recommended configuration that serves as the comparison baseline for all subsequent parameter adjustments.

### Step 3: Select and Tune Parameters

**Action**:

| Configuration Item | Meaning (Rationale) | Recommended Configuration | Selection and Tuning Advice |
| --- | --- | --- | --- |
| `scope` | The current SSZ weight quantizer only registers `per_channel`. Each weight channel optimizes its scale independently (the offset is 0 when symmetric). | Fixed to `per_channel`. | When a granularity such as per-group is needed, choose another weight quantization algorithm. |
| `dtype` | SSZ currently registers INT8/INT4 symmetric per-channel. Low-bit INT4 benefits more easily from iterative scale optimization; for INT8, if MinMax already meets the accuracy requirement, the extra search can be skipped. | Use `int4` for a W4A8 target; for ordinary W8 weights, first compare whether MinMax is already sufficient. | The dtype is decided by the deployment target. Switching the bit width requires rerunning SSZ; do not reuse the scale of another bit width. |
| `symmetric` | Although the SSZ algorithm code contains asymmetric update logic, the current QABC registration only exposes the INT8/INT4 **symmetric per-channel** SSZ combination. | Fixed to `true`. | Do not change it to false based on the general QConfig enumeration; the current registration path does not provide the corresponding combination. |
| `method` | Fixed to `ssz`. Starting from the initial quantization parameters, the algorithm alternately solves the least-squares scale with the current integer codes and then requantizes; it can converge early when the error no longer improves. | Fixed. | Switching the method means switching the weight quantization algorithm. |
| `ext.step` | Controls the maximum number of SSZ main iterations. When it is not configured, the source code defaults to at most `50` iterations and has a strict early exit on MSE convergence; the FlexAWQSSZ W4A8 practice in the repository uses `10`. | For standalone SSZ with accuracy as the priority, it can be omitted to use the default of at most 50; when reusing the W4A8 practice or when the outer layer still performs extensive searching, `10` can be used. | A small `step` significantly reduces time, but if it is too small the iteration may stop before the scale has converged. When comparing 10 with the default, look at the final quantization MSE/task accuracy and whether the actual iterations often converge early, and then decide whether a larger value is needed. |

#### Parameter Combinations and Selection Order

The core degrees of freedom of SSZ are the bit width and `step`; scope/symmetric/method are basically fixed at present. For W4A8 scenarios, first determine the INT4 target, then decide between 10 and the more thorough default iterations according to whether it is inside a FlexAWQ outer search.

**Output**: An algorithm parameter plan that completes single-variable tuning, with a clear selection rationale and tuning direction for each key field.

### Step 4: Converge the Parameter Plan Based on Results

**Action**:

- SSZ is more suitable for 4-bit weight accuracy optimization. If the target is only W8, first compare whether simple MinMax already meets the accuracy requirement, to avoid unnecessary search overhead.
- **Run the recommended configuration first, then tune one variable at a time.** Do not change the bit width, granularity, algorithm parameters, and layer scope simultaneously; otherwise it is difficult to attribute an accuracy change to a specific item.
- **Prefer local fallback over raising accuracy globally.** If only a few layers are sensitive, use `exclude` or a mixed strategy to preserve high accuracy for them; this is usually more cost-effective than raising the bit width of the entire model.
- **The model's practical configuration and deployment capability take precedence.** The introductory recommendation is meant to establish a stable starting point; if the target model already has a `lab_practice` recipe, prefer reusing the verified combination.

**Output**: A final parameter plan ready for the subsequent quantization workflow, retaining a record of adjustments relative to the recommended configuration.

## 5. Terminology

| Term | Brief Description | Link |
| --- | --- | --- |
| SSZ Weight Quantization Algorithm | Explains the definition, core principle, key properties, applicable scenarios, and limitations of the algorithm. | [SSZ Weight Quantization Algorithm Terminology Entry](./term_ssz.md) |

## 6. Related Documents

| API or Document | Brief Description | Link |
| --- | --- | --- |
| linear_quant Configuration Description | Field types, default values, valid values, and complete configuration constraints. | [linear_quant Configuration Description](../../../api_reference/config/processor/linear_quant.md) |
| modelslim_v1 Configuration Description | Consult this when you need to explore task-level advanced configurations such as runner, prior, save, and dataset. | [modelslim_v1 Configuration Description](../../../api_reference/config/quant/modelslim_v1.md) |
| Weight Quantization Usage Guide | User guide: quantization command parameters and complete usage instructions. | [Weight Quantization Usage Guide](../../../user_guide/usage_weight_quantization.md) |
