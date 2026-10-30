# Histogram Parameter Configuration Workflow Guide

## 1. Scope

This guide is intended for users who need to use the [Histogram-Based Activation Quantization Algorithm](./term_histogram_activation_quantization.md). As an activation quantization method of the `linear_quant` processor, Histogram automatically searches for the optimal clipping interval through histogram analysis, improving the accuracy of static activation quantization.

Applicable scenarios:

- The activation distribution has a long tail or outliers, so the bulk interval has insufficient resolution under direct MinMax quantization;
- Need to choose a parameter estimation method that automatically clips the interval for static activation quantization.

To check whether the model is on the officially pre-validated list, see the [Foundation Model Support Matrix](../../model/README.md); for a quick introduction to basic CLI usage, see the [One-Click Quantization Usage Guide](../../../user_guide/usage_one_click_quantization.md).

## 2. Inputs and Deliverables

| Type | Name | Source or Storage Location | Format or Constraints | Acceptance Criteria |
| --- | --- | --- | --- | --- |
| Input | Floating-point model weight directory | Model download or a local path | HuggingFace format containing `config.json` and `*.safetensors` shards | Loadable by the target Transformers version |
| Input | Calibration dataset | Built-in `lab_calib/` or a user-defined path | Text prompts in JSONL or JSON format; 50 samples recommended | Readable by the current quantization workflow and covering the main input forms |
| Input | Quantization configuration file | Local YAML file | Complies with the `modelslim_v1` protocol specification | Passes schema validation |
| Deliverable | Histogram parameter configuration plan | User quantization YAML or task configuration | Parameter values are valid and the scope of effect is explicit; the selection rationale of key parameters is explained | Can serve as algorithm configuration input for the subsequent quantization workflow and can reproduce the baseline selection in this guide |

## 3. Process Overview

The overall workflow of Histogram is as follows:

```mermaid
flowchart LR
    A[Collect the activation histogram] --> B[Search for a valid clipping interval]
    B --> C[Determine the quantization range]
    C --> D[Perform activation quantization]
    D --> E[Compare the impact of outliers]
```

Key details of each stage are as follows:

- **Prerequisite: model adaptation.** Model adaptation must be completed before quantization. The adapter implements interfaces such as `PipelineInterface` (the basic pipeline interface); see the [LLM Model Quantization Integration Guide](../../ptq/llm/integration_guide_large_language_model_quantization.md) for details. Model adaptation must be completed first before the quantization workflow can run.
- **Collect the activation histogram**: the observer runs a forward pass with the calibration data (50 samples recommended) and maintains a 2048-bin histogram for the whole activation tensor.
- **Search for a valid clipping interval**: search the histogram for candidate clipping boundaries, allowing a small tail to be discarded in order to reduce the quantization step size of the bulk interval.
- **Determine the quantization range**: generate the final clipping interval and quantization parameters according to whether symmetric or asymmetric is used; the optimal clipping boundary differs between the two.
- **Perform activation quantization**: compute the scale/zero-point from the clipping interval and write the quantization parameters; this applies only to the activation side, and the weights still need another registered algorithm.
- **Compare the impact of outliers**: evaluate the effect of discarding the tail on the end-to-end accuracy and confirm whether clipping brings a gain.

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
        act:
          scope: "per_tensor"  # Currently only per_tensor is supported
          dtype: "int8"        # Currently only int8 is supported
          symmetric: false     # Supports symmetric/asymmetric quantization, with values True/False respectively
          method: "histogram"  # Configured as "histogram" to enable histogram activation quantization
        weight:
          scope: "per_channel"
          dtype: "int8"
          symmetric: true
          method: "minmax"     # Histogram weight quantization is not supported, so this must not be configured as "histogram"
```

**Output**: A reproducible recommended configuration that serves as the comparison baseline for all subsequent parameter adjustments.

### Step 3: Select and Tune Parameters

**Action**:

| Configuration Item | Meaning (Rationale) | Recommended Configuration | Selection and Tuning Advice |
| --- | --- | --- | --- |
| `qconfig.act.scope` | The current Histogram activation quantizer only registers `per_tensor`. The observer maintains a histogram for the whole activation tensor and searches for the clipping boundary; this method cannot be used to switch directly to per-token/per-channel. | Fixed to `per_tensor`. | If a granularity such as per-token is needed, switch to the registered MinMax or another quantizer; the field alone cannot be changed. |
| `qconfig.act.dtype` | The current Histogram activation quantizer only registers INT8. | Fixed to `int8`. | Other dtypes are not part of the current implementation. |
| `qconfig.act.symmetric` | Controls whether the final clipping interval/quantization parameters use the symmetric or asymmetric scheme. The value of the histogram algorithm lies in allowing a small tail to be discarded in order to reduce the quantization step size of the bulk interval. | `true` can be used first when there is no obvious one-sided shift; the current example uses `false`, which suits a distribution with an obvious shift and a backend that supports asymmetric INT8. | Compare according to the activation distribution and the deployment support. Asymmetric can cover a shifted distribution using the zero point but adds an offset; symmetric is simpler. The optimal histogram clipping boundary differs between the two schemes, so recalibration is required after switching. |
| `qconfig.act.method` | Fixed to `histogram`. The current observer internally uses a 2048-bin histogram and reduces the quantization error by searching the clipping boundary; these internal search details are not currently regular user parameters of QConfig. | Fixed to `histogram`. | If the activations have no obvious long tail and MinMax already meets the accuracy requirement, prefer MinMax to reduce calibration complexity; use Histogram only when outliers need to be countered through clipping. |
| `qconfig.weight` | Histogram currently implements only activation quantization, so the weights need a separately selected registered weight quantizer. The example is paired with INT8 per-channel symmetric MinMax, one of the most common W8A8 baselines in the repository. | For weights, start with `int8 + per_channel + symmetric + minmax`; for low-bit weight requirements, select separately according to guides such as SSZ/GPTQ. | Do not write `method: histogram` into the weight section. If the weight algorithm is changed, the calibration of the activation Histogram and the final model error also change, so the whole evaluation should be redone. |

#### Parameter Combinations and Selection Order

The key decision for Histogram is not "how many bins to tune", because the current user configuration does not expose such knobs; the main choices are **whether it is worth replacing pure MinMax with clipping**, and whether symmetric/asymmetric matches the activation distribution and the deployment capability.

**Output**: An algorithm parameter plan that completes single-variable tuning, with a clear selection rationale and tuning direction for each key field.

### Step 4: Converge the Parameter Plan Based on Results

**Action**:

- **Run the recommended configuration first, then tune one variable at a time.** Do not change the bit width, granularity, algorithm parameters, and layer scope simultaneously; otherwise it is difficult to attribute an accuracy change to a specific item.
- **Prefer local fallback over raising precision globally.** If only a few layers are sensitive, use `exclude` or a mixed strategy to preserve high precision for them; this is usually more cost-effective than raising the bit width of the entire model.
- **The model's practical configuration and deployment capability take precedence.** The introductory recommendation is meant to establish a stable starting point; if the target model already has a `lab_practice` recipe, prefer reusing the verified combination.

**Output**: A final parameter plan ready for the subsequent quantization workflow, retaining a record of adjustments relative to the recommended configuration.

## 5. Terminology

| Term | Brief Description | Link |
| --- | --- | --- |
| Histogram-Based Activation Quantization Algorithm | Explains the definition, core principle, key properties, applicable scenarios, and limitations of the algorithm. | [Histogram-Based Activation Quantization Algorithm Terminology Entry](./term_histogram_activation_quantization.md) |

## 6. Related Documents

| API or Document | Brief Description | Link |
| --- | --- | --- |
| linear_quant Configuration Description | Field types, default values, valid values, and complete configuration constraints. | [linear_quant Configuration Description](../../../api_reference/config/processor/linear_quant.md) |
| modelslim_v1 Configuration Description | Consult this when you need to explore task-level advanced configurations such as runner, prior, save, and dataset. | [modelslim_v1 Configuration Description](../../../api_reference/config/quant/modelslim_v1.md) |
| Weight Quantization Usage Guide | User guide: quantization command parameters and complete usage instructions. | [Weight Quantization Usage Guide](../../../user_guide/usage_weight_quantization.md) |
