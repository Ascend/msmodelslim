# HQQ Parameter Configuration Workflow Guide

## 1. Scope

HQQ weight quantization algorithm. As a weight quantization method of the `linear_quant` processor, HQQ iteratively optimizes the offset through half-quadratic splitting to achieve calibration-free weight quantization, making it suitable for scenarios without calibration data or where quantization time is a concern.

This guide is intended for users who configure HQQ as the weight quantization method. Its focus is not to expand on the complete execution command, but to explain why the recommended configuration is a suitable starting point, which parameters really need tuning, and which item should be changed first when accuracy or resource problems arise. Such algorithms usually directly affect the quantization scale, the rounding method, the quantization granularity, or the low-bit representation, so the parameter choices simultaneously affect accuracy, compression ratio, and deployment compatibility. On first use, it is recommended to fix the target bit width and the evaluation method first and adopt only the recommended starting point given in this guide; once the baseline is confirmed stable, tune the parameters that really affect the algorithm's behavior one by one.

If the target model already has a complete and validated quantization recipe, prefer reusing that recipe; if the current algorithm is incompatible with the target model structure, numeric format, or deployment backend, do not bypass the support constraints by increasing the search effort or expanding the scope.

## 2. Inputs and Deliverables

| Type | Name | Source or Storage Location | Format or Constraints | Acceptance Criteria |
| --- | --- | --- | --- | --- |
| Input | Target model and quantization objective | The model to quantize and the deployment/evaluation plan | Clarifies the target numerical format or bit width, the target modules, the accuracy/performance goals, and the deployment constraints | Explains why this algorithm is chosen and where it fits in the overall quantization plan |
| Input | Configuration constraints and practical baseline | This algorithm's configuration description, the target model's existing `lab_practice` recipe (if any) | Field names, supported combinations, scope, and model adaptation are consistent with the current version | The recommended starting point is traceable to the current configuration definition or a verified practice |
| Deliverable | HQQ parameter configuration plan | User quantization YAML or task configuration | Parameter values are valid and the scope of effect is explicit; the selection rationale of key parameters is explained | Can serve as algorithm configuration input for the subsequent quantization workflow and can reproduce the baseline selection in this guide |

## 3. Process Overview

A relatively robust set of parameters has already been selected and fixed in the algorithm implementation, and the weight bit width currently supported by the HQQ algorithm is fixed to INT8 with the granularity fixed to per-channel, so this stable configuration should be used to establish the baseline at the introductory stage. The procedure below does not expand on the quantization command itself.

```mermaid
flowchart LR
    A[Determine the weight bit width and granularity] --> B[Perform quantization]
    B[Perform quantization] --> C[Measure accuracy and performance]
    C[Measure accuracy and performance] --> D[Switch the algorithm if necessary]
```

## 4. Procedure

### Step 1: Confirm Objectives and Constraints

**Action**: first fix the target model, the final quantization format/bit width, the target of application, and the evaluation baseline, then confirm the parameter combinations supported by the current version. If the target model already has a `lab_practice` recipe, prefer using that recipe as the practical baseline; if there is no model-specific recipe, use the general recommended starting point given in this guide. HQQ is a data-free method and can complete weight quantization without fixed calibration data, but you should still confirm that the deployment side supports the W8A16 format.

**Output**: A clear configuration objective: target bit width/format, scope of application, evaluation baseline, and deployment constraints.

### Step 2: Establish the Recommended Baseline

**Action**:

The following configuration establishes the **first comparable baseline**. If the target model already has a `lab_practice` recipe, prefer the verified recipe, then use this section to understand why each parameter is chosen this way.

```yaml
spec:
  process:
    - type: "linear_quant"
      qconfig:
        act:
          scope: "per_tensor"
          dtype: "float"
          symmetric: true
          method: "none"
        weight:
          scope: "per_channel"
          dtype: "int8"
          symmetric: false
          method: "hqq"
      include: ["*"]
```

The recommended values above are used to establish the first reproducible baseline. The configurations most worth attention include `qconfig.weight.scope`, `qconfig.weight.dtype`, `qconfig.weight.symmetric`, and `qconfig.weight.method`. The recommended values given here for `qconfig.weight` are the only selectable values.

**Output**: A reproducible recommended baseline configuration.

### Step 3: Select and Tune Parameters

**Action**:

Parameter selection is best understood in three layers: the four **algorithm support constraints** `dtype/scope/symmetric/method` are fixed values in the HQQ algorithm; next, determine the **scope of application** such as `include/exclude` and subgraph types; and finally adjust the numeric parameters that change the error and the overhead. The recommended values below distinguish the code defaults, the practice values validated in the repository's `lab_practice`, and the recommended starting point for first use. If the target model already has a practical configuration, prefer following it and then make single-variable adjustments according to the explanations in this section.

| Configuration Item | Meaning (Rationale) | Recommended Configuration | Selection and Tuning Advice |
| --- | --- | --- | --- |
| `qconfig.weight.scope` | The HQQ weight quantizer only registers `per_channel`, and each output channel optimizes its offset independently. | Fixed to `per_channel`. | When a granularity such as per_group / per_tensor is needed, choose another weight quantization algorithm. |
| `qconfig.weight.dtype` | HQQ currently only registers INT8 per-channel asymmetric. HQQ improves MinMax accuracy through an iterative offset and is suitable for W8 weights. | Use `int8` for a W8 target. | When a low bit width such as int4 is needed, choose algorithms such as SSZ / GPTQ; switching the bit width requires requantization, so do not reuse the old scale. |
| `qconfig.weight.symmetric` | HQQ only supports asymmetric quantization; a symmetric configuration reports an error directly during the validation stage. | Fixed to `false`. | Do not change it to `true`; the current registration path does not provide a symmetric combination. |
| `qconfig.weight.method` | Fixed to `hqq`. Starting from the initial MinMax parameters, the algorithm fixes the scale and iteratively optimizes the offset. | Fixed. | Switching the method means switching the weight quantization algorithm. |

### Parameter Combinations and Selection Order

1. **Fix the deployment target and the supported combination first**: determine the final bit width/numeric format first, then confirm that the current quantizer actually registers the corresponding `dtype + scope + symmetric + method` combination; do not cross the support constraints in pursuit of a particular parameter value.
2. **Then fix the scope of application**: when using an existing model recipe, reuse its `include/exclude` or subgraph types first; when there is no recipe, first establish a baseline with a clearly defined coverage scope and confirm that the algorithm really applies to the expected modules.
3. **Tune layer by layer**: change only one item at a time in the scope of application or the data selection, and keep the same evaluation set. If there is no stable gain after the adjustment, return to the previous baseline instead of continuing to stack multiple changes.

The core degree of freedom of HQQ lies in the scope of application; the iteration parameters in the algorithm and the configuration-side `scope/dtype/symmetric/method` are basically fixed, so the key is to confirm the deployment combination and evaluate whether the W8A16 accuracy meets the target.

**Output**: An algorithm parameter plan that completes single-variable tuning, with a clear selection rationale and tuning direction for each key field.

### Step 4: Converge the Parameter Plan Based on Results

**Action**:

When tuning, it is recommended to first record a complete baseline, including the quantization scope used, the key parameters, and the end-to-end metrics. Change only one variable per round and compare the result directly with the baseline; if an adjustment brings a gain, continue to search nearby values in small steps. For cases where only a few layers or modules are anomalous, prefer local exclusion, local fallback, or mixed precision, rather than directly raising the precision configuration of the whole model, which usually preserves compression and performance gains more easily.

- **Run the recommended baseline first, then tune one variable at a time.** Do not change the bit width, granularity, algorithm parameters, and layer scope simultaneously; otherwise it is difficult to attribute an accuracy change to a specific item.
- **HQQ is a calibration-free starting point for accuracy improvement.** If MinMax accuracy is insufficient for a W8A16 target and calibration data is hard to prepare, try HQQ first; if the accuracy still falls short, consider SSZ/GPTQ with calibration data or an outlier suppression algorithm.
- **Prefer local fallback over raising precision globally.** If only a few layers are sensitive, use `exclude` or a mixed strategy to preserve high precision for them; this is usually more cost-effective than raising the bit width of the entire model.
- **The model's practical configuration and deployment capability take precedence.** The introductory recommendation is meant to establish a stable starting point; if the target model already has a `lab_practice` recipe, prefer reusing the verified combination.

**Output**: A final parameter plan ready for the subsequent quantization workflow, retaining a record of the adjustments relative to the recommended baseline.

## 5. Terminology

| Term | Brief Description | Link |
| --- | --- | --- |
| HQQ Weight Quantization Algorithm | Explains the definition, core principle, key properties, applicable scenarios, and limitations of the algorithm. | [HQQ Weight Quantization Algorithm Terminology Entry](./term_hqq.md) |

## 6. Interface Document List

| Interface or Capability | Brief Description | Link |
| --- | --- | --- |
| linear_quant Configuration Description | Field types, default values, valid values, and complete configuration constraints. | [linear_quant Configuration Description](../../../api_reference/config/processor/linear_quant.md) |
| modelslim_v1 Configuration Description | Consult this when you need to explore task-level advanced configurations such as runner, prior, save, and dataset. | [modelslim_v1 Configuration Description](../../../api_reference/config/quant/modelslim_v1.md) |
