# Flex AWQ SSZ Parameter Configuration Workflow Guide

## 1. Scope

Flex AWQ SSZ is a flexible activation-aware weight quantization smoothing algorithm. As an outlier suppression algorithm, Flex AWQ SSZ is usually a preprocessing step before quantization: it evaluates parameters with a real quantizer and automatically searches for the optimal `alpha` to suppress activation outliers, improving the accuracy of low-bit quantization.

## 2. Inputs and Deliverables

| Type | Name | Source or Storage Location | Format or Constraints | Acceptance Criteria |
| --- | --- | --- | --- | --- |
| Input | Target model and quantization objective | The model to quantize and the deployment/evaluation plan | Clarifies the target numerical format or bit width, the target modules, the accuracy/performance goals, and the deployment constraints | Explains why this algorithm is chosen and where it fits in the overall quantization plan |
| Input | Model adapter | User adaptation code, invoked via `--model_type` | Implements the `PipelineInterface` and `FlexSmoothQuantInterface` interfaces | Properly driven by the runner and processor |
| Input | Configuration constraints and practical baseline | This algorithm's configuration description, the target model's existing `lab_practice` recipe (if any) | Field names, supported combinations, scope, and model adaptation are consistent with the current version | The recommended starting point is traceable to the current configuration definition or a verified practice |
| Input | Calibration data (when required by the algorithm) | Task `dataset`, a calibration set, or a model practice recipe | The data should represent the real input distribution; for multi-stage algorithms, keep the data distribution consistent across stages | Readable by the current quantization workflow and covering the main input forms |
| Deliverable | Flex AWQ SSZ parameter configuration plan | User quantization YAML or task configuration | Parameter values are valid and the scope of effect is explicit; the selection rationale of key parameters is explained | Can serve as algorithm configuration input for the subsequent quantization workflow and can reproduce the baseline selection in this guide |

## 3. Process Overview

```mermaid
flowchart LR
    A[Adapt the model<br/>pipeline/algorithm interfaces] --> B[Determine the target format<br/>such as W4A8]
    B --> C[Evaluate candidate scales<br/>with a real quantizer]
    C --> D[Search for alpha]
    D --> E[Fuse the scaling]
    E --> F[Proceed to subsequent quantization]
```

## 4. Procedure

### Step 1: Adapt the Model Pipeline and Algorithm Interfaces

**Goal**: The quantization tool cannot directly operate on models with arbitrary structures; an adapter is needed to convert model operations into standard methods. Flex AWQ SSZ and Flex Smooth Quant share the smoothing processor framework, so the model adapter must inherit and implement the pipeline interface and the flexible smoothing interface to provide the model's smoothing subgraph topology. Model adaptation must be completed before the Flex AWQ SSZ algorithm can run.

**Action**: The model adaptation code must implement the following interfaces, all of which are aggregated and provided by `msmodelslim.model.interface_hub`:

1. **`PipelineInterface`** (`ModelSlimPipelineInterfaceV1`): the basic pipeline interface, responsible for model loading (`init_model`), data preprocessing (`handle_dataset`), layer-wise traversal (`generate_model_visit`), layer-wise forward pass (`generate_model_forward`), and so on.
2. **`FlexSmoothQuantInterface`**: the dedicated interface for flexible smooth quantization, aggregated and exported by `msmodelslim.model.interface_hub`. It provides the model's smooth subgraph topology configuration describing the connections between Norm and the subsequent Linear, OV, up/down structures, so that the smooth processor can locate the modules to smooth.

For a standard open-source LLM implemented with HuggingFace Transformers, it is recommended to build the adapter through compositional inheritance:

```python
from msmodelslim.model.interface_hub import (
    IModel,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1 as PipelineInterface,
    # ---- Flex AWQ SSZ algorithm adaptation interfaces (the focus of this step) ----
    FlexSmoothQuantInterface,  # Algorithm adaptation: get_adapter_config_for_subgraph returns the smoothing subgraph topology
)

class MyModelAdapter(TransformersModel, ModelInfoInterface, PipelineInterface, FlexSmoothQuantInterface):
    pass
```

Descriptions of each interface method:

- The standard methods of **`PipelineInterface`**: loading data (`handle_dataset`), loading the model (`init_model`), layer-wise traversal (`generate_model_visit`), layer-wise forward pass (`generate_model_forward`), and controlling the KV cache (`enable_kv_cache`). The quantization runner and processor drive the model only through these standard methods and are independent of the model's internal structure.
- The core methods of **`FlexSmoothQuantInterface`**: return the topology configuration of the subgraphs to be smoothed (subgraph types such as `norm-linear`, `linear-linear`, `ov`, and `up-down`); the processor uses it to locate the modules to be smoothed and evaluates candidate smoothing scales with a real quantizer.

The adapter is then referenced in the quantization command via `--model_type MyModelAdapter`.

To develop new model adaptation code, see the [LLM Model Quantization Integration Guide](../../ptq/llm/integration_guide_large_language_model_quantization.md). Note that model adaptation must be completed first before the subsequent quantization workflow can run.

**Output**: A usable model adapter registered in the framework (invoked via `--model_type`).

### Step 2: Establish the Recommended Configuration

**Action**:

The following configuration establishes the **first comparable baseline**. If the target model already has a `lab_practice` recipe, prefer the verified recipe, then use this section to understand why each parameter is chosen this way.

```yaml
spec:
  process:
    - type: "flex_awq_ssz"
      # When alpha is omitted it is searched automatically, which is usually more reliable than a manually fixed value.
      qconfig:
        act:
          scope: "per_token"
          dtype: "int8"
          symmetric: true
          method: "minmax"
        weight:
          scope: "per_channel"
          dtype: "int4"
          symmetric: true
          method: "ssz"
          ext:
            step: 10
      enable_subgraph_type:
        - "norm-linear"
        - "linear-linear"
        - "ov"
        - "up-down"
      include: ["*"]
      exclude: []
```

**Output**: A reproducible recommended configuration that serves as the comparison baseline for all subsequent parameter adjustments.

### Step 3: Select and Tune Parameters

**Action**:

| Configuration Item | Meaning (Rationale) | Recommended Configuration | Selection and Tuning Advice |
| --- | --- | --- | --- |
| `type` | Processor identifier, fixed to `flex_awq_ssz`. | Fixed. | Do not tune. |
| `alpha` | Controls the exponential weight of the activation statistics in the smoothing scale. If `alpha` is not provided, the current implementation searches for alpha by simulating the quantization error with the real `qconfig`; omitting it therefore hands the choice to a data-driven search. | It is recommended to omit `alpha` and let the algorithm search; set it manually only when a fixed value has been validated on the same model and the same qconfig. | A manual value is meaningful only for the current calibration distribution, subgraph, and qconfig. After the final weight bit width, the SSZ settings, or the subgraph scope change, the search should be redone rather than reusing the old alpha. |
| `beta` | The configuration layer provides a coefficient in the range 0~1, but the current FlexAWQSSZ search/application path fixes beta to `0` and mainly optimizes alpha; its behavior differs from the two-parameter search of FlexSmoothQuant. | Usually omit it or keep the default; do not treat beta as a main tuning knob in the current version. | Unless a later implementation or model recipe explicitly uses beta, do not tune beta manually based on FlexSmoothQuant experience; the main basis for selection in the current algorithm is alpha plus a real qconfig. |
| `qconfig` | The search directly constructs the final linear quantizer to evaluate candidate smoothing scales, so qconfig is part of the AWQ search objective. The DeepSeek/GLM W4A8 practices uniformly use activation INT8 per-token MinMax plus weight INT4 per-channel SSZ, with `step: 10` configured in SSZ. | For a W4A8 target, prefer the combination verified in the repository; for other targets, it must be changed to the real final qconfig. | Any change to a key qconfig item requires the alpha search to be redone. In particular, switching weights from MinMax to SSZ, or from INT8 to INT4, changes the real quantization error of the candidate scales. |
| `qconfig.weight.ext.step` | The maximum number of iterations passed to SSZ. SSZ internally defaults to at most 50 iterations and supports early convergence; the FlexAWQSSZ practices in the repository use `10` to balance the search cost. | Use `10` when reusing the W4A8 practice; when accuracy is the priority for a standalone model and time permits, it can be omitted so that SSZ uses the default of at most 50. | A `step` that is too small may leave SSZ insufficiently optimized under each alpha candidate, which in turn affects the alpha ranking; but a value that is too large also significantly amplifies the cost of "outer alpha search x inner SSZ". Fix the qconfig first, then choose between time and convergence. |
| `enable_subgraph_type` | Determines the type of fusable subgraph on which to operate. Although four types are supported by default, all 5 FlexAWQSSZ practices in the current repository enable only `["up-down"]`, which shows that model recipes often deliberately narrow the scope. | When a practical recipe exists, follow its subgraph types exactly; when there is no recipe, do not mechanically copy the "four default types" - choose first according to the model structure and the target layers. | Expanding the subgraph types increases the search and transformation scope and may also introduce incompatible structures. If the algorithm is intended only for low-bit optimization of the MLP up/down, establishing a baseline from `up-down` first is easier to interpret. |
| `include` | Allow list for the scope; a module-name pattern determines which matched modules enter the current algorithm. It only decides *where* the algorithm is applied and does not change the algorithm's internal formula. | Start from `["*"]` when there is no model-specific recipe; when a `lab_practice` recipe exists, inherit its module scope directly. | First ensure the scope covers the intended modules, then check accuracy. If the scope is too broad, a few incompatible structures or sensitive layers amplify the overall risk; if it is too narrow, the algorithm's benefit may not be visible. When narrowing the scope, adjust by structural family or known sensitive layers rather than deleting a large number of layers by index alone. |
| `exclude` | Deny list for the scope; matched modules are removed from the `include` candidates. It takes precedence over `include`. It is suitable for protecting sensitive layers, the first and last layers, or model-specific incompatible structures. | Keep the list empty by default; add entries only when a practical recipe, a compatibility constraint, or a sensitivity result provides explicit evidence. | For local accuracy problems, prefer using `exclude` for a small-scope fallback; this preserves gains more easily than raising the bit width of the whole model or disabling the entire algorithm. After adding each exclusion, confirm that the wildcard does not accidentally affect adjacent modules. |

#### Parameter Combinations and Selection Order

The parameters of FlexAWQSSZ are strongly coupled: **fix the final qconfig first -> then fix the subgraphs -> and finally search for alpha**. The SSZ `step` affects both the evaluation quality of each candidate and the total search time, so do not change it substantially at the same time as the manual alpha value. Change only one variable per round and keep the same calibration set and evaluation set; if there is no stable gain after the adjustment, return to the previous baseline.

**Output**: An algorithm parameter plan that completes single-variable tuning, with a clear selection rationale and tuning direction for each key field.

### Step 4: Write the Quantization Configuration and Run the Command

**Goal**: Integrate the above steps into a complete YAML quantization configuration file and start the quantization workflow via the CLI.

#### Complete Example: Flex AWQ SSZ + W4A8 Dynamic Quantization (Recommended Starting Point)

##### Configuration File: `flex_awq_ssz_w4a8.yaml`

```yaml
apiversion: modelslim_v1
spec:
  runner: auto
  process:
    - type: flex_awq_ssz            # Evaluate candidate smoothing scales with a real quantizer and search for alpha
      qconfig:                      # Consistent with the final quantization target: W4A8
        act:
          dtype: int8
          scope: per_token
          symmetric: true
          method: minmax
        weight:
          dtype: int4
          scope: per_channel
          symmetric: true
          method: ssz
          ext:
            step: 10                # Maximum number of SSZ iterations, balancing the search cost
      enable_subgraph_type:
        - "up-down"
      include: ["*"]
    - type: linear_quant            # connect to W4A8 dynamic quantization
      qconfig:
        act:
          dtype: int8
          scope: per_token
          symmetric: true
          method: minmax
        weight:
          dtype: int4
          scope: per_channel
          symmetric: true
          method: ssz
      include: ["*"]
  save:
    - type: ascendv1_saver          # Ascend inference standard save format
  dataset: mix_calib.jsonl          # built-in mixed calibration set
```

##### Run the command (single-card quantization)

```bash
msmodelslim quant \
  --model_path <float_model_dir> \
  --save_path <quantized_weight_output_dir> \
  --model_type <model_adapter_name> \
  --config ./flex_awq_ssz_w4a8.yaml \
  --device npu \
  --device_id 0
```

**Output**: Complete quantized weight files and description files are generated in the directory specified by `--save_path`.

## 5. Terminology

| Term | Brief Description | Link |
| --- | --- | --- |
| Flex AWQ SSZ Flexible Activation-Aware Weight Quantization Smoothing Algorithm | Explains the definition, core principle, key properties, applicable scenarios, and limitations of the algorithm. | [Flex AWQ SSZ Flexible Activation-Aware Weight Quantization Smoothing Algorithm Terminology Entry](./term_flex_awq_ssz.md) |

## 6. Related Documents

| API or Document | Brief Description | Link |
| --- | --- | --- |
| `PipelineInterface` / `FlexSmoothQuantInterface` | Model pipeline and Flex AWQ SSZ algorithm adaptation interfaces, both exported together from `msmodelslim.model.interface_hub`. | [Interface hub module](../../../../../msmodelslim/model/interface_hub.py) |
| flex_awq_ssz Configuration Description | Field types, default values, valid values, and complete configuration constraints. | [flex_awq_ssz Configuration Description](../../../api_reference/config/processor/flex_awq_ssz.md) |
| modelslim_v1 Configuration Description | Consult this when you need to explore task-level advanced configurations such as runner, prior, save, and dataset. | [modelslim_v1 Configuration Description](../../../api_reference/config/quant/modelslim_v1.md) |
| Weight Quantization Usage Guide | User guide: quantization command parameters and complete usage instructions. | [Weight Quantization Usage Guide](../../../user_guide/usage_weight_quantization.md) |
