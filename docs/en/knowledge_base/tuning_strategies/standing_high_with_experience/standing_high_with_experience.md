# Standing High With Experience Tuning Algorithm

## Overview

Standing High With Experience (an expert-experience-based standing-high algorithm) is an auto tuning strategy that builds on the [Standing High Tuning Algorithm](../standing_high/standing_high.md) and incorporates **expert experience**. This strategy automatically generates a complete standing-high configuration based on expert experience. You only need to specify the **quantization type** (`quant_type`) and the **model structure configuration** (`structure_configs`), without providing a complete quantization configuration, to start standing-high tuning.

Internally, the algorithm delegates the generated complete configuration to the Standing High strategy for execution. Therefore, the standing-high process (floating-point pre-check, binary search for the minimum fallback level, the standing-high process, and traversal of outlier suppression strategies) is the same as that of Standing High; the only difference is that the **configuration source** is automatically filled in from expert experience. If the floating-point pre-check finds that the all-fallback floating-point baseline already fails to meet the accuracy requirement, the algorithm likewise throws `SpecError` (error code 302), exits, and prompts you to relax the accuracy expectation.

## **Principles and Features**

### Principle

Standing High With Experience combines **configuration simplification** with **expert experience-based parameter population**. The process consists of the following three phases:

1. **User side**: Configure only the `quant_type` (such as `w8a8` or `w4a8`) and `structure_configs` (such as the `GQA` and `FFN` structure types and their `include`/`exclude` modes).
2. **Strategy side**: Generate the complete Standing High configuration based on expert experience, `quant_type`, and `structure_configs`, then assemble it into `StandingHighStrategyConfig`.
3. **Execution side**: Pass the generated configuration to `StandingHighStrategy` to perform the standing-high search. The subsequent process is the same as that of the [Standing High Tuning Algorithm](../standing_high/standing_high.md), iteratively optimizing until the optimal configuration that meets the accuracy requirement is obtained.

### Features

1. **Out-of-the-box**: No need to manually provide initial quantization configurations or outlier suppression strategies. This reduces the configuration threshold and helps users quickly get started.
2. **Expert experience-driven**: Quantization configurations and strategies are managed through preset expert experience configuration templates, facilitating maintenance and reuse.
3. **Consistency with Standing High**: The execution phase reuses the binary search, Standing High process, and strategy traversal logic of the base algorithm to maximize the number of quantized layers while ensuring accuracy.
4. **Good scalability**: Users can extend the supported model structures and quantization types by modifying the [expert experience configuration](../../../../../msmodelslim/core/tune_strategy/common/config_builder/expert_experience/expert_experience.yaml). This process requires no changes to the strategy code. **Only those with a deep understanding of quantization** are advised to make such modifications. Otherwise, the tuning behavior may become unstable, or model accuracy may become unpredictable.

## Application Requirements

**Inference engine and fallback support**: Consistent with Standing High, ensure the inference engine (such as vLLM-Ascend) supports quantization fallback. When hybrid operators are used, arbitrary fallback may not be supported. Please confirm based on the actual environment.

**Model adapter**:

- Implement **`ModelSlimPipelineInterfaceV1`** (automatic sensitive layer analysis, delegated to Standing High) and **`StandingHighWithExperienceInterface`** (`load_model` only, to filter unsupported outlier suppression strategies).
- Typical adapter declaration: inherit **`StandingHighWithExperienceInterface`** (`load_model` probe) **and** **`ModelSlimPipelineInterfaceV1`** (sensitive layer analysis and quantization pipeline).
- Sensitive layer analysis does not call `load_model` in advance on the strategy side; for details, see the [Auto Tuning Configuration Guide](../../../api_reference/config/tuning/README.md) and the [LLM Integration Guide](../../ptq/llm/integration_guide_large_language_model_quantization.md).

## Function Description

### Usage Description

It is started through `msmodelslim tune` in the auto tuning process. In the tuning YAML file, set `type` to `standing_high_with_experience` in the `strategy` field, and configure the other fields as described below. For the complete tuning configuration and command parameters, see [Auto Tuning Instructions](../../../user_guide/usage_auto_precision_tuning.md).

**Comparison with Standing High**

| Dimension          | Standing High | Standing High With Experience |
|----------------|---------------|--------------------------------|
| Configuration complexity    | Requires manual creation of the initial quantization configuration and outlier suppression strategies.| Requires only the quantization type and model structure.|
| Execution logic  | Consistent          | Consistent (delegates to the same `StandingHighStrategy`)|
| Application scenarios      | When fine-grained control over each quantization item and strategy is necessary.| When out-of-the-box functionality and automatic strategy selection based on the model structure are preferred.|

### YAML Configuration Example

Configuration under the `strategy` field in the auto tuning configuration file

```yaml
strategy:
  type: standing_high_with_experience
  quant_type: w8a8
  structure_configs:
    - type: "GQA"
      include:
        - "*self_attn*"
    - type: "FFN"
      include:
        - "*mlp*"
```

### YAML Configuration Fields

#### `type` - Strategy Type

**Purpose**: Specifies the tuning algorithm type. When Standing High With Experience is used, set this parameter to `standing_high_with_experience`.

**Type**: String

**Value**: `standing_high_with_experience`

#### `quant_type` - Quantization Type

**Purpose**: Specifies the target quantization type. It must be within the `supported_quant_types` range defined in the expert experience configuration file `expert_experience.yaml`. Currently, `w8a8` and `w4a8` are supported.

**Type**: String

**Example**: `w8a8`

**Note**: Based on this parameter, the corresponding quantization configuration template and outlier suppression strategy list are selected according to expert experience.

#### `structure_configs` - Model Structure Configuration List

**Purpose**: Describes the structure types of the model and the specific modules (`include`/`exclude`) involved in quantization. This information is used to retrieve the corresponding quantization configuration for each structure based on expert experience.

**Type**: `list`. Each element is an object containing the following fields.

| Field  | Purpose        | Type | Required/Optional| Description|
|----------|--------------|-------|-----------|------|
| type     | Specifies the structure type.    | string | Required| The supported substructure types include `GQA`, `FFN`, `MHA`, `MoE`, `MLA`, `DSA`, `SWA`, and `GatedDeltaNet`. The specified type must exist in the [expert experience configuration](../../../../../msmodelslim/core/tune_strategy/common/config_builder/expert_experience/expert_experience.yaml).|
| include  | Specifies the included modules.  | list[string] | Required| Module name matching patterns. The list must not be empty or contain empty strings. Example values: `["*self_attn*"]` and `["*mlp*"]`.|
| exclude  | Specifies the excluded modules.  | list[string] | Optional| Name matching patterns for modules to be excluded from quantization. Example value: `["*kv_b_proj"]`.|

**Notes**: `include` and `exclude` use wildcards (such as `*`) to match **module names in the model**. A module name is the fully qualified name of a linear layer (such as `model.layers.0.self_attn.q_proj`). **include** specifies the modules belonging to the current structure type. Only modules matched by `include` will apply the expert experience configuration associated with that type. `exclude` specifies modules to be removed from the `include` selection. This field is used to handle specific layers within the same structure separately. When configuring these fields, you can identify the naming patterns of each substructure from the actual architecture of the model and ensure the `include` and `exclude` patterns align with those structure types.

**Precautions**: The `include`/`exclude` scope of each configuration must be non-overlapping (orthogonal), which means each layer is covered by only one configuration. Overlapping scopes may lead to repeated quantization and abnormal results.

**Configuration Example**

```yaml
structure_configs:
  - type: "GQA"
    include:
      - "*self_attn*"
    exclude:
      - "*kv_b_proj*"
  - type: "FFN"
    include:
      - "*mlp*"
```
