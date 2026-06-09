---
toc_depth: 3
---
# Quick Quantization Guide

## Contents

- [Overview](#overview)
- [Preparations](#preparations)
- [Quick Start](#quick-start)
  - [Syntax](#syntax)
  - [Parameters](#parameters)
  - [Examples](#examples)
- [Advanced Features](#advanced-features)
  - [Layer-wise and DP Layer-wise Quantization](#layer-wise-and-dp-layer-wise-quantization)
- [Quantization Configuration Protocol](#quantization-configuration-protocol)
  - [Configuration Protocol Overview](#configuration-protocol-overview)
  - [modelslim_v1 Configuration](#modelslim_v1-configuration)
  - [multimodal_sd_modelslim_v1 Configuration](#multimodal_sd_modelslim_v1-configuration)
  - [multimodal_vlm_modelslim_v1 Configuration](#multimodal_vlm_modelslim_v1-configuration)
  - [modelslim_v0 Configuration](#modelslim_v0-configuration)
- [Appendixes](#appendixes)

## Overview

Quick quantization is designed for users of all experience levels. It integrates the quantization capabilities of popular open-source models to provide a true "out-of-the-box" experience. This feature supports global command invocation, allowing you to perform quantization on target model weights simply by specifying the required parameters.

There are two ways to perform quick quantization:

1. **Method 1 (recommended)**: Use this method for mainstream models already supported by the tool where no special quantization requirements exist. After you specify the `quant_type` parameter, the tool automatically applies the optimal configuration from the best practices library. For details, see [Quantization Configuration Protocol](#quantization-configuration-protocol).
2. **Method 2**: Use this method if a model or quantization strategy is not yet in the best practices library, or if you have specific custom requirements. After you specify the `config_path` parameter, the tool applies the custom quantization settings in your configuration file. For details, see [Quantization Configuration Protocol](#quantization-configuration-protocol).

## Preparations

Install msModelSlim. For details, see [msModelSlim Installation Guide](../../getting_started/install_guide.md).

## Quick Start

### Syntax

Quick quantization is initiated through the command line using the following syntax:

```bash
msmodelslim quant [ARGS]
```

After the command is executed, the system matches the optimal configuration from the best practices library based on your input to perform quantization.

**Precautions**

1. The configuration files in the best practices library are stored in `msmodelslim/lab_practice`.

2. The system searches for the best practices YAML file according to the following priority levels (from highest to lowest):

   - Priority 1: Best practices strategy YAML file matching the specified quantization mode and the specified scenario tag (`tag`).
   - Priority 2: Best practices strategy YAML file matching the specified quantization mode but ignoring the specified scenario tag (`tag`). User confirmation is required.
   - Priority 3: Default practices strategy YAML file matching the specified quantization mode and ignoring the scenario tag (`tag`). Accuracy is not guaranteed, and user confirmation is required.
   - Priority 4: Best practices strategy YAML file matching the recommended quantization mode (W8A8) and the specified scenario tag (`tag`). User confirmation is required.
   - Priority 5: Best practices strategy YAML file matching the recommended quantization mode (W8A8) but ignoring the scenario tag (`tag`). User confirmation is required.
   - Priority 6: Default practices strategy YAML file matching the recommended quantization mode (W8A8) and ignoring the scenario tag (`tag`). Accuracy is not guaranteed, and user confirmation is required.

3. To print quantization run logs, set the following environment variables.

   | Environment Variable                 | Purpose       | Mandatory (Yes/No)| Description            |
   |-----------------------|-----------|------|----------------|
   | MSMODELSLIM_LOG_LEVEL | Outputs logs at the specified level and above.| No  | Valid values: `INFO` (default) or `DEBUG`|

### Parameters

| Parameter             | Purpose       | Mandatory (Yes/No)             | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
|-------------------|-----------|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| model_path        | Specifies the model path.     | Yes               | Type: `str`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| save_path         | Specifies the save path for quantized weights. | Yes               | Type: `str`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| device            | Specifies the quantization device.     | No               | 1. Type: `str`.<br>2. Example values: `'npu'`, `'npu:0,1,2,3'`, `'cpu'`<br>3. Default value: `'npu'` (single device).<br>4. If distributed layer-wise quantization is enabled and multiple devices are specified (such as `'npu:0,1,2,3'`), the system initiates data parallelism (DP) layer-wise quantization. Ensure the specified algorithm supports distributed execution. For details, see [Layer-wise and DP Layer-wise Quantization](#layer-wise-and-dp-layer-wise-quantization).                                                                                                                                                                                                             |
| model_type        | Specifies the model name.     | Yes               | 1. Type: `str`.<br>2. The value is case-sensitive. For details, see [Foundation Model Support Matrix](../../model_support/foundation_model_support_matrix.md).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |  |
| config_path       | Specifies the configuration path.   | Mutually exclusive with `quant_type` | 1. Type: `str`.<br>2. Configuration file format: YAML.<br>3. msModelSlim supports only verified configurations from the best practices library. Users are responsible for the results of custom configurations. For details, see [Quantization Configuration Protocol](#quantization-configuration-protocol).<br> 4. After `config_path` is specified, the `tag` parameter becomes invalid.                                                                                                                                                                                                                                                                                                          |
| quant_type        | Specifies the quantization type.     | Mutually exclusive with `config_path`| Valid values: `w4a8`, `w4a8c8`, `w8a8`, `w8a8s`, `w8a8c8`, `w8a16`, and `w16a16s`. For details, see [Foundation Model Support Matrix](../../model_support/foundation_model_support_matrix.md).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| tag               | Specifies the scenario tag for verification. | No               | 1. Type: `str`.<br> 2. The value is case-insensitive. Multiple tags are supported and must be separated by spaces. This allows you to explicitly specify a scenario.<br> 3. Currently, two types of tags are supported, and one scenario can be specified for each type: inference engine (such as `MindIE`, `vLLM-Ascend`, and `SGLang`) and hardware form (such as `Atlas_A2_Inference`, `Atlas_A3_Inference`, `Atlas_A2_Training`, `Atlas_A3_Training`,`Atlas_300I_Duo`, and `CPU`).<br> 4. If no verified configuration for the current scenario is found, the system interacts with you to confirm whether to use the quantization configuration that matches the `quant_type` or `model_type`. |
| debug | Specifies whether to enable the debug mode.| No| 1. Type: Boolean. Default value: `False`.<br>2. After this mode is enabled, the quantization context is automatically saved to the `save_path/debug_info` directory for troubleshooting and algorithm analysis. For details, see the [Debug Mode User Guide](debug_mode.md).                                                                                                                                                                                                                                                                                                                                                                                                                         |
| trust_remote_code | Specifies whether to trust custom code.| No               | 1. Type: Boolean. Default value: `False`.<br>2. Setting this parameter to `True` enables the execution of custom code, which may pose security risks. Ensure the loaded custom code file is secure.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| h, help           | Displays help information for command-line options.| No               | -                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |

### Examples

#### Example 1: Using the quantization type parameter (recommended)

Quantize the Qwen2.5-7B-Instruct model in w8a8 mode by using the quick quantization feature:

```bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --device npu \
  --model_type Qwen2.5-7B-Instruct \
  --quant_type w8a8 \
  --trust_remote_code True
```

where

- `${MODEL_PATH}` specifies the path of the original floating-point weights of Qwen2.5-7B-Instruct.
- `${SAVE_PATH}` specifies the user-defined path for saving the quantized weights.

#### Example 2: Using a configuration file

Use a custom configuration file for quantization:

```bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --device npu \
  --model_type ${MODEL_TYPE} \
  --config_path ${CONFIG_PATH} \
  --trust_remote_code ${TRUST_REMOTE_CODE}
```

#### Example 3: Performing multi-device distributed quantization

Use four NPUs for distributed quantization:

```bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --device npu:0,1,2,3 \
  --model_type ${MODEL_TYPE} \
  --quant_type w8a8 \
  --trust_remote_code True
```

**Note**: Before configuring DP layer-wise quantization, ensure the specified algorithm supports distributed execution. For details, see [Layer-wise and DP Layer-wise Quantization](#layer-wise-and-dp-layer-wise-quantization).

### Output Description

For details about the result files generated by quick quantization and their descriptions, see [Quick Quantization Results](./quantization_result.md).

## Advanced Features

### Layer-wise and DP Layer-wise Quantization

#### Introduction

Layer-wise quantization is an important feature of the `modelslim_v1` quantization service. By processing the model layer by layer, it significantly reduces memory consumption, enabling the quantization of large-scale models.

On this basis, DP layer-wise quantization utilizes DP across multiple devices to significantly improve quantization efficiency while maintaining the low-memory consumption benefit of the layer-wise approach.

#### Application Scenarios

| Type| Scenario Description| Recommended Solution| Description|
|----------|----------|----------|------|
| Large model quantization| Models with a scale of 32 billion parameters (32B) or larger.| Layer-wise quantization| This solution significantly saves memory.|
| Memory-constrained environments| NPU memory is insufficient to load the entire neural network.| Layer-wise quantization| This solution drastically reduces memory overhead.|
| High-efficiency quantization| Ultra-large models or massive calibration datasets| DP layer-wise quantization| This solution significantly accelerates quantization through multi-device parallelism.|

#### Working Principles and Advantages

| Feature| Traditional Quantization (Model-wise)| Layer-wise Quantization (Single-device)| DP Layer-wise Quantization (Multi-device DP)|
|----------|----------|----------|----------|
| **Processing method**| Full neural network processing at the model level| Sequential layer-wise processing on a single device| Parallel layer-wise processing on multiple devices|
| **Memory usage**| 2 to 3 times the model size| **2 to 3 times the size of a single layer**| **2 to 3 times the size of a single layer**|
| **Quantization efficiency**| Fast (for small models)| Slow (for large models)| **Significantly improved through multi-device parallelism**|
| **Applicable models**| Small models (< 32B)| Large models (≥ 32B)| Ultra-large models or massive calibration datasets|

#### Configuration Method

**1. Specify `runner` in the configuration file**

You can specify the `runner` type in the YAML configuration file. Set `runner` to `layer_wise` for layer-wise quantization or `dp_layer_wise` for DP layer-wise quantization. If it is set to `auto` (default), the system automatically selects `layer_wise` or `dp_layer_wise` based on the number of devices.

```yaml
apiversion: "modelslim_v1"       # Specifies the protocol version.
spec:
  runner: "dp_layer_wise"       # Specifies the quantization scheduler: DP layer-wise scheduler.
  process:
    - type: "linear_quant"
      qconfig:
        act:                    # Activation quantization configuration
          scope: "per_tensor"   # Specifies the quantization scope: per_tensor, which indicates static quantization and shares quantization parameters on the entire tensor.
          dtype="int8"        # Specifies the quantization data type. Default value: int8.
          symmetric: false      # Specifies whether to enable symmetric quantization. Default value: false.
          method: "minmax"      # Specifies the quantization method. Default value: minmax.
        weight:                 # Specifies the weight quantization configuration.
          scope: "per_channel"  # Specifies the weight quantization granularity: per_channel quantization.
          dtype="int8"        # Specifies the quantization data type. Default value: int8.
          symmetric: true       # Specifies whether to enable symmetric quantization. Default value: true.
          method: "minmax"      # Specifies the quantization method. Default value: minmax.
      include: ["*"]            # Specifies the layers to be included. Wildcard matching is supported. Default value: ["*"].
```

**2. Configure the devices by using command-line parameters**

```bash
# Single-device layer-wise quantization
msmodelslim quant --device npu:0 ...

# Multi-device DP layer-wise quantization (DP is automatically enabled)
msmodelslim quant --device npu:0,1,2,3 ...
```

#### Precautions

1. **Distributed algorithm support**: When using `dp_layer_wise`, ensure that all processors (such as `linear_quant`) and algorithms (such as `minmax`, `ssz`, and `iter_smooth`) support distributed execution.
2. **Acceleration and calibration**: Acceleration achieved through multi-device parallelism depends on the size of the calibration dataset. If the dataset is too small, the communication overhead may outweigh the parallelization benefits, making the acceleration effect negligible.
3. **Multimodal restrictions**: DP layer-wise quantization currently does not support multimodal models. For multimodal scenarios, use single-device `layer_wise` quantization.

#### Model Adaptation

Layer-wise quantization supports all models listed as compatible with quick quantization in the [Foundation Model Support Matrix](../../model_support/foundation_model_support_matrix.md).
DP layer-wise quantization inherits support from the layer-wise approach and is compatible with all supported Large Language Models (LLMs).

**Note**: DP layer-wise quantization currently does not support multimodal models. For multimodal models, use single-device `layer_wise` quantization.

#### Algorithm Adaptation

Layer-wise quantization (`layer_wise`) supports all algorithms in the `modelslim_v1` architecture.

DP layer-wise quantization (`dp_layer_wise`) currently supports only the following algorithms:

**Outlier Suppression Algorithms**

| Algorithm Name| Type| Supported| Description|
|----------|----------|---------|------|
| Iterative Smooth | iter_smooth | ✅| Distributed execution is fully supported.|
| Flex Smooth Quant | flex_smooth | ✅| Distributed execution is fully supported.|

**Quantization Algorithms**

| Algorithm Name| Quantization Method| Supported| Description|
|----------|---------|---------|------|
| MinMax | minmax | ✅| Distributed execution is fully supported.|
| SSZ | ssz | ✅| Distributed execution is fully supported.|

## Quantization Configuration Protocol

### Configuration Protocol Overview

The quick quantization configuration protocol is built on a hierarchical design that abstracts the entire quantization pipeline into a YAML schema. This includes the quantization service version, pipeline type, processing methods, saving policies, and calibration datasets. This approach allows developers to focus on high-level policies and workflows without hardcoding details in Python.

#### Basic Structure

The basic structure of the YAML configuration file is as follows:

```yaml
apiversion: "modelslim_v1"   # Specifies the protocol version, which is used to select the version of the backend quantization service.
spec:                         # Specific quantization service configuration fields
  runner: "auto"              # Specifies the quantization scheduler type. Default value: auto.
  prior: [ ]                  # (Optional) Specifies the pre-phase list: phases executed before the main process. Each phase has a process and a dataset.
  process: [ ]                # Specifies a list of processors to be executed in sequence.
  save: [ ]                   # Specifies a list of savers that define how to save the quantization result.
  dataset: "mix_calib.jsonl"  # Specifies the calibration dataset file to be matched from the lab_calib directory.
```

#### Protocol Version Description

| Parameter          | Mandatory (Yes/No)| Description                                                                                                                                                                 | Purpose                               |
|--------------|-------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------|
| apiversion   | Yes   | 1. Supported versions: `"modelslim_v0"`, `"modelslim_v1"`, `"multimodal_vlm_modelslim_v1"` and `"multimodal_sd_modelslim_v1"`.<br> 2. The tool selects the corresponding quantization backend based on this field.<br> 3. Configuration fields and parameter requirements vary across different service versions.| Specifies the version of the backend quantization service. Each service uses a distinct configuration protocol.|
| spec         | Yes   | 1. **Pipeline definition**: specifies the type of the quantization pipeline.<br> 2. **Processor configuration**: defines the parameters for various quantization processors.<br>3. **Saving policy**: specifies the format and method for saving quantization results.<br>4. **Dataset configuration**: specifies the calibration dataset.                           | Specifies detailed quantization service parameters, including the quantization strategy, processing workflow, and saving methods.                                |

**Protocol Version Maintenance Policies**

| Protocol Version        | Maintenance Policy| Status |
|--------------|------|-----|
| modelslim_v0 | Scheduled for deprecation| Not recommended|
| modelslim_v1 | Under active development| Recommended |
| multimodal_vlm_modelslim_v1 | Under active development| Recommended |
| multimodal_sd_modelslim_v1 | Under active development| Recommended |

### `modelslim_v1` Configuration

#### Description

`modelslim_v1` is the next-generation quantization framework for ModelSlim and is rapidly evolving.

Compared with `modelslim_v0`, `modelslim_v1` has the following advantages:

- Algorithms are implemented independently, allowing for flexible configuration combinations.
- Layer-wise quantization is supported, which greatly reduces resource consumption.
- It operates without dependency on specific CANN versions.

#### `runner` - Quantization Scheduler Type

**Purpose**: Specifies the type of quantization scheduler to be used.
**Type**: String
**Default value**: `"auto"`

| Value| Purpose| Application Scenario| Feature|
|--------|------|----------|------|
| auto | Automatically selects the optimal strategy| Most scenarios| The tool automatically selects the optimal strategy based on the model size, available memory, and device configuration.|
| layer_wise | Performs layer-wise quantization| Large models (≥ 32B)| This option features low memory usage. Adaptation may be required.|
| dp_layer_wise | Performs DP layer-wise quantization| Large models (≥ 32B) in multi-device scenarios| This option significantly improves quantization efficiency through multi-device parallelism.|
| model_wise | Performs model-wise quantization (non-layer-wise)| Small models (< 32B)| This option features high memory usage and good compatibility.|

#### `prior` - Preprocessing Stage Configuration

**Purpose**: Specifies one or more pre-processing stages to be executed before the main process (`spec.process`). This parameter is used for multi-stage algorithms, such as Adapt Rotation (`adapt_rotation`).

**Type**: List (each item in the list is a stage)
**Default value**: `[]` (no pre-processing stages are specified)

**Stage Fields**

| Field| Type| Description|
|------|------|------|
| process | List| Specifies the list of processors to be executed in this stage. The format is the same as `spec.process`.|
| dataset | String (optional)| Specifies the name of the calibration dataset file used in this stage (matched from the `lab_calib` directory). If this field is not specified, `spec.dataset` is used.|

**Execution sequence**: Stages are executed sequentially according to their order in the `prior` list. The main process (`spec.process`) is executed only after all these stages are complete. Preprocessing stages and the main process share the same model instance. Preprocessing stages can pass results to processors in the main process through the context.

**Typical usage**: For algorithms such as [Adapt Rotation](../../quantization_algorithms/outlier_suppression_algorithms/adapt_rotation.md), place Stage1 within the `process` field of a `prior` stage and configure its `dataset`. Place Stage2 and subsequent quantization steps in `spec.process`. For details, see the YAML configuration example in the Adapt Rotation document.

#### `process` - Processor Configuration Field

**Purpose**: Specifies a list of processors for quantization, which are executed in sequential order.

**Features**:

- **List structure**: The `process` field is a list containing multiple processor configurations, distinguished by their `type` field.
- **Sequential execution**: Processors are executed according to their order in the list.
- **Flexible combination**: Different processor types can be combined to implement complex quantization strategies. However, not all combinations are valid. Adhere to the following guidelines for successful configuration (if you are unsure, refer to the examples in this document or consult a specialist):
  - Perform outlier suppression before quantization. For example, when combining Iterative Smooth and W8A8 quantization, Iterative Smooth must be executed before the quantization step.
  - Avoid defining multiple quantization settings for the same layer. Duplicate definitions for a single layer may cause runtime errors or unexpected results, such as accuracy loss or quantization failure.

##### Supported Processors

| Processor| Type| Configuration Example| Configuration Fields|
| :--- | :--- | :--- | :--- |
| SmoothQuant | Outlier suppression| [SmoothQuant Configuration Example](../../quantization_algorithms/outlier_suppression_algorithms/smooth_quant.md#yaml-configuration-example)| [SmoothQuant Configuration Fields](../../quantization_algorithms/outlier_suppression_algorithms/smooth_quant.md#yaml-configuration-fields)|
| Iterative Smooth | Outlier suppression| [Iterative Smooth Configuration Example](../../quantization_algorithms/outlier_suppression_algorithms/iterative_smooth.md#yaml-configuration-example)| [Iterative Smooth Configuration Fields](../../quantization_algorithms/outlier_suppression_algorithms/iterative_smooth.md#yaml-configuration-fields)|
| Flex Smooth Quant | Outlier suppression| [Flex Smooth Quant Configuration Example](../../quantization_algorithms/outlier_suppression_algorithms/flex_smooth_quant.md#yaml-configuration-example)| [Flex Smooth Quant Configuration Fields](../../quantization_algorithms/outlier_suppression_algorithms/flex_smooth_quant.md#yaml-configuration-fields)|
| Flex AWQ SSZ | Outlier suppression| [Flex AWQ SSZ Configuration Example](../../quantization_algorithms/outlier_suppression_algorithms/flex_awq_ssz.md#yaml-configuration-example)| [Flex AWQ SSZ Configuration Fields](../../quantization_algorithms/outlier_suppression_algorithms/flex_awq_ssz.md#yaml-configuration-fields)|
| KV Smooth | Outlier suppression| [KV Smooth Configuration Example](../../quantization_algorithms/outlier_suppression_algorithms/kv_smooth.md#yaml-configuration-example)| [KV Smooth Configuration Fields](../../quantization_algorithms/outlier_suppression_algorithms/kv_smooth.md#yaml-configuration-fields)|
| QuaRot | Outlier suppression| [QuaRot Configuration Example](../../quantization_algorithms/outlier_suppression_algorithms/quarot.md#yaml-configuration-example)| [QuaRot Configuration Fields](../../quantization_algorithms/outlier_suppression_algorithms/quarot.md#yaml-configuration-fields)|
| linear_quant | Quantization| [Linear Quantization Configuration Example](../../quantization_algorithms/quantization_algorithms/linear_quant.md#yaml-configuration-example)| [Linear Quantization Configuration Fields](../../quantization_algorithms/quantization_algorithms/linear_quant.md#yaml-configuration-fields)|
| group | Quantization| [Group Configuration Example](group.md#yaml-configuration-example)| [Group Configuration Fields](group.md#yaml-configuration-fields)|
| KVCache Quant | Quantization| [KVCache Quant Configuration Example](../../quantization_algorithms/quantization_algorithms/kvcache_quant.md#yaml-configuration-example)| [KVCache Quant Configuration Fields](../../quantization_algorithms/quantization_algorithms/kvcache_quant.md#yaml-configuration-fields)|
| FA3 Quant | Quantization| [FA3 Quant Configuration Example](../../quantization_algorithms/quantization_algorithms/fa3_quant.md#yaml-configuration-example)| [FA3 Quant Configuration Fields](../../quantization_algorithms/quantization_algorithms/fa3_quant.md#yaml-configuration-fields)|
| Float Sparse | Quantization| [Float Sparse Configuration Example](../../quantization_algorithms/quantization_algorithms/float_sparse.md#yaml-configuration-example)| [Float Sparse Configuration Fields](../../quantization_algorithms/quantization_algorithms/float_sparse.md#yaml-configuration-fields)|
| AutoRound | Quantization| [AutoRound Configuration Example](../../quantization_algorithms/quantization_algorithms/autoround.md#yaml-configuration-example)| [AutoRound Configuration Fields](../../quantization_algorithms/quantization_algorithms/autoround.md#yaml-configuration-fields)|
| LAOS (W4A4 scheme)| Comprehensive scheme| [LAOS Configuration Example](../../quantization_algorithms/quantization_algorithms/laos.md#yaml-configuration-example)| [LAOS Configuration Fields](../../quantization_algorithms/quantization_algorithms/laos.md#yaml-configuration-fields)|

#### `save` (Saver Configuration Fields)

**Purpose**: Defines the list of savers for storing quantization results.

##### ascendv1_saver

**Purpose**: Saves the quantization results in the `ascendv1` format.

**Configuration Example**

```yaml
spec:
  save:
    - type: "ascendv1_saver"        # Specifies the ascendv1 saver type.
      part_file_size: 4            # Specifies the shard file size (GB).
```

**Field Description**

| Field| Purpose| Description|
|--------|------|------|
| type | Specifies the saver type.| The value is fixed to `"ascendv1_saver"`, which identifies the object as an Ascend saver.|
| part_file_size | Specifies the shard file size.| The maximum size for each shard file (GB).|

#### `dataset` - Calibration Dataset Configuration

**Purpose**: Specifies the name of the calibration dataset file. The system matches the specified file within the `lab_calib` directory.
**Type**: String
**Default value**: `"mix_calib.jsonl"`

| Attribute| Description|
|------|------|
| File location| Located within the `lab_calib` directory.|
| File format| JSONL format.|
| Purpose| Specifies the calibration dataset for activation quantization.|

#### Examples

In quick quantization, the `qconfig.act.scope` field distinguishes between **static** and **dynamic** quantization:

- **Static quantization (`per_tensor`)**: Quantization parameters are calculated and fixed during the calibration phase. Since no calculation is required during inference, it offers **optimal inference performance** and broad hardware compatibility.
- **Dynamic quantization (`per_token`)**: Quantization parameters are calculated dynamically for each token during inference. This provides **higher accuracy** and effectively handles outliers in activation distributions.
The following example shows the static quantization configuration for a dense model:

```yaml
apiversion: modelslim_v1       # Specifies the protocol version.

spec:                          # Specifications definition
  process:                     # Processor execution list
    - type: "linear_quant"     # Specifies the processor type: linear layer quantization.
      qconfig:
        act:                   # Activation quantization configuration
          scope: "per_tensor"  # Specifies the quantization scope: per_tensor, which indicates static quantization and shares quantization parameters on the entire tensor.
          dtype: "int8"        # Specifies the quantization data type: int8.
          symmetric: False     # Disables symmetric quantization (performs asymmetric quantization, which is recommended for static quantization).
          method: "minmax"     # Specifies the quantization method: minmax.
        weight:                # Weight quantization configuration
          scope: "per_channel" # Specifies the weight quantization granularity: per_channel quantization.
          dtype: "int8"        # Specifies the quantization data type: int8.
          symmetric: True      # Enables symmetric quantization.
          method: "minmax"     # Specifies the quantization method: minmax.
      include: ["*"]            # Specifies the layers to be included. Wildcard matching is supported. Default value: ["*"].
      exclude: ["*down_proj*"] # Specifies the layers to be excluded. Wildcard matching is supported. Default value: [].

  save:                        # Saver configuration list
    - type: "ascendv1_saver"   # Specifies the standard Ascend V1 saver type.
      part_file_size: 4        # Specifies the shard file size: 4GB (recommended for large models).
```

The following example shows how to use different quantization strategies (mixed quantization) for different layers of the MoE model.

```yaml
apiversion: modelslim_v1       # Specifies the protocol version.
                               #
# Define the W8A8 dynamic quantization configuration template.
default_w8a8_dynamic: &default_w8a8_dynamic
  act:                        # Activation quantization configuration
    scope: "per_token"         # Specifies the quantization scope: per_token, which indicates dynamic quantization and uses independent quantization parameters for each token.
    dtype: "int8"              # Specifies the quantization data type: int8.
    symmetric: True            # Enables symmetric quantization.
    method: "minmax"          # Specifies the quantization algorithm: minmax.
  weight:                      # Weight quantization configuration
    scope: "per_channel"       # Specifies the weight quantization granularity: per_channel quantization.
    dtype: "int8"              # Specifies the quantization data type: int8.
    symmetric: True            # Enables symmetric quantization.
    method: "minmax"          # Specifies the quantization algorithm: minmax.
                               #
# Define the W8A8 static quantization configuration template.
default_w8a8: &default_w8a8    # Static quantization template definition
  act:                        # Activation quantization configuration
    scope: "per_tensor"        # Specifies the quantization scope: per_tensor, which indicates static quantization where the entire tensor shares the same quantization parameters.
    dtype: "int8"              # Specifies the quantization data type: int8.
    symmetric: False           # Disables symmetric quantization (performs asymmetric quantization, which is recommended for static quantization).
    method: "minmax"          # Specifies the quantization algorithm: minmax.
  weight:                      # Weight quantization configuration
    scope: "per_channel"       # Specifies the weight quantization granularity: per_channel quantization.
    dtype: "int8"              # Specifies the quantization data type: int8.
    symmetric: True            # Enables symmetric quantization.
    method: "minmax"          # Specifies the quantization algorithm: minmax.
                               #
spec:                          # Specifications definition
  process:                     # Processor execution list
    - type: "group"            # Uses the group processor, which allows different configurations to be applied to different layers.
      configs:                 # List of sub-processor configurations within the group
        - type: "linear_quant" # Specifies linear quantization processor 1.
          qconfig: *default_w8a8 # References the static quantization template and applies static quantization to the Attention layer for performance optimization.
          include: ["*self_attn*"] # Matches layers that contain self_attn.
                               #
        - type: "linear_quant" # Specifies linear quantization processor 2.
          qconfig: *default_w8a8_dynamic # References the dynamic quantization template and applies dynamic quantization to the MLP layer to ensure accuracy.
          include: ["*mlp*"]   # Matches layers that contain mlp.
          exclude: ["*gate"]   # Excludes gating layers from the preceding match.
                               #
  save:                        # Saver configuration list
    - type: "ascendv1_saver"   # Specifies the standard Ascend V1 saver type.
      part_file_size: 4        # Specifies the shard file size: 4GB (recommended for large models).
```

### `multimodal_sd_modelslim_v1` Configuration

#### Description

`multimodal_sd_modelslim_v1` is a quantization service specifically designed for multimodal generative models (such as Wan2.1), built on top of the `modelslim_v1` framework.

**Key Features**

- **Multimodal support**: Provides tailored model optimization for tasks such as text-to-video generation.
- **Dynamic and static quantization**: Supports both dynamic and static activation quantization to adapt to diverse input scenarios.
- **Layer-wise processing**: Supports layer-wise quantization, significantly reducing GPU memory consumption during foundation model quantization.
- **Calibration data caching**: Supports caching and reusing calibration data, improving quantization efficiency.

**Supported Model Types**

- Wan2.1: supports tasks such as text-to-video generation.
- Other multimodal generative models: to be supported gradually in the future.

**Configuration Characteristics**

- Provides the `multimodal_sd_config` field for model-specific configuration parameters.
- Provides the `dump_config` field for capturing and storing calibration data.
- Provides the `model_config` field for parameters related to model loading and inference.

#### `runner` - Quantization Scheduler Type

Due to GPU memory constraints, multimodal generative models currently default to and support only layer-wise quantization (`layer_wise`). By default, `runner` does not need to be configured. If it is set to any value other than `layer_wise`, a warning will be triggered, and the system will automatically switch to `layer_wise` mode.

#### `process` - Processor Configuration Field

This configuration field is identical to the one for `modelslim_v1`. For details, see [`modelslim_v1` Configuration/`process` - Processor Configuration Field](#process---processor-configuration-field).

#### <span id="save---saver-configuration-field-sd">save - Saver Configuration Field</span>

**Purpose**: Defines the list of savers for storing quantization results.

##### `mindie_format_saver`

**Purpose**: Saves results in the MindIE-SD format, which is specifically designed for multimodal generative models.

**Configuration Example**

```yaml
spec:
  save:
    - type: "mindie_format_saver"   # Specifies the MindIE-SD saver type.
      part_file_size: 0            # Specifies the shard file size (GB). 0 disables sharding.
```

**Field Description**

| Field| Purpose| Description|
|--------|------|------|
| type | Specifies the saver type.| The value is fixed to `"mindie_format_saver"`, which identifies the object as a MindIE-SD saver.|
| part_file_size | Specifies the shard file size.| Specifies the shard file size (GB). `0` disables sharding.|

#### `multimodal_sd_config` - Multimodal Generative Specific Configuration Field

**Purpose**: Defines configuration parameters specific to multimodal generative models, including calibration data capture, model loading, and inference configurations.

##### `dump_config` - Calibration Data Capture Configuration

**Purpose**: Configures the capture mode and storage path for calibration data.

**Configuration Example**

```yaml
spec:
  multimodal_sd_config:
    dump_config:
      enable_dump: True            # Specifies whether to enable calibration data loading/dumping. Default value: True.
      capture_mode: "args"         # Specifies the data capture mode. Currently, only "args" is supported.
      dump_data_dir: ""            # Specifies the calibration data dumping directory. An empty string defaults to the weight save path.
```

**Field Description**

| Field| Purpose| Description                                                                                                           | Value|
|--------|------|---------------------------------------------------------------------------------------------------------------|--------|
| enable_dump | Specifies whether to enable dumping.| Controls whether to load and save calibration data. If the value is `False`, calibration data is not loaded or saved. **Note**: When `enable_dump` is set to `False`, the tool issues a log prompt **requiring interactive user confirmation** (such as "Enter `y` to continue, otherwise exit"). This is to verify whether the scenario requires data dumping. Only pure dynamic quantization scenarios operate without calibration data. Scenarios involving static quantization or outlier suppression require it. To dump calibration data, set `enable_dump` to `True`.| `True` (default) or `False`.|
| capture_mode | Specifies the data capture mode.| Specifies how model input data is captured.                                                                                                | Currently, only `"args"` is supported. Other modes will be supported in the future.|
| dump_data_dir | Specifies the calibration data directory.| Specifies the directory for retrieving and saving calibration data. An empty string defaults to the model weight save path. If `calib_data.pth` exists in the specified directory, it is directly loaded as the calibration data. If `calib_data.pth` does not exist, the program automatically saves and loads the calibration data by using the dump mechanism. **This parameter is takes effect only when `enable_dump` is set to `True`.**| A directory string.|

**Capture Mode Description**

- **args**: captures positional arguments, suitable for most multimodal generative models.

##### `model_config` - Model Loading and Inference Configuration

**Purpose**: Specifies the parameters for model loading and inference to customize the default model loading and inference parameters. The configurable fields and type restrictions in `model_config` are subject to the original inference project repository of the multimodal generative model.

**Configuration Example**

```yaml
spec:
  multimodal_sd_config:
    model_config:
      prompt: "A stylish woman walks down a Tokyo street..." # Specifies the calibration prompt.
      offload_model: True          # Specifies whether to offload the model to the CPU after inference.
      frame_num: 121               # Specifies the number of frames generated for the video.
      task: "t2v-14B"              # Specifies the task type.
      size: "1280*720"             # Specifies the dimensions for generation.
      sample_steps: 50             # Specifies the number of sampling steps.
      ......
```

**Field Description**

| Field| Purpose| Description| Value|
|--------|------|------|--------|
| prompt | Specifies the calibration prompt.| Text description used to generate calibration data.| A string|
| offload_model | Specifies whether to enable model offloading.| `True` enables offloading the model to the CPU after inference.| `True`/`False`|
| frame_num | Specifies the number of frames generated for the video.| The number of frames generated for the video.| An integer greater than 0|
| task | Specifies the task type.| Specifies the model task type, where `"t2v-14B"` indicates the text-to-video task of the 14B model and `"t2v-1.3B"` indicates the text-to-video task of the 1.3B model.| `"t2v-14B"`, `"t2v-1.3B"`|
| size | Specifies the dimensions for generation.| The dimensions of the generated video or image.| `"1280*720"`, `"832*480"`|
| sample_steps | Specifies the number of sampling steps.| The number of sampling steps for the diffusion model.| An integer greater than 0|

#### `dataset` - Calibration Dataset Configuration

Configuration of the calibration dataset through the `dataset` field is not supported. The processing method for quantization calibration data in multimodal generative models differs significantly from that in large language models. The system determines whether to directly load the data or automatically obtain and save it by verifying whether calibration data named `calib_data.pth` exists in the specified `dump_data_dir` directory. For details, see [`dump_config` - Calibration Data Capture Configuration](#dump_config---calibration data capture configuration).

#### Example

- W8A8 dynamic quantization for the Wan2.1 model: [wan2_1_w8a8_dynamic.yaml](https://gitcode.com/Ascend/msmodelslim/blob/master/lab_practice/wan2_1/wan2_1_w8a8_dynamic.yaml)

### `multimodal_vlm_modelslim_v1` Configuration

#### Description

`multimodal_vlm_modelslim_v1` is a quantization service specifically designed for multimodal vision language models (VLMs). It is built on the `modelslim_v1` framework.

**Key Features**

- **Multimodal VLM support**: Provides model optimization for image-text multimodal understanding.
- **Layer-wise processing**: Employs layer-wise quantization, significantly reducing GPU memory consumption during foundation model quantization.
- **Multiple dataset formats**: Supports image directories and multimodal calibration datasets with custom text prompts through JSON or JSONL formats.

**Supported Model Types**

- Qwen2.5-Omni series: multimodal end-to-end models (text, image, audio, and video), such as Qwen2.5-Omni-7B.
- Qwen3-VL-MoE series: multimodal models, such as Qwen3-VL-235B-A22B and Qwen3-VL-30B-A3B.
- Other multimodal VLM models: to be supported gradually in the future.

**Configuration Characteristics**

- Supports calibration dataset configuration through the `dataset` field across three methods. Method 1 uses `index.json` or `index.jsonl` (recommended, supports multimodality). Method 2 uses a pure image directory (deprecated). Method 3 uses an image directory combined with a single JSON or JSONL file (deprecated). For details, see [`dataset` - Calibration Data Path Configuration](#dataset---calibration-data-path-configuration).
- Supports default text prompt configuration through the `default_text` field. This field is mandatory for Method 2. For Method 1, it is used if the `text` field is missing from an entry.
- Uses `layer_wise` (layer-wise quantization) mode by default to optimize large-scale multimodal models.

#### `runner` - Quantization Scheduler Type

Due to GPU memory constraints, multimodal VLM models currently default to and support only layer-wise quantization (`layer_wise`). By default, `runner` does not need to be configured. If it is set to any value other than `layer_wise`, a warning will be triggered, and the system will automatically switch to `layer_wise` mode.

#### <span id="process---processor-configuration-field -vlm">`process` - Processor Configuration Field</span>

This configuration field is identical to the one for `modelslim_v1`. For details, see [`modelslim_v1` Configuration/`process` - Processor Configuration Field](#process---processor-configuration-field).

#### `default_text` - Default Text Prompt Configuration

**Function**: Specifies the default text prompt uniformly for all calibration images.
**Type**: String
**Default value**: `"Describe this image in detail."`
**Restrictions**: The text prompt cannot be an empty string. This field is invalid when the `dataset` field is configured as an image directory that contains JSON or JSONL files (used to describe the custom text prompt for each image).

#### <span id="save---saver-configuration-field-vlm">save - Saver Configuration Field</span>

This configuration field is identical to the one for `modelslim_v1`. For details, see [modelslim_v1 Configuration/`save` - Saver Configuration Field](#save---saver-configuration-field).

**Recommended Configuration**

```yaml
spec:
  save:
    - type: "ascendv1_saver"    # Specifies the ascendv1 saver type.
      part_file_size: 4        # Specifies the shard file size (GB). You are advised to save large models in shards.
```

#### `dataset` - Calibration Data Path Configuration

**Function**: Specifies the path to the calibration dataset.

**Type**: String

The following three configuration methods are supported: `dataset` can be configured as a short identifier (can be found under a `dataset_dir` such as `lab_calib`), an absolute path, or a relative path.

---

**Method 1 (recommended): `index.json` or `index.jsonl`**

Point to an `index.json` or `index.jsonl` file, or to a directory **containing exactly one `index.json` or `index.jsonl` file**. This method supports multimodal data (including images, audio, and video) and uses a standardized format. Future features will build upon this method.

- Each entry is a JSON object that must contain **at least the `text` field** (a non-empty string). If omitted, `default_text` in the configuration is used.
- Optional fields (the path must exist if these fields are provided): `image` (`.jpg`/`.jpeg`/`.png`), `audio` (`.wav`/`.mp3`), and `video` (`.mp4`). The path must be relative to the directory where the index file is located.

Directory example:

```text
calib_dir/
├── index.jsonl
├── img1.jpg
├── img2.png
└── a.wav
```

`index.jsonl` example:

```json
{"image": "img1.jpg", "text": "Describe this image."}
{"image": "img2.jpg", "audio": "a.wav", "text": "What is in this picture?"}
```

Configuration example: `dataset: "/path/to/calib_dir"`, `dataset: "/path/to/index.jsonl"`, or a short identifier that resolves to either of the preceding paths.

---

**Method 2: Pure Image Directory**

The directory contains only image files and does not contain any `.json` or `.jsonl` files. All images use `default_text` in the configuration as the unified text prompt.

**Note**: This method is deprecated and will no longer be updated. For new scenarios, use Method 1.

Directory example:

```text
calibImages/
├── img1.jpg
├── img2.png
└── img3.jpeg
```

Configuration example:

```yaml
spec:
  dataset: "calibImages"   # Set it to a short identifier, absolute path, or relative path.
  default_text: "Describe this image in detail."
```

---

**Method 3: Image Directory + Single `.json` or `.jsonl` File (Arbitrary Filename)**

The directory contains images and **a single** `.json` or `.jsonl` file with any filename (except `index.json` or `index.jsonl`), which is used to specify custom text for each image. Only `image` fields are supported. `audio` and `video` fields are not supported.

**Note**: This method is deprecated and will no longer be updated. For new scenarios, use Method 1.

Directory example:

```text
calibImages/
├── img1.jpg
├── img2.png
├── img3.jpeg
└── calib_data.jsonl
```

`calib_data.jsonl` example:

```json
{"image": "img1.jpg", "text": "What objects are in this image?"}
{"image": "img2.png", "text": "Describe the scene."}
```

Configuration example: `dataset: "calibImages"` or the corresponding path.

#### Example

- W8A8 quantization for the Qwen2.5-Omni model: [qwen2_5_omni_thinker_w8a8.yaml](https://gitcode.com/Ascend/msmodelslim/blob/master/lab_practice/qwen2_5_omni_thinker/qwen2_5_omni_thinker_w8a8.yaml)
- W8A8 mixed quantization for the Qwen3-VL-MoE model: [qwen3_vl_moe_w8a8.yaml](https://gitcode.com/Ascend/msmodelslim/blob/master/lab_practice/qwen3_vl_moe/qwen3_vl_moe_w8a8.yaml)

### `modelslim_v0` Configuration

#### Description

The `modelslim_v0` quantization service is primarily composed of legacy interfaces, such as `Calibrator` and `AntiOutlier`. Its configuration protocol (`YAML`) remains fundamentally consistent with the original Python API interfaces, facilitating a smooth migration from older versions.

**Related Documents**

- [Calibrator.md](../../python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_Calibrator.md)
- [AntiOutlier.md](../../python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/AntiOutlier.md)

**Note**: The `modelslim_v0` protocol version is deprecated and will soon be phased out. It is not recommended for use. You are advised to use `modelslim_v1` or later.

## Appendixes

### References

- For ultra-large models, you can use layer-wise quantization to significantly reduce GPU memory consumption. For details, see [Layer-wise and DP Layer-wise Quantization](#layer-wise-and-dp-layer-wise-quantization).
- For details about algorithms supported for quick quantization, see [Quantization Algorithms Supported for Quick Quantization V1](../../quantization_algorithms/README.md).

### FAQ

#### Q1: How Do I Select a Proper Quantization Scheduler?

- `auto`: suitable for most scenarios. The tool automatically selects the optimal strategy based on model size, available memory, and device configuration.
- `layer_wise`: suitable for models with a scale of 32B or larger, or memory-constrained environments.
- `dp_layer_wise`: suitable for ultra-large models or multi-device scenarios with large-scale calibration datasets.
- `model_wise`: suitable for small models (< 32B), offering the best compatibility.

#### Q2: What Are the Differences Between Layer-Wise Quantization and DP Layer-Wise Quantization?

- **Layer-wise quantization**: performs quantization layer by layer on a single device, where the memory usage is 2 to 3 times the size of a single layer.
- **DP layer-wise quantization**: performs quantization layer by layer in parallel across multiple devices, significantly increasing quantization speed while maintaining the low memory usage advantage. For details, see [Layer-wise and DP Layer-wise Quantization](#layer-wise-and-dp-layer-wise-quantization).

#### Q3: How Do I Determine Whether to Use Layer-Wise Quantization?

You are advised to use layer-wise quantization in the following situations:

- Model scale > 32B.
- Insufficient NPU memory.
- Out-of-memory (OOM) errors occur during quantization.

#### Q4: Does Multi-Device Quantization Always Accelerate the Process?

Not necessarily. The acceleration effect of multi-device quantization is affected by several factors:

- Calibration dataset size: If the calibration dataset is small, communication overhead may exceed the parallelization gains.
- Number of devices: A higher number of devices does not guarantee faster speed. Choose a reasonable number based on the actual situation.
- Algorithm support: Only algorithms that support distributed execution can function correctly in multi-device environments. For details, see [Layer-wise and DP Layer-wise Quantization](#layer-wise-and-dp-layer-wise-quantization).

#### Q5: How Do I Tune the Quantization Accuracy?

For details about quantization accuracy tuning, see [Quantization Accuracy Tuning Guide](../../case_studies/quantization_precision_tuning_guide.md).
