# Quick Quantization Guide

## 1. Overview

Quick quantization is designed for users of all experience levels. It integrates the quantization capabilities of popular open-source models to provide a true "out-of-the-box" experience. This feature supports global command invocation, allowing you to perform quantization on target model weights simply by specifying the required parameters.

There are two ways to perform quick quantization:

1. **Method 1 (recommended)**: Use this method for mainstream models already supported by the tool where no special quantization requirements exist. After you specify the `quant_type` parameter, the tool automatically applies the optimal configuration from the best practices library. For details, see [Quantization Configuration Protocol](#5-quantization-configuration-protocol).
2. **Method 2**: Use this method if a model or quantization strategy is not yet in the best practices library, or if you have specific custom requirements. After you specify the `config_path` parameter, the tool applies the custom quantization settings in your configuration file. For details, see [Quantization Configuration Protocol](#5-quantization-configuration-protocol).

## 2. Preparations

Install msModelSlim. For details, see [msModelSlim Installation Guide](../../../install_guide/install_guide.md).

## 3. Quick Start

### 3.1 Syntax

Quick quantization is initiated through the command line using the following syntax:

```bash
msmodelslim quant [ARGS]
```

When `--quant_type` is specified, the system automatically selects the optimal configuration from the best-practice library for quantization. When `--config_path` is provided, the user-specified configuration is used directly, bypassing the best-practice library.

**Precautions**

1. The configuration files in the best practices library are stored in [msmodelslim/lab_practice](../../../../../lab_practice/).

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

### 3.2 Parameters

| Parameter             | Purpose       | Mandatory (Yes/No)             | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
|-------------------|-----------|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| model_path        | Specifies the model path.     | Yes               | Type: `str`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| save_path         | Specifies the save path for quantized weights. | Yes               | Type: `str`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| device            | Specifies the quantization device.     | No               | 1. Type: `str`.<br>2. Example values: `'npu'`, `'npu:0,1,2,3'`, `'cpu'`<br>3. Default value: `'npu'` (single device).<br>4. If distributed layer-wise quantization is enabled and multiple devices are specified (such as `'npu:0,1,2,3'`), the system initiates data parallelism (DP) layer-wise quantization. Ensure the specified algorithm supports distributed execution. For details, see [Layer-wise and DP Layer-wise Quantization](#41-layer-wise-and-dp-layer-wise-quantization).                                                                                                                                                                                                             |
| model_type        | Specifies the model name.     | Yes               | 1. Type: `str`.<br>2. The value is case-sensitive. For details, see [Foundation Model Support Matrix](../../model_support/foundation_model_support_matrix.md).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| config_path       | Specifies the configuration path.   | Mutually exclusive with `quant_type` | 1. Type: `str`.<br>2. Configuration file format: YAML.<br>3. msModelSlim supports only verified configurations from the best practices library. Users are responsible for the results of custom configurations. For details, see [Quantization Configuration Protocol](#5-quantization-configuration-protocol).<br> 4. After `config_path` is specified, the `tag` parameter becomes invalid.                                                                                                                                                                                                                                                                                                          |
| quant_type        | Specifies the quantization type.     | Mutually exclusive with `config_path`| Valid values: `w4a4`, `w4a8`, `w4a4c8`, `w4a4f8`, `w4a8c8`, `w8a16`, `w8a8`, `w8a8s`, `w8a8c8`, `w8a8f8`, `w4a4f4`, and `w16a16s`. For details, see [Foundation Model Support Matrix](../../model_support/foundation_model_support_matrix.md). If no matching best-practice configuration is found, the system will prompt the user to confirm whether to apply the recommended configuration. For details, see [Command Format Search Priority](#31-syntax).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| tag               | Specifies the scenario tag for verification. | No               | 1. Type: `str`.<br> 2. The value is case-insensitive. Multiple tags are supported and must be separated by spaces. This allows you to explicitly specify a scenario.<br> 3. Currently, two types of tags are supported, and one scenario can be specified for each type: inference engine (such as `MindIE`, `vLLM-Ascend`, and `SGLang`) and hardware form (such as `Atlas_A2_Inference`, `Atlas_A3_Inference`, `Atlas_A2_Training`, `Atlas_A3_Training`, `Atlas_300I_Duo`, `Ascend_950`, and `CPU`).<br> 4. If no verified configuration for the current scenario is found, the system interacts with you to confirm whether to use the quantization configuration that matches the `quant_type` or `model_type`. |
| debug | Specifies whether to enable the debug mode.| No| 1. Type: Boolean. Default value: `False`.<br>2. After this mode is enabled, the quantization context is automatically saved to the `save_path/debug_info` directory for troubleshooting and algorithm analysis. For details, see the [Debug Mode User Guide](debug_mode.md).                                                                                                                                                                                                                                                                                                                                                                                                                         |
| trust_remote_code | Specifies whether to trust custom code.| No               | 1. Type: Boolean. Default value: `False`.<br>2. Setting this parameter to `True` enables the execution of custom code, which may pose security risks. Ensure the loaded custom code file is secure.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| h, help           | Displays help information for command-line options.| No               | -                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |

### 3.3 Examples

#### 3.3.1 Example 1: Using the quantization type parameter (recommended)

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
- `--device npu`: specifies a single NPU for quantization.
- `--model_type Qwen2.5-7B-Instruct`: specifies the model type; must match the name in the support matrix.
- `--quant_type w8a8`: specifies the quantization type to W8A8.
- `--trust_remote_code True`: trusts remote code (required for some models).

#### 3.3.2 Example 2: Using a configuration file

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

Where:

- `${MODEL_PATH}` — Path to the original floating-point weights
- `${SAVE_PATH}` — User-defined path for saving quantized weights
- `--device npu` — The single NPU for quantization
- `${MODEL_TYPE}` — Model type; must match the name in the support matrix
- `${CONFIG_PATH}` — Path to a custom YAML configuration file
- `${TRUST_REMOTE_CODE}` — Whether to trust remote code

#### 3.3.3 Example 3: Performing multi-device distributed quantization

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

Where:

- `${MODEL_PATH}` — Path to the original floating-point weights
- `${SAVE_PATH}` — User-defined path for saving quantized weights
- `--device npu:0,1,2,3` — The four NPUs for distributed quantization
- `${MODEL_TYPE}` — Model type; must match the name in the support matrix
- `--quant_type w8a8` — W8A8 quantization type
- `--trust_remote_code True` — Trusts remote code

>[!NOTE]
>
>Before configuring DP layer-wise quantization, ensure the specified algorithm supports distributed execution. For details, see [Layer-wise and DP Layer-wise Quantization](#41-layer-wise-and-dp-layer-wise-quantization).

### 3.4 Output Description

For details about the result files generated by quick quantization and their descriptions, see [Quick Quantization Results](./quantization_result.md).

## 4. Advanced Features

### 4.1 Layer-wise and DP Layer-wise Quantization

#### 4.1.1 Introduction

Layer-wise quantization is an important feature of the `modelslim_v1` quantization service. By processing the model layer by layer, it significantly reduces memory consumption, enabling the quantization of large-scale models.

On this basis, DP layer-wise quantization utilizes DP across multiple devices to significantly improve quantization efficiency while maintaining the low-memory consumption benefit of the layer-wise approach.

#### 4.1.2 Application Scenarios

| Type| Scenario Description| Recommended Solution| Description|
|----------|----------|----------|------|
| Large model quantization| Models with a scale of 32 billion parameters (32B) or larger.| Layer-wise quantization| This solution significantly saves memory.|
| Memory-constrained environments| NPU memory is insufficient to load the entire neural network.| Layer-wise quantization| This solution drastically reduces memory overhead.|
| High-efficiency quantization| Ultra-large models or massive calibration datasets| DP layer-wise quantization| This solution significantly accelerates quantization through multi-device parallelism.|

#### 4.1.3 Working Principles and Advantages

| Feature| Traditional Quantization (Model-wise)| Layer-wise Quantization (Single-device)| DP Layer-wise Quantization (Multi-device DP)|
|----------|----------|----------|----------|
| **Processing method**| Full neural network processing at the model level| Sequential layer-wise processing on a single device| Parallel layer-wise processing on multiple devices|
| **Memory usage**| 2 to 3 times the model size| **2 to 3 times the size of a single layer**| **2 to 3 times the size of a single layer**|
| **Quantization efficiency**| Fast (for small models)| Slow (for large models)| **Significantly improved through multi-device parallelism**|
| **Applicable models**| Small models (< 32B)| Large models (≥ 32B)| Ultra-large models or massive calibration datasets|

#### 4.1.4 Configuration Method

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

#### 4.1.5 Precautions

1. **Distributed algorithm support**: When using `dp_layer_wise`, ensure that all processors (such as `linear_quant`) and algorithms (such as `minmax`, `ssz`, and `iter_smooth`) support distributed execution.
2. **Acceleration and calibration**: Acceleration achieved through multi-device parallelism depends on the size of the calibration dataset. If the dataset is too small, the communication overhead may outweigh the parallelization benefits, making the acceleration effect negligible.
3. **Multimodal restrictions**: DP layer-wise quantization currently does not support multimodal models. For multimodal scenarios, use single-device `layer_wise` quantization.

#### 4.1.6 Model Adaptation

Layer-wise quantization supports all models listed as compatible with quick quantization in the [Foundation Model Support Matrix](../../model_support/foundation_model_support_matrix.md).
DP layer-wise quantization inherits support from the layer-wise approach and is compatible with all supported Large Language Models (LLMs).

**Note**: DP layer-wise quantization currently does not support multimodal models. For multimodal models, use single-device `layer_wise` quantization.

#### 4.1.7 Algorithm Adaptation

Layer-wise quantization (`layer_wise`) supports all algorithms in the `modelslim_v1` architecture.

DP layer-wise quantization (`dp_layer_wise`) currently supports only the following algorithms:

**Outlier Suppression Algorithms**

| Algorithm Name| Type| Supported| Description|
|----------|----------|---------|------|
| Iterative Smooth | iter_smooth | ✅| Distributed execution is fully supported.|
| Flex Smooth Quant | flex_smooth | ✅| Distributed execution is fully supported.|
| Flex AWQ SSZ | flex_awq_ssz | ✅ | Distributed execution is fully supported. |
| QuaRot | quarot | ✅ | Distributed execution is fully supported. |
| Online QuaRot | online_quarot | ✅ | Distributed execution is fully supported. |
| Adapt Rotation | adapt_rotation | ✅ | Distributed execution is fully supported. |

**Quantization Algorithms**

| Algorithm Name| Quantization Method| Supported| Description|
|----------|---------|---------|------|
| MinMax | minmax | ✅| Distributed execution is fully supported.|
| SSZ | ssz | ✅| Distributed execution is fully supported.|
| CeilX | ceil_x | ✅ | Distributed execution is fully supported. |
| FA3 Quant | fa3_quant | ✅ | Distributed execution is fully supported. |

## 5. Quantization Configuration Protocol

### 5.1 Configuration Protocol Overview

The quick quantization configuration protocol is built on a hierarchical design that abstracts the entire quantization pipeline into a YAML schema. This includes the quantization service version, pipeline type, processing methods, saving policies, and calibration datasets. This approach allows developers to focus on high-level policies and workflows without hardcoding details in Python.

#### 5.1.1 Basic Structure

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

#### 5.1.2 Protocol Version Description

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

### 5.2 `modelslim_v1` Configuration

#### 5.2.1 Description

`modelslim_v1` is the next-generation quantization framework for ModelSlim and is rapidly evolving.

Compared with `modelslim_v0`, `modelslim_v1` has the following advantages:

- Algorithms are implemented independently, allowing for flexible configuration combinations.
- Layer-wise quantization is supported, which greatly reduces resource consumption.
- It operates without dependency on specific CANN versions.

#### 5.2.2 `runner` - Quantization Scheduler Type

**Purpose**: Specifies the type of quantization scheduler to be used.
**Type**: String
**Default value**: `"auto"`

| Value| Purpose| Application Scenario| Feature|
|--------|------|----------|------|
| auto | Automatically selects the optimal strategy| Most scenarios| The tool automatically selects the optimal strategy based on the model size, available memory, and device configuration.|
| layer_wise | Performs layer-wise quantization| Large models (≥ 32B)| This option features low memory usage. Adaptation may be required.|
| dp_layer_wise | Performs DP layer-wise quantization| Large models (≥ 32B) in multi-device scenarios| This option significantly improves quantization efficiency through multi-device parallelism.|
| model_wise | Performs model-wise quantization (non-layer-wise)| Small models (< 32B)| This option features high memory usage and good compatibility.|

#### 5.2.3 `prior` - Preprocessing Stage Configuration

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

#### 5.2.4 `process` - Processor Configuration Field

**Purpose**: Specifies a list of processors for quantization, which are executed in sequential order.

**Features**:

- **List structure**: The `process` field is a list containing multiple processor configurations, distinguished by their `type` field.
- **Sequential execution**: Processors are executed according to their order in the list.
- **Flexible combination**: Different processor types can be combined to implement complex quantization strategies. However, not all combinations are valid. Adhere to the following guidelines for successful configuration (if you are unsure, refer to the examples in this document or consult a specialist):
  - Perform outlier suppression before quantization. For example, when combining Iterative Smooth and W8A8 quantization, Iterative Smooth must be executed before the quantization step.
  - Avoid defining multiple quantization settings for the same layer. Duplicate definitions for a single layer may cause runtime errors or unexpected results, such as accuracy loss or quantization failure.

##### 5.2.4.1 Supported Processors

| Processor| Type| Configuration Example| Configuration Fields|
| :--- | :--- | :--- | :--- |
| SmoothQuant | Outlier suppression| [SmoothQuant Configuration Example](../../quantization_algorithms/outlier_suppression_algorithms/smooth_quant.md#yaml-configuration-example)| [SmoothQuant Configuration Fields](../../quantization_algorithms/outlier_suppression_algorithms/smooth_quant.md#yaml-configuration-fields)|
| Iterative Smooth | Outlier suppression| [Iterative Smooth Configuration Example](../../quantization_algorithms/outlier_suppression_algorithms/iterative_smooth.md#yaml-configuration-example)| [Iterative Smooth Configuration Fields](../../quantization_algorithms/outlier_suppression_algorithms/iterative_smooth.md#yaml-configuration-fields)|
| Flex Smooth Quant | Outlier suppression| [Flex Smooth Quant Configuration Example](../../quantization_algorithms/outlier_suppression_algorithms/flex_smooth_quant.md#yaml-configuration-example)| [Flex Smooth Quant Configuration Fields](../../quantization_algorithms/outlier_suppression_algorithms/flex_smooth_quant.md#yaml-configuration-fields)|
| Flex AWQ SSZ | Outlier suppression| [Flex AWQ SSZ Configuration Example](../../quantization_algorithms/outlier_suppression_algorithms/flex_awq_ssz.md#yaml-configuration-example)| [Flex AWQ SSZ Configuration Fields](../../quantization_algorithms/outlier_suppression_algorithms/flex_awq_ssz.md#yaml-configuration-fields)|
| AWQ | Outlier suppression | [AWQ YAML Example](../../quantization_algorithms/outlier_suppression_algorithms/awq_smooth.md#yaml-configuration-example) | [YAML Configuration Fields](../../quantization_algorithms/outlier_suppression_algorithms/awq_smooth.md#yaml-configuration-fields) |
| KV Smooth | Outlier suppression| [KV Smooth Configuration Example](../../quantization_algorithms/outlier_suppression_algorithms/kv_smooth.md#yaml-configuration-example)| [KV Smooth Configuration Fields](../../quantization_algorithms/outlier_suppression_algorithms/kv_smooth.md#yaml-configuration-fields)|
| QuaRot | Outlier suppression| [QuaRot Configuration Example](../../quantization_algorithms/outlier_suppression_algorithms/quarot.md#yaml-configuration-example)| [QuaRot Configuration Fields](../../quantization_algorithms/outlier_suppression_algorithms/quarot.md#yaml-configuration-fields)|
| Adapt Rotation | Outlier suppression | [Adapt Rotation YAML Example](../../quantization_algorithms/outlier_suppression_algorithms/adapt_rotation.md#yaml-configuration-example) | [Configuration Fields](../../quantization_algorithms/outlier_suppression_algorithms/adapt_rotation.md#yaml-configuration-fields) |
| linear_quant | Quantization| [Linear Quantization Configuration Example](../../quantization_algorithms/quantization_algorithms/linear_quant.md#yaml-configuration-example)| [Linear Quantization Configuration Fields](../../quantization_algorithms/quantization_algorithms/linear_quant.md#yaml-configuration-fields)|
| group | Quantization| [Group Configuration Example](group.md#42-yaml-configuration-example)| [Group Configuration Fields](group.md#43-yaml-configuration-fields)|
| KVCache Quant | Quantization| [KVCache Quant Configuration Example](../../quantization_algorithms/quantization_algorithms/kvcache_quant.md#yaml-configuration-example)| [KVCache Quant Configuration Fields](../../quantization_algorithms/quantization_algorithms/kvcache_quant.md#yaml-configuration-fields)|
| FA3 Quant | Quantization| [FA3 Quant Configuration Example](../../quantization_algorithms/quantization_algorithms/fa3_quant.md#yaml-configuration-example)| [FA3 Quant Configuration Fields](../../quantization_algorithms/quantization_algorithms/fa3_quant.md#yaml-configuration-fields)|
| Float Sparse | Quantization| [Float Sparse Configuration Example](../../quantization_algorithms/quantization_algorithms/float_sparse.md#yaml-configuration-example)| [Float Sparse Configuration Fields](../../quantization_algorithms/quantization_algorithms/float_sparse.md#yaml-configuration-fields)|
| AutoRound | Quantization| [AutoRound Configuration Example](../../quantization_algorithms/quantization_algorithms/autoround.md#yaml-configuration-example)| [AutoRound Configuration Fields](../../quantization_algorithms/quantization_algorithms/autoround.md#yaml-configuration-fields)|
| SVDQuant (W4A4 scheme)| Comprehensive scheme| [SVDQuant Configuration Example](../../quantization_algorithms/quantization_algorithms/svdquant.md#yaml-configuration-example)| [SVDQuant Configuration Fields](../../quantization_algorithms/quantization_algorithms/svdquant.md#yaml-configuration-field-detailed-explanation)|

#### 5.2.5 `save` (Saver Configuration Fields)

**Purpose**: Defines the list of savers for storing quantization results. `modelslim_v1` currently supports two savers: `ascendv1_saver` (for Ascend inference, default) and `compressed_tensors` (for vLLM and the broader Hugging Face ecosystem). For a format comparison, see the [Format Support Matrix](../../quantization_formats/README.md).

##### 5.2.5.1 ascendv1_saver

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

##### 5.2.5.2 compressed_tensors

**Purpose**: Saves models in the [compressed-tensors](../../quantization_formats/compressed_tensors.md) format, targeting HuggingFace‑ecosystem inference frameworks such as vLLM. Quantization metadata is written to the `quantization_config` field in `config.json`, and weights are saved as `model*.safetensors` files.

**Use Cases**:

- Target inference framework is vLLM or another engine that supports HF `quantization_config`.
- Currently supports **W8A8 static quantization** (`act.scope: "per_tensor"`) and **W8A8 dynamic quantization** (`act.scope: "per_token"`).

**Configuration Example**:

```yaml
spec:
  save:
    - type: "compressed_tensors"   # Saver type: saves in compressed-tensors format
      part_file_size: 4            # Shard file size (GB); 0 means no sharding
```

**Field Descriptions**:

| Field | Purpose | Description |
|-------|-------------|-------|
| `type` | Saver type identifier | Fixed value: `"compressed_tensors"` |
| `part_file_size` | Shard file size | Size per shard file in GB; `0` disables sharding |

**Limitations**:

- Currently only supports linear layer quantization; KV cache quantization is not supported.
- Distributed export is not supported.

For format specifications and tensor naming conventions, refer to the [compressed-tensors Format Guide](../../quantization_formats/compressed_tensors.md).

**Complete Configuration Example** (W8A8 static quantization + compressed-tensors save):

```yaml
apiversion: modelslim_v1

spec:
  process:
    - type: "linear_quant"
      qconfig:
        act:
          scope: "per_tensor"      # W8A8 Static: activation quantization
          dtype: "int8"
          symmetric: False
          method: "minmax"
        weight:
          scope: "per_channel"
          dtype: "int8"
          symmetric: True
          method: "minmax"
      include: ["*"]

  save:
    - type: "compressed_tensors"
      part_file_size: 4
```

#### 5.2.6 `dataset` - Calibration Dataset Configuration

**Purpose**: Specifies the name of the calibration dataset file. The system matches the specified file within the `lab_calib` directory.
**Type**: String
**Default value**: `"mix_calib.jsonl"`

| Attribute| Description|
|------|------|
| File location| Located within the `lab_calib` directory.|
| File format| JSONL format.|
| Purpose| Specifies the calibration dataset for activation quantization.|

#### 5.2.7 Examples

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

### 5.3 `multimodal_sd_modelslim_v1` Configuration

#### 5.3.1 Description

`multimodal_sd_modelslim_v1` is designed for multimodal **generative** models such as text-to-video and image-to-video. Built on the `modelslim_v1` framework, it defaults to layer-wise quantization (`layer_wise`).

**Key Features**

- **Multimodal Generation** – Supports calibration samples from text, images, and other modalities. Samples are bridged via adapters to the original inference engine for floating-point replay and dump.
- **Dual Orchestration Paths** – Automatically dispatches based on the model adapter interface:
  - New models use `MultimodalPipelineInterface` (`inference_config` + `prepare_calib_data`).
  - Models already integrated in the main repository retain `LegacyMultimodalPipelineInterface` (`model_config` + `run_calib_inference`).
- **Multi-Expert Quantization** – For dual-DiT experts (e.g., Wan2.2), performs dump, quantization, and saving per expert name.
- **Calibration Data Caching** – Stores calibration data as per-expert `.pth` files. If a file exists, it is loaded; otherwise, floating-point inference is triggered to generate it.

**Supported Model Types** (The `--model_type` flag must match the names defined in [`config/config.ini`](../../../../../config/config.ini).)

| Orchestration | Typical `model_type` | Description |
|---------------|----------------------|-------------|
| Refactored | `Wan2.2-T2V-A14B`, `Wan2.2-I2V-A14B`, `Wan2.2-TI2V-5B`, `HunyuanVideo` | Uses `inference_config`. See the [Multimodal Generation Model Integration Guide](../../../development_guide/integrating_multimodal_generation_model.md) for details. |
| Legacy | `Wan2_1` / `Wan2.1`, `Wan2_2` / `Wan2.2` (single), `flux1`, `qwen_image_edit`, etc. | Uses the legacy `model_config`; behavior remains compatible with historical main‑repository versions. |

**Configuration Notes**

- `spec.dataset` – Calibration samples (short names or paths). Under the refactored path, the adapter's `handle_dataset` loads them as a list of `VlmCalibSample`.
- `multimodal_sd_config.dump_config` – Directory for calibration `.pth` files and capture mode.
- `multimodal_sd_config.inference_config` – **Recommended**. Inference parameters are strongly validated via Pydantic and bridged to the original inference engine CLI.
- `multimodal_sd_config.model_config` – **Deprecated** (legacy only). Do not configure this together with `inference_config`.

#### 5.3.2 `runner` - Quantization Scheduler Type

Due to GPU memory constraints, multimodal generative models currently default to and support only layer-wise quantization (`layer_wise`). By default, `runner` does not need to be configured. If it is set to any value other than `layer_wise`, a warning will be triggered, and the system will automatically switch to `layer_wise` mode.

#### 5.3.3 `process` - Processor Configuration Field

This configuration field is identical to the one for `modelslim_v1`. For details, see [`modelslim_v1` Configuration/`process` - Processor Configuration Field](#524-process---processor-configuration-field).

#### 5.3.4 <span id="save---saver-configuration-field-sd">save - Saver Configuration Field</span>

**Purpose**: Defines the list of savers for storing quantization results.

##### 5.3.4.1 `mindie_format_saver`

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

#### 5.3.5 `multimodal_sd_config` - Multimodal Generative Specific Configuration Field

**Purpose**: Defines configuration parameters specific to multimodal generative models, including calibration data capture, model loading, and inference configurations.

##### 5.3.5.1 <span id="dump_config---calibration-data-capture-configuration">`dump_config` — Calibration Data Capture Configuration</span>

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
| `enable_dump` | Enables dump. | **Legacy path**: When `False`, no load/dump occurs; `calib_data` for each expert is set to `None` (explicit YAML config, no extra confirmation required). <br><br> **Refactored path**: Quantization service delegates calibration prep to the adapter's `prepare_calib_data`. Typically works with per‑expert `.pth` files under `dump_data_dir` — loads if present, otherwise triggers floating‑point dump. <br><br> For purely dynamic quantization (e.g., W8A8 MXFP8), set to `False`; `.pth` files are not required, but `calib_data` keys must still exist for each expert. | `True` (default) / `False` |
| capture_mode | Specifies the data capture mode.| Specifies how model input data is captured.                                                                                                | Currently, only `"args"` is supported. Other modes will be supported in the future.|
| `dump_data_dir` | Calibration data directory | Root directory for retrieving and saving `.pth` files. If empty, uses the weight `save_path`. Under the refactored path, filenames are generated per expert as `calib_data_<task_config>_<expert_name>.pth` (e.g., `calib_data_t2v-A14B_low_noise_model.pth`). **If all `.pth` files exist, they are loaded directly; if any are missing, floating-point inference is performed to dump and write them.** | String (path) |

**Capture Mode Description**

- **`args`**: Captures positional arguments; suitable for most multimodal generation models.

**Calibration Data File Naming (Refactored Path)**

- Single DiT (e.g., HunyuanVideo): `calib_data_<task_config>_.pth` (`expert_name` is an empty string)
- Dual-expert Wan2.2: `calib_data_<task_config>_low_noise_model.pth`, `calib_data_<task_config>_high_noise_model.pth`
- During quantization, each expert returned by `init_model()` must have a corresponding **key** in `calib_data`; missing keys cause a fail-fast error. Setting `calib_data[expert] = None` indicates the expert has no dump tensors (e.g., fully dynamic quantization), but the key must still be present.

**Legacy Path Notes**: When `enable_dump=False`, no load/dump occurs, and `calib_data` for each expert is set directly to `None`. Legacy cache files are still named `calib_data_<task_config>_<expert>.pth` (consistent with the refactored naming), and the single `calib_data.pth` file is no longer used.

##### 5.3.5.2 `inference_config` – Inference Parameters (Recommended)

**Purpose**: Configures floating-point inference replay and quantization bridging parameters. The quantization service calls `validate_inference_config`, which uses the adapter-declared `InferenceConfig` (Pydantic, `extra="forbid"`) for validation, then merges the result via `configure_runtime` into the original inference engine's `model_args`.

**Example Configuration** (Wan2.2-T2V-A14B):

```yaml
spec:
  dataset: wan2_2_t2v
  multimodal_sd_config:
    inference_config:
      size: "1280*720"
      frame_num: 81
      sample_steps: 40
      convert_model_dtype: True
      task: "t2v-A14B"   # Must match the current --model_type; do not use this to switch between T2V/I2V/TI2V
```

**Field Description**:

| Aspect | Description |
|--------|-------------|
| Valid fields | Determined by each model adapter's `*InferenceConfig` and the original inference engine CLI; invalid fields cause validation failure. |
| Relationship with `model_type` | The scenario is fixed by the CLI `--model_type` (e.g., `Wan2.2-T2V-A14B`). Do not rely solely on the `task` field in YAML to switch scenarios. |
| Mutual exclusivity | Cannot be used together with `model_config` in the same YAML. |

##### 5.3.5.3 `model_config` – Model Loading & Inference Configuration (Legacy, Deprecated)

**Purpose**: Read only by **Legacy** adapters (`LegacyMultimodalPipelineInterface`) during the `set_model_args` phase. Field definitions are determined by the original inference engine.

**Migration**: New integrations and refactored models should use `inference_config` instead. Using `model_config` alone triggers a deprecation warning; configuring it together with `inference_config` causes an error.

**Configuration Example** (Legacy Wan2.1):

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
```

#### 5.3.6 `dataset` – Calibration Dataset Configuration

**Purpose**: Specifies calibration samples for use by `handle_dataset` (refactored path) or `run_calib_inference` (legacy path).

**Type**: `string` – a short name, absolute path, or relative path. Short names are resolved under [`lab_calib`](../../../../../lab_calib/).

**Sample Format**: Recommended format is **`index.json` / `index.jsonl`**, where each entry contains at least a non-empty **`text`** field. For image-to-video scenarios, an **`image`** field is required per model specifications. Field conventions are similar to those for multimodal understanding; see [dataset – Calibration Data Path Configuration](#dataset---calibration-data-path-configuration) for details.

**Sample Requirements by Scenario** (refactored Wan2.2):

| `model_type` | Sample Requirements |
|--------------|----------------------|
| `Wan2.2-T2V-A14B` | Must contain `text`; `image` is not allowed |
| `Wan2.2-I2V-A14B` | Must contain `text` and an accessible `image` |
| `Wan2.2-TI2V-5B` | Must contain `text`; `image` is optional |

For calibration `.pth` generation and reuse logic, refer to [dump_config – Calibration Data Capture Configuration](#dump_config---calibration-data-capture-configuration).

#### 5.3.7 Usage Examples

- Wan2.1 Legacy W8A8 dynamic quantization: [wan2_1_w8a8_dynamic.yaml](../../../../../lab_practice/wan2_1/wan2_1_w8a8_dynamic.yaml)
- Wan2.2-T2V W8A8 MXFP8 + QuaRot + FA3: [wan2_2_w8a8f8_mxfp_t2v.yaml](../../../../../lab_practice/wan2_2/wan2_2_w8a8f8_mxfp_t2v.yaml)
- Wan2.2-I2V W8A8 MXFP8 + QuaRot + FA3: [wan2_2_w8a8f8_mxfp_i2v.yaml](../../../../../lab_practice/wan2_2/wan2_2_w8a8f8_mxfp_i2v.yaml)
- Wan2.2-TI2V W8A8 MXFP8 + QuaRot + FA3: [wan2_2_w8a8f8_mxfp_ti2v.yaml](../../../../../lab_practice/wan2_2/wan2_2_w8a8f8_mxfp_ti2v.yaml)
- HunyuanVideo: [hunyuan_video_w8a8f8_mxfp.yaml](../../../../../lab_practice/hunyuan_video/hunyuan_video_w8a8f8_mxfp.yaml)

### 5.4 `multimodal_vlm_modelslim_v1` Configuration

#### 5.4.1 Description

`multimodal_vlm_modelslim_v1` is a quantization service specifically designed for multimodal vision language models (VLMs). It is built on the `modelslim_v1` framework.

**Key Features**

- **Multimodal VLM support**: Provides model optimization for image-text multimodal understanding.
- **Layer-wise processing**: Employs layer-wise quantization, significantly reducing GPU memory consumption during foundation model quantization.
- **Multiple dataset formats**: Supports image directories and multimodal calibration datasets with custom text prompts through JSON or JSONL formats.

**Supported Model Types**

- Qwen2.5-Omni series: multimodal end-to-end models (text, image, audio, and video), such as Qwen2.5-Omni-7B.
- Qwen3-VL-MoE series: multimodal models, such as Qwen3-VL-235B-A22B and Qwen3-VL-30B-A3B.
- Other multimodal VLM models: Refer to the [Multimodal Model Support List](../../model_support/foundation_model_support_matrix.md#foundation-model-support-matrix).

**Configuration Characteristics**

- Supports calibration dataset configuration through the `dataset` field across three methods. Method 1 uses `index.json` or `index.jsonl` (recommended, supports multimodality). Method 2 uses a pure image directory (deprecated). Method 3 uses an image directory combined with a single JSON or JSONL file (deprecated). For details, see [`dataset` - Calibration Data Path Configuration](#dataset---calibration-data-path-configuration).
- Supports default text prompt configuration through the `default_text` field. This field is mandatory for Method 2. For Method 1, it is used if the `text` field is missing from an entry.
- Uses `layer_wise` (layer-wise quantization) mode by default to optimize large-scale multimodal models.

#### 5.4.2 `runner` - Quantization Scheduler Type

Due to GPU memory constraints, multimodal VLM models currently default to and support only layer-wise quantization (`layer_wise`). By default, `runner` does not need to be configured. If it is set to any value other than `layer_wise`, a warning will be triggered, and the system will automatically switch to `layer_wise` mode.

#### 5.4.3 <span id="process---processor-configuration-field -vlm">`process` - Processor Configuration Field</span>

This configuration field is identical to the one for `modelslim_v1`. For details, see [`modelslim_v1` Configuration/`process` - Processor Configuration Field](#524-process---processor-configuration-field).

#### 5.4.4 `default_text` - Default Text Prompt Configuration

**Function**: Specifies the default text prompt uniformly for all calibration images.
**Type**: String
**Default value**: `"Describe this image in detail."`
**Restrictions**: The text prompt cannot be an empty string. This field is invalid when the `dataset` field is configured as an image directory that contains JSON or JSONL files (used to describe the custom text prompt for each image).

#### 5.4.5 <span id="save---saver-configuration-field-vlm">save - Saver Configuration Field</span>

This configuration field is identical to the one for `modelslim_v1`. For details, see [modelslim_v1 Configuration/`save` - Saver Configuration Field](#525-save-saver-configuration-fields).

**Recommended Configuration**

```yaml
spec:
  save:
    - type: "ascendv1_saver"    # Specifies the ascendv1 saver type.
      part_file_size: 4        # Specifies the shard file size (GB). You are advised to save large models in shards.
```

#### 5.4.6 <span id="dataset---calibration-data-path-configuration">`dataset` - Calibration Data Path Configuration</span>

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

#### 5.4.7 Example

- W8A8 quantization for the Qwen2.5-Omni model: [qwen2_5_omni_thinker_w8a8.yaml](https://gitcode.com/Ascend/msmodelslim/blob/26.1.0/lab_practice/qwen2_5_omni_thinker/qwen2_5_omni_thinker_w8a8.yaml)
- W8A8 mixed quantization for the Qwen3-VL-MoE model: [qwen3_vl_moe_w8a8.yaml](https://gitcode.com/Ascend/msmodelslim/blob/26.1.0/lab_practice/qwen3_vl_moe/qwen3_vl_moe_w8a8.yaml)

### 5.5 `modelslim_v0` Configuration

#### 5.5.1 Description

The `modelslim_v0` quantization service is primarily composed of legacy interfaces, such as `Calibrator` and `AntiOutlier`. Its configuration protocol (`YAML`) remains fundamentally consistent with the original Python API interfaces, facilitating a smooth migration from older versions.

**Related Documents**

- [Calibrator.md](../../../python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_Calibrator.md)
- [AntiOutlier.md](../../../python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/AntiOutlier.md)

**Note**: The `modelslim_v0` protocol version is deprecated and will soon be phased out. It is not recommended for use. You are advised to use `modelslim_v1` or later.

## 6. Appendixes

### 6.1 References

- For ultra-large models, you can use layer-wise quantization to significantly reduce GPU memory consumption. For details, see [Layer-wise and DP Layer-wise Quantization](#41-layer-wise-and-dp-layer-wise-quantization).
- For details about algorithms supported for quick quantization, see [Quantization Algorithms Supported for Quick Quantization V1](../../quantization_algorithms/README.md).

### 6.2 FAQs

Q1: How Do I Select a Proper Quantization Scheduler?

- `auto`: suitable for most scenarios. The tool automatically selects the optimal strategy based on model size, available memory, and device configuration.
- `layer_wise`: suitable for models with a scale of 32B or larger, or memory-constrained environments.
- `dp_layer_wise`: suitable for ultra-large models or multi-device scenarios with large-scale calibration datasets.
- `model_wise`: suitable for small models (< 32B), offering the best compatibility.

Q2: What Are the Differences Between Layer-Wise Quantization and DP Layer-Wise Quantization?

- **Layer-wise quantization**: performs quantization layer by layer on a single device, where the memory usage is 2 to 3 times the size of a single layer.
- **DP layer-wise quantization**: performs quantization layer by layer in parallel across multiple devices, significantly increasing quantization speed while maintaining the low memory usage advantage. For details, see [Layer-wise and DP Layer-wise Quantization](#41-layer-wise-and-dp-layer-wise-quantization).

Q3: How Do I Determine Whether to Use Layer-Wise Quantization?

You are advised to use layer-wise quantization in the following situations:

- Model scale > 32B.
- Insufficient NPU memory.
- Out-of-memory (OOM) errors occur during quantization.

Q4: Does Multi-Device Quantization Always Accelerate the Process?

Not necessarily. The acceleration effect of multi-device quantization is affected by several factors:

- Calibration dataset size: If the calibration dataset is small, communication overhead may exceed the parallelization gains.
- Number of devices: A higher number of devices does not guarantee faster speed. Choose a reasonable number based on the actual situation.
- Algorithm support: Only algorithms that support distributed execution can function correctly in multi-device environments. For details, see [Layer-wise and DP Layer-wise Quantization](#41-layer-wise-and-dp-layer-wise-quantization).

Q5: How Do I Tune the Quantization Accuracy?

For details about quantization accuracy tuning, see [Quantization Accuracy Tuning Guide](../../../best_practices/quantization_precision_tuning_guide.md).
