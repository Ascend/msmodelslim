# Standing High Tuning Algorithm

## Overview

Standing High is an auto tuning algorithm based on quantization fallback layer selection and outlier suppression strategies. By gradually reducing the quantization fallback level (`disable_level`) and testing different outlier suppression strategies, the algorithm identifies the optimal quantization configuration that maximizes the number of quantized layers while meeting accuracy requirements. Quantization fallback refers to the process where specific layers bypass quantization and are reverted to floating-point precision.

## **Principles and Features**

### Principle

The core idea of the Standing High algorithm is "standing high": the algorithm first performs a floating-point pre-check to evaluate the accuracy of the all-fallback floating-point baseline. If the floating-point baseline cannot meet the accuracy requirement, the algorithm directly reports an error and exits; otherwise, it evaluates the initial configuration in the configuration file and returns directly if the accuracy requirement is met. Otherwise, it uses binary search to quickly locate the minimum fallback level that meets the accuracy requirement, and then gradually reduces the number of fallback layers through the standing-high process, combining different outlier suppression strategies to find the optimal configuration. The algorithm uses an adaptive step mechanism to speed up the search when a configuration that meets the requirement is found.

**1. Initialization Phase**

The algorithm first loads the model and the quantization fallback layer selector. It then runs the layer selector by using the calibration dataset (`template.dataset`) defined in the configuration file to determine which layers must revert to floating-point precision. The layer selector is implemented based on sensitive layer analysis, which evaluates the sensitivity of each layer to quantization accuracy based on the model structure and calibration data. Finally, it groups the layers to facilitate the subsequent selection of the quantization fallback level.

**2. Floating-Point Pre-Check (All-Fallback Baseline Evaluation)**

Before entering the quantization search, the algorithm first builds an "all-fallback floating-point" baseline configuration: all `linear_quant` processors exclude all layers (no Linear layer is quantized), and no outlier suppression processor is injected, producing a pure floating-point model without quantization or weight transformation, whose accuracy is then evaluated.

- If the floating-point baseline meets the accuracy requirement, it indicates that the accuracy target is achievable on the floating-point model, and the subsequent search continues;
- If the floating-point baseline does not meet the accuracy requirement, it indicates that the accuracy target is set too high (even the floating-point model cannot reach it). The algorithm terminates immediately, throws `SpecError` (error code 302), exits, and prompts you to relax the accuracy expectation. This avoids continuing to consume quantization and evaluation resources on an unattainable accuracy target.

**3. Initial Evaluation**

The algorithm generates and evaluates a quantization configuration by using the initial configuration (`template`) provided in the configuration file. The initial configuration is based on the quantization configuration within the `template`. If this configuration meets the accuracy requirements, the algorithm returns the results immediately and terminates the tuning process.

**4. Binary Search for the Minimum Fallback Level**

If the initial configuration fails to meet accuracy requirements, the algorithm identifies the minimum fallback level that meets those requirements by using a binary search. The fallback levels are automatically computed by the layer selector based on the model structure. A higher level indicates that more layers will revert to floating-point precision. The algorithm performs the binary search between level 1 and the maximum fallback level to quickly locate the optimal starting point.

**5. Standing-High Process**

Starting from the identified minimum fallback level, the algorithm performs the Standing High process to gradually reduce the number of fallback layers and quantize more layers:

(1) **Reduce the fallback level**: Starting from the current fallback level, the algorithm attempts to reduce the fallback level. This reduces the number of layers that revert to floating-point precision and allows for the quantization of more layers.

(2) **Traverse outlier suppression strategies**: For each fallback level, the algorithm traverses all outlier suppression strategies (`anti_outlier_strategies`) defined in the configuration file and tries different strategy combinations.

(3) **Evaluate configurations**: The algorithm generates quantization configurations and evaluates the accuracy for each combination of fallback level and outlier suppression strategy.

(4) **Apply an adaptive step mechanism**: If the algorithm identifies a configuration that meets the accuracy requirements, it records that configuration and doubles the step (up to the current fallback level) to accelerate the search. If no strategies meet the requirements, the step is reset to 1 and the algorithm continues with further attempts.

(5) **Termination conditions**: The algorithm terminates and returns the currently recorded optimal configuration when any of the following occurs: the initial configuration already meets the accuracy requirement; during the standing-high process the fallback level has reached 0 and all outlier suppression strategies have been tried; or the step has reached 1 and none of the outlier suppression strategies meets the accuracy requirement. In addition, if the floating-point pre-check fails, the algorithm directly reports an error and exits before the search starts (see "Floating-Point Pre-Check").

**6. Traversal of Outlier Suppression Strategies**

The algorithm possesses a memory function when traversing outlier suppression strategies. It improves efficiency by using a starting point based on the last successful strategy. The traversal follows the order of the `anti_outlier_strategies` defined within the configuration file.

### Features

1. **Efficient search mechanism**: The algorithm uses two mechanisms to improve the search efficiency:
   - **Binary search**: When locating the minimum fallback level, the algorithm utilizes a binary search to quickly narrow the search scope.
   - **Adaptive step**: During the Standing High process, the step is doubled to accelerate the search if the algorithm finds a configuration that meets the requirements.

2. **Strategy memory**: The algorithm retains a memory of the outlier suppression strategy traversal. It improves efficiency by starting from the last successful strategy.

3. **Step-by-step optimization**: Starting from the minimum fallback level that meets accuracy requirements, the algorithm gradually reduces the number of fallback layers to quantize more layers and find the optimal configuration.

4. **Floating-point pre-check**: The all-fallback floating-point baseline is evaluated at the start of tuning. If the accuracy target cannot be met even by the floating-point model, `SpecError` (error code 302) is thrown immediately and the process exits, avoiding an invalid search under an unattainable accuracy target.

## Application Requirements

**Inference engine support**: Pay close attention to the support for arbitrary fallback within the inference engine. Generally, vLLM-Ascend supports arbitrary fallback in single-operator mode, but it may not provide this support when hybrid operators are used. When using the Standing High algorithm, ensure the configured inference engine (such as vLLM-Ascend) supports the quantization fallback feature.

**Model adaptation**: When the strategy starts, it **automatically runs sensitive layer analysis** to generate fallback candidates. The model adapter must implement **`ModelSlimPipelineInterfaceV1`** (that is, `PipelineInterface` in `core/runner/pipeline_interface.py`). The analysis is performed by `PipelineAnalysisService` calling the adapter's `init_model`, `handle_dataset`, and visit/forward pipelines through `LayerWiseRunner`, and it **does not call `load_model` in advance on the strategy side**. This is consistent with the requirements of the CLI command `msmodelslim analyze`. For details, see the [LLM Integration Guide](../../ptq/llm/integration_guide_large_language_model_quantization.md).

## Function Description

### Usage Description

It is started through `msmodelslim tune` in the auto tuning process. In the tuning YAML file, set `type` to `standing_high` in the `strategy` field, and configure the other fields as described below. For the complete tuning configuration and command parameters, see [Auto Tuning Instructions](../../../user_guide/usage_auto_precision_tuning.md).

### YAML Configuration Example

Configuration under the `strategy` field in the auto tuning configuration file

```yaml
strategy:
  type: standing_high
  anti_outlier_strategies:
    - [ ]
    - - type: "iter_smooth"
        alpha: 0.5
  template:
    runner: auto
    process:
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
        include: [ "*" ]
        exclude: [ ]
    save:
      - type: ascendv1_saver
        part_file_size: 4
    dataset: mix_calib.jsonl
  metadata:
    config_id: standing_high
    label:
      w_bit: 8
      a_bit: 8
      is_sparse: false
      kv_cache: false
```

### YAML Configuration Fields

#### `type` - Strategy Type

**Purpose**: Specifies the type of the tuning algorithm. When the Standing High algorithm is used, set this field to `standing_high`.

**Type**: String

**Value**: `standing_high`

#### `anti_outlier_strategies` - Outlier Suppression Strategy List

**Purpose**: Specifies a list of outlier suppression strategies, where each element is a list of processor configurations. The algorithm traverses these strategy combinations to find the configuration that meets the accuracy requirements.

**Configuration Description**

- An empty list `[]` can be configured, indicating that no outlier suppression strategy is used.
- A single outlier suppression strategy can be configured, such as `iter_smooth`.
- A mixture of multiple outlier suppression strategies can be configured by placing multiple strategies within the same list.

**Configuration Examples**

```yaml
# No outlier suppression strategy is used.
anti_outlier_strategies: [ ]
```

```yaml
anti_outlier_strategies:
  - - type: "iter_smooth"  # Only a single outlier suppression strategy is used.
      alpha: 0.5
  - - type: "iter_smooth"  # A mixture of strategies is used: iter_smooth and quarot.
      alpha: 0.5
    - type: "quarot"
```

#### `template` - Basic Quantization Configuration

**Purpose**: Specifies the basic quantization configuration template that defines the foundational settings for quantization, including the quantization scheduler, processor, saver, and dataset. This configuration serves as the starting point for tuning. The selection of the basic configuration affects the number of tuning iterations to some extent.

**Configuration protocol**: The configuration protocol of the `template` field is consistent with the `spec` field in the one-click quantization configuration protocol. For detailed configuration instructions, see the [modelslim_v1 Configuration Guide](../../../api_reference/config/quant/modelslim_v1.md).

**Core Fields**

- **runner**: quantization scheduler type, which defines the scheduling mode for quantization processing (such as `auto`, `layer_wise`, `dp_layer_wise`, or `model_wise`).
- **process**: processor list, which defines the processor configuration for quantization processing (such as `linear_quant` or `iter_smooth`).
- **save**: saver list, which defines the mode for saving quantization results (such as `ascendv1_saver`).
- **dataset**: calibration dataset configuration, which specifies the name of the calibration dataset file.

**Configuration Requirements**

- Must contain at least one `linear_quant` processor.
- Can contain other processors (such as the outlier suppression processor).

**Configuration Example**

```yaml
template:
  runner: auto
  process:
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
      include: [ "*" ]
      exclude: [ ]
  save:
    - type: ascendv1_saver
      part_file_size: 4
  dataset: mix_calib.jsonl
```

#### `metadata` - Strategy Metadata

**Purpose**: Defines the metadata information for a strategy. This information is used to identify and classify quantization configurations (such as quantization bit width labels) so that they can be easily retrieved and reused in the best practice library.

If the environment variable `MSMODELSLIM_CUSTOM_PRACTICE_REPO` is set before auto tuning is started, the final quantization configuration will be written to the best practice library corresponding to this path after the accuracy requirements are met. When performing quick quantization (`msmodelslim quant`) later, you only need to specify the same `quant_type` in the same `MSMODELSLIM_CUSTOM_PRACTICE_REPO` environment. The tool will then automatically match and use the corresponding configuration from the best practice library for quick quantization based on the retrieval rules of the library.

**Field Description**

| Field| Purpose| Type| Description|
|--------|------|------|------|
| config_id | Specifies the configuration file name.| string | Identifier for a quantization configuration.|
| label | Specifies the configuration file label.| object | The label information for a quantization configuration includes the number of quantization bits and sparsity.|

**`label` Field Description**

| Field| Purpose| Type| Description|
|--------|------|------|------|
| w_bit | Specifies weight quantization bits.| int | For example, `8` indicates 8-bit quantization.|
| a_bit | Specifies activation quantization bits.| int | Number of bits for activation quantization. For example, `8` indicates 8-bit quantization.|
| is_sparse | Specifies whether to enable sparse quantization.| bool | Specifies whether to enable sparse quantization.|
| kv_cache | Specifies whether to quantize the KV cache.| bool | Specifies whether to quantize the KV cache.|

**Configuration Example**

```yaml
metadata:
  config_id: standing_high
  label:
    w_bit: 8
    a_bit: 8
    is_sparse: false
    kv_cache: false
```
