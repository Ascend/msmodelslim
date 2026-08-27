# AutoRound: Low-bit Quantization Algorithm

## Overview

- **Origin**: A low-bit Sign Stochastic Gradient Descent-based (SignSGD-based) quantization method developed by Intel.
- **Background**: Traditional quantization methods, such as standard rounding, are not the optimal choice for weight quantization. These methods often introduce significant quantization error that substantially reduces model accuracy. This loss is especially pronounced in low-bit scenarios, such as 4-bit quantization or lower.
- **Core idea**: The algorithm introduces learnable rounding offset parameters and uses the SignSGD optimizer to adaptively adjust the rounding direction of each weight. It employs a temperature scheduling strategy to gradually harden the rounding operations, which effectively reduces quantization reconstruction error. This approach achieves an optimal balance between model accuracy and compression efficiency in ultra-low-bit scenarios.

## Preparations

Install msModelSlim. For details, see [msModelSlim Installation Guide](../../../install_guide/install_guide.md).

## Principle and Implementation

### Principle

The core idea of AutoRound is to optimize the rounding process for weights. Rather than using standard rounding, the algorithm uses a SignSGD-based approach to adaptively learn the optimal rounding direction (up or down) for each weight while performing targeted adjustments to the scaling factor and zero point.

**Core Formula**

In traditional quantization, the quantization formula of weight `W` is typically as follows:

Ŵ = s × clip(⌊W/s + zp⌉, n, m)

where `s` represents the scaling factor, `zp` represents the zero point, and `n` and `m` represent the lower and upper bounds of the quantized value.

AutoRound introduces a learnable rounding offset `V` and optional scaling factor adjustment parameters `α` and `β`:

`Ŵ = s × clip(⌊W/s + zp + V⌉, n, m)`, `s = (max(W) × α - min(W) × β) / (2^bit - 1)`

where `V` controls the rounding direction, while `α` and `β` are used to adjust the range of the scaling factor.

**Algorithm Process**

As a layer-wise optimization algorithm, it performs the following steps for each decoder layer:

1. **Baseline establishment**: Perform floating-point forward propagation and record the original output as the accuracy baseline.
2. **Parameter initialization**: Initialize the scaling factor and rounding offset, then introduce them into the quantization process as trainable parameters.
3. **Quantization reconstruction**: Perform quantization and dequantization operations, then execute forward propagation to obtain the quantization result.
4. **Loss calculation**: Compare the difference between the quantized output and the floating-point output to calculate the reconstruction loss.
5. **Parameter update**: Update the scaling factor and rounding offset through the SignSGD optimizer.
6. **Iterative optimization**: Repeat steps 1 through 5 until the convergence condition is met or the maximum number of iterations is reached.
7. **Final quantization**: Apply the optimal parameters obtained through iterative training to produce the final quantized weights.

### Implementation

#### Code Implementation

- The algorithm is implemented in [msmodelslim/processor/quant/autoround.py](../../../../../msmodelslim/processor/quant/autoround.py), with the core class being `AutoroundQuantProcessor`.

#### Initialization Phase

  * Layer configuration initialization: Read the quantization configuration and allocate the corresponding scheme to each network layer.
  * Parameter pre-allocation: Initialize the floating-point output, quantized output, and optimal parameters.

#### `pre_run` Phase

  * Gradient freezing: Disable automatic gradient calculation for all network layers to prevent direct weight optimization during training.

#### `preprocess` Phase (Executed in a Layer-wise Loop)

  * Baseline output collection: Perform floating-point forward propagation for the current layer and record the floating-point output as the optimization baseline.
  * Linear layer wrapping: Wrap each linear layer to inject trainable scaling factors and rounding offset parameters.
  * Computational graph construction: Construct a differentiable computational graph that includes quantization and dequantization operations to support gradient backpropagation.

#### `process` Phase (Executed in a Layer-wise Loop)

  * Trainer initialization: Set parameters such as the learning rate and iteration count, then configure the SignSGD optimizer.
  * Input and output configuration: Use the quantized output of the previous layer as the input for the current layer, and define the optimization objective as the difference between the floating-point output and quantized output.
  * Parameter optimization: Update the scaling factor and rounding offset through multiple iterations to minimize reconstruction error.
  * Convergence monitoring: Monitor loss changes in real time and stop optimization when the convergence threshold or the maximum number of iterations is reached.

#### `postprocess` Phase (Executed in a Layer-wise Loop)

  * Parameter application: Apply the optimized quantization parameters to the weight quantization of the corresponding layer.
  * Unwrapping: Remove the wrapping from the linear layers and restore the original network structure.
  * Forward propagation: Perform forward propagation on the quantized current layer and use the result as the input for the next layer.

#### `post_run` Phase

  * Cleanup: Remove temporary attributes from all modules to complete the final cleanup of the quantization process.

## Application Requirements

- **Low-bit quantization**: This algorithm is suitable for 4-bit quantization in extremely low-bit quantization scenarios.
- **High precision**: This algorithm maintains high model accuracy even under low-bit conditions.
- **Compute resources**: Additional optimization is required, and the computing cost is higher than that of simple quantization methods.
- **Usage limitations**:
  - This algorithm is applicable to the quantization of linear layers in LLMs.
  - Sufficient calibration data or a sufficient number of training iterations are required to optimize parameters.
  - **Low-bit quantization highly depends on effective outlier suppression algorithms. It is recommended that you use AutoRound in combination with outlier suppression methods such as [QuaRot](../outlier_suppression_algorithms/quarot.md) or [Iterative Smooth](../outlier_suppression_algorithms/iterative_smooth.md). Independent use of AutoRound is not recommended, especially for beginners who lack experience in quantization tuning. Failure to follow these recommendations may lead to a severe deterioration in model accuracy, abnormal dialogue output, or other unexpected behavior. Users shall assume all risks associated with such independent use.**

## Function Description

**Note: The algorithm implementation includes a training process that requires a minimum amount of NPU memory. Only devices with NPU memory of 64 GB or more are supported.**

### YAML Configuration Example

The following example shows a YAML configuration when the algorithm is used as a processor.

```yaml
# AutoRound supports mixed quantization, which allows you to apply different quantization configurations to different layers. This example shows a mixed configuration of W8A8 and W4A4.
# W8A8 dynamic quantization configuration
default_w8a8_dynamic: &default_w8a8_dynamic
  weight:
    scope: "per_group"        # Specifies the weight quantization scope.
    dtype: "int8"             # Specifies the weight quantization data type.
    symmetric: True           # Specifies whether to enable symmetric quantization.
    method: "autoround"       # Specifies the weight quantization method. This example uses the AutoRound algorithm, which includes parameter training.
    ext:
      group_size: 256         # Specifies the quantization group size. Grouping is performed on the input_features dimension of the target nn.Linear. This value must be a divisor of the input_features dimension.
      scale_dtype: "bfloat16" # Specifies the scaling factor data type.
  act:
    scope: "per_token"        # Specifies the activation quantization scope.
    dtype: "int8"             # Specifies the activation quantization data type.
    symmetric: True           # Specifies whether to enable symmetric quantization.
    method: "minmax"          # Specifies the activation quantization method. The MinMax algorithm is used in this example.

# W4A4 dynamic quantization configuration template
default_w4a4_dynamic: &default_w4a4_dynamic
  weight:
    scope: "per_group"
    dtype: "int4"
    symmetric: True
    method: "autoround"
    ext:
      group_size: 256
      scale_dtype: "bfloat16"
  act:
    scope: "per_token"
    dtype: "int4"
    symmetric: True
    method: "minmax"


spec:
  process:
    - type: "autoround_quant"     # Specifies the processor type. The value is fixed to autoround_quant.
      iters: 400                  # Specifies the number of optimization iterations.
      enable_minmax_tuning: True # Specifies whether to enable MinMax tuning.
      enable_round_tuning: True   # Specifies whether to enable rounding tuning.
      strategies:
        # Strategy 1: Apply W8A8 quantization to all layers except the up_proj, gate_proj, and o_proj layers.
        - qconfig: *default_w8a8_dynamic
          exclude:
            - "*.up_proj"
            - "*.gate_proj"
            - "*.o_proj"
        # Strategy 2: Apply W4A4 quantization to the up_proj, gate_proj, and o_proj layers.
        - qconfig: *default_w4a4_dynamic
          include:
            - "*.up_proj"
            - "*.gate_proj"
            - "*.o_proj"

```

### YAML Configuration Fields

| Field| Purpose| Type| Description| Default Value|
|--------|------|------|------|--------|
| type | Specifies the processor type identifier.| `string` | The value is fixed to `autoround_quant`, which identifies the object as the AutoRound quantization processor.| `"autoround_quant"` |
| iters | Specifies the number of optimization iterations.| `int` | The number of iterations must be greater than 0, as it affects both optimization quality and computation time.| `10` |
| enable_minmax_tuning | Specifies whether to enable MinMax tuning.| `bool` | `True` enables MinMax tuning, while `False` disables it.| `True` |
| enable_round_tuning | Specifies whether to enable rounding tuning.| `bool` | `True` enables rounding tuning, while `False` disables it.| `True` |
| strategies | Specifies the quantization strategy configuration.| `array[object]` | This field supports mixed-precision quantization using INT4 and INT8.| See the [strategies-Quantization Strategy Configuration](#strategies---quantization-strategy-configuration) section below.|

#### `strategies` - Quantization Strategy Configuration

**Purpose**: Configures quantization strategies for different layers and supports mixed quantization.

| Field| Purpose| Type| Description| Example Value|
|--------|------|------|------|--------|
| qconfig | Specifies the quantization configuration parameters.| `object` | This section contains parameters for activation quantization and weight quantization.| [act](#qconfigact---activation-quantization-configuration) and [weight](#qconfigweight---weight-quantization-configuration)|
| include | Specifies the layers to be included.| `array[string]` | Wildcard matching is supported to specify layers for quantization.| `["*"]`, `["*self_attn*"]` |
| exclude | Specifies the layers to be excluded.| `array[string]` | Wildcard matching is supported, and this field has a higher priority than `include`.| `["*down_proj*"]` |

#### `qconfig.act` - Activation Quantization Configuration

**Purpose**: Configures quantization parameters for activation values.

| Parameter| Purpose| Value| Description| Default Value|
|--------|------|--------|------|--------|
| scope | Specifies the quantization scope.| `"per_token"` | `per_token` provides independent parameters for each token through dynamic quantization. AutoRound currently supports only `per_token`.| `"per_token"` |
| dtype | Specifies the quantization data type.| `"int8"`, `"int4"` | This parameter specifies 8-bit or 4-bit integer quantization.| `"int8"` |
| symmetric | Specifies whether to enable symmetric quantization.| `True` | `True` enables symmetric quantization, with a zero point of `0`. AutoRound activation quantization supports only symmetric quantization.| `True` |
| method | Specifies the quantization method.| `"minmax"` | `minmax` indicates the MinMax activation quantization algorithm.| `"minmax"` |

#### `qconfig.weight` - Weight Quantization Configuration

**Purpose**: Configures quantization parameters for weights.

| Parameter| Purpose| Value| Description| Default Value|
|--------|------|--------|------|--------|
| scope | Specifies the quantization scope.| `"per_channel"`, `"per_group"` | `per_channel`: uses independent parameters for each channel.<br>`per_group`: uses independent parameters for each group. AutoRound weight quantization does not support `per_tensor`.| `"per_group"` |
| dtype | Specifies the quantization data type.| `"int8"`, `"int4"` | This parameter specifies 8-bit or 4-bit integer quantization.| `"int8"` |
| symmetric | Specifies whether to enable symmetric quantization.| `True`, `False` | `True`: enables symmetric quantization, with a zero point of `0`.<br>`False`: enables asymmetric quantization, with an adjustable zero point.| `True` |
| method | Specifies the quantization method.| `"autoround"` | `autoround` indicates the AutoRound algorithm, which includes parameter training.| `"autoround"` |
| ext | Specifies the extended configuration.| `object` | This section contains AutoRound-specific configuration parameters.| See the [`ext` - AutoRound Extended Configuration](#ext---autoround-extended-configuration) section below.|

#### `ext` - AutoRound Extended Configuration

**Purpose**: Configures AutoRound-specific configuration parameters.

| Parameter| Purpose| Type| Description| Example Value|
|--------|------|------|------|--------|
| group_size | Specifies the quantization group size.| `int` | The quantization group size must be a divisor of the `input_features` dimension of the `nn.Linear` layer.| `256` |
| scale_dtype | Specifies the data type of the scaling factor.| `string` | The data type of the scaling factor affects both accuracy and memory usage. Valid values: `"float16"`, `"float32"`, and `"bfloat16"`| `"bfloat16"` |

### Layer Filtering Mechanism

The layer filtering mechanism specifies the layers to be quantized through `include` and `exclude` pattern matching. For details about filtering rules, matching modes, examples, and common layer naming patterns, see [Layer Filtering Mechanism](linear_quant.md#layer-filtering-mechanism) in *Linear Quantization Algorithm*.

## FAQ

### Optimization Does Not Converge

**Symptom**: During the optimization process, the gap between the floating-point output and the quantized output fluctuates significantly or fails to converge.

**Solution**: Adjust the learning rate or increase the number of iterations.

### Significant Accuracy Drop

**Symptom**: The accuracy of the quantized model drops more than expected.

**Solution**: Increase the number of optimization steps, adjust the quantization configuration, reduce the number of layers using `W4A4` quantization, or use a larger volume of high-quality calibration data.

### Incorrect `group_size` Configuration

**Symptom**: A shape-related exception is thrown during quantization, such as: "shape '[-1, 257]' is invalid for input of size 512".

**Cause**: The `group_size` parameter must be a divisor of the `input_features` dimension of the target `nn.Linear` layer. Otherwise, group-wise quantization will fail.

**Solution**:

  - Check the `input_features` dimension of each layer in the model and ensure that the `group_size` is a divisor of that dimension.
  - Common `input_features` dimensions include `4096`, `8192`, and `11008`.
  - The recommended `group_size` values are `128`, `256`, and `512`, which are typically divisors of the `input_features` for most layers.

### Layer Matching Alarms

**Symptom**: During quantization, the tool reports a layer matching alarm.

**Solution**: Check the model definition to ensure that the `include` and `exclude` patterns match the correct layers. If the `include` and `exclude` patterns do not match any layer, the tool reports an alarm. For details about common matching failure causes and troubleshooting procedures, see [Layer Matching Alarms](linear_quant.md#layer-matching-alarms) in *Linear Quantization Algorithm*.
