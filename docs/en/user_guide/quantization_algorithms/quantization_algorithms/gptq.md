# GPTQ: Weight Quantization Algorithm

## Overview

- **Problem**: Traditional quantization methods (such as MinMax) result in large quantization error when the weight distribution is non-uniform, which affects model accuracy.
- **Objective**: To minimize the overall quantization error and improve model accuracy by using a column-wise optimization method. It compensates for the quantization error of the current column by adjusting the weights of subsequent unquantized columns.

## Preparations

Install msModelSlim. For details, see [msModelSlim Installation Guide](../../../install_guide/install_guide.md).

## Principle and Implementation

### Principle

Weights are quantized layer by layer and column by column. Unquantized weights are compensated based on the quantization error and the Hessian matrix of activation values to minimize the overall quantization error of the weights.

**Core Concepts**

1. **Layer-wise quantization**: Quantize each layer of the model independently to avoid error accumulation.
2. **Error correction based on second-order information**: Use the Hessian matrix to evaluate the impact of weight quantization on the output, and dynamically adjust unquantized weights to compensate for the error.
3. **Block-wise quantization**: Divide weights into multiple blocks to reduce computational complexity.
4. **Lazy batch update**: Delay the update of unquantized weights and combine multiple updates to reduce computational overhead.

### Implementation

The algorithm is implemented in [`msmodelslim/core/quantizer/impl/gptq.py`](../../../../../msmodelslim/core/quantizer/impl/gptq.py):

  - `per_channel` implementation class: `WeightPerChannelGPTQ`
  - `per_group` implementation class: `WeightPerGroupGPTQ`

## Application Requirements

- **High accuracy requirement**: This algorithm is suitable for model quantization scenarios that require high accuracy.
- **Computing cost**: The GPTQ algorithm computes the Hessian matrix and matrix factorization for activation values. It also quantizes weights column by column. Therefore, the algorithm features slow quantization speed and high computing cost.
- **Usage limitations**:
    - Currently, the algorithm supports symmetric and asymmetric `per_channel` and `per_group` quantization for INT8 scenarios.
    - Currently, the algorithm does not support symmetric and asymmetric `per_channel` and `per_group` quantization for INT4 scenarios (support will be added in the future).
    - Because GPTQ depends on model activation values, the calibration dataset must cover all experts in the MoE model quantization scenario. Due to this high requirement for the calibration dataset, GPTQ is not recommended for this scenario.
    - `per_tensor` quantization granularity is currently not supported.
    - Weights must be 2D tensors.

## Function Description

### YAML Configuration Example

The following example shows a YAML configuration when the algorithm is used as a processor.

`per_channel` quantization example

```yaml
spec:
  process:
    - type: "linear_quant"
      qconfig:
        weight:
          scope: "per_channel"   # Specifies the quantization scope.
          dtype: "int8"          # Specifies the quantization data type.
          symmetric: true        # Specifies whether to use symmetric quantization.
          method: "gptq"         # Specifies the quantization algorithm: GPTQ
          ext: # Optional. Specifies extended parameters.
            percdamp: 0.01       # Optional. Specifies the damping coefficient. Default value: 0.01.
            block_size: 128      # Optional. Specifies the block size. Default value: 128.
```

`per_group` quantization example

```yaml
spec:
  process:
    - type: "linear_quant"
      qconfig:
        weight:
          scope: "per_group"   # Specifies the quantization scope.
          dtype: "int8"        # Specifies the quantization data type.
          symmetric: true      # Specifies whether to enable symmetric quantization.
          method: "gptq"       # Specifies the quantization algorithm: GPTQ.
          ext: # Optional. Specifies extended parameters.
            percdamp: 0.01     # Optional. Specifies the damping coefficient. The default value is 0.01.
            block_size: 128    # Optional. Specifies the block size. The default value is 128.
            group_size: 128    # Optional. Specifies the group size. The default value is 128.
```

### YAML Configuration Fields

#### `qconfig.weight` - Weight Quantization Configuration

| Field      | Purpose    | Optional Value                           | Description                                           | Default Value                        |
|-----------|--------|--------------------------------|-----------------------------------------------|-----------------------------|
| scope     | Specifies the quantization scope.  | `"per_channel"`, `"per_group"` | `per_channel`: independent parameters for each channel.<br>`per_group`: independent parameters for each group.| `"per_channel"`             |
| dtype     | Specifies the quantization data type.| `"int8"`                       | 8-bit integer quantization.                                       | `"int8"`                    |
| symmetric | Specifies whether to enable symmetric quantization.| `true`, `false`                | `true`: enables symmetric quantization, with a zero point of `0`.<br>`false`: enables asymmetric quantization, with an adjustable zero point.       | `true`                      |
| method    | Specifies the quantization algorithm.  | `"gptq"`                       | `gptq`: enables GPTQ weight quantization.                               | `"gptq"`                    |
| ext       | Specifies the extended configuration.  | `object`                       | This section contains GPTQ-specific configuration parameters.                                | For details, see the **`ext` (GPTQ Extended Configuration)** section below.|

**`ext` (GPTQ Extended Configuration)**

**Purpose**: Configures GPTQ-specific configuration parameters.

| Field       | Purpose    | Type     | Description                                         | Example Value      |
|------------|--------|---------|---------------------------------------------|-----------|
| percdamp   | Specifies the damping coefficient.  | `float` | `percdamp` smooths gradient updates and reduces the impact of quantization noise on training.          | Default value: `0.01`|
| block_size | Specifies the iteration block size.| `int`   | The iteration block size must be a divisor of the `out_features` dimension of the `nn.Linear` layer to be quantized.  | Default value: `128` |
| group_size | Specifies the quantization group size.| `int`   | The quantization group size must be a divisor of the `input_features` dimension of the `nn.Linear` layer to be quantized.| Default value: <idp:inline displayname="code" id="code42048012483">128</idp:inline> |

## FAQ

### What Are the Meanings and Purposes of the Three Hyperparameters `percdamp`, `block_size`, and `group_size` in the GPTQ Algorithm?

- **`percdamp`**
  - **Meaning**: Damping percentage used to add a small diagonal damping to the computation of the inverse of the Hessian matrix to prevent numerical instability.
  - **Purpose**: GPTQ must solve a system of linear equations when updating weights. This process involves computing the inverse of the Hessian matrix. When the Hessian matrix is close to singular, computing the inverse directly causes numerical explosion.percdamp
    A small value (typically `max(diag(H)) * percdamp`) is added to the diagonal of the Hessian matrix to improve the condition number and stabilize the computation of the inverse.
  - **Typical value**: `0.01` (1% damping)
  - **Impact**: An excessively large value may lead to loss of accuracy, while an excessively small value may cause numerical instability.
- **`block_size`**
  - **Meaning**: Block size, which refers to the number of columns (the column blocks of the weight matrix) processed simultaneously in one iteration of GPTQ.
  - **Purpose**: GPTQ performs quantization layer by layer and block by block. `block_size` determines the size of the column group for computing the inverse of the Hessian matrix and updating weights.
     Larger blocks can improve the efficiency of parallel computation but consume more GPU memory, while smaller blocks may slow down the speed.
  - **Typical value**: `128`
  - **Impact**: `block_size` affects the quantization speed and memory usage, but it has little impact on the final quantization accuracy.
- **`group_size`**
  - **Description**: Group size used for group-wise quantization.
  - **Purpose**: In GPTQ, weights can share quantization parameters (`scale` and `zero_point`) by group. `group_size` specifies the number of consecutive elements in each group. For example,
    `group_size=128` indicates that one set of quantization parameters is used for every 128 weights. A smaller `group_size` can better adapt to the local distribution of weights and improve quantization accuracy. However, it requires storing more scaling factors, which increases the size of the model.
  - **Typical value**: `128`
  - **Impact**: A smaller `group_size` results in higher quantization accuracy but larger model file size. Generally, `128` is a good balance between accuracy and compression ratio.
