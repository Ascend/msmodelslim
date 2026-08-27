# SSZ: Weight Quantization Algorithm

## Overview

- **Problem**: Traditional quantization methods (such as MinMax) result in large quantization error when the weight distribution is non-uniform, which affects model accuracy.
- **Objective**: Minimize quantization error by iteratively searching for the optimal scaling factor (scale) and offset, which improves the accuracy of the quantized model.

## Preparations

Install msModelSlim. For details, see [msModelSlim Installation Guide](../../../install_guide/install_guide.md).

## Principle and Implementation

### Principle

**Core Concepts**

1. **Iterative optimization**: The algorithm optimizes quantization parameters step by step through multiple iterations.
2. **Least squares method**: The algorithm uses the least squares method to calculate the optimal scale and offset for the current iteration.
3. **Greedy update**: The algorithm retains only parameters that minimize quantization error.
4. **Convergence determination**: The algorithm determines convergence by monitoring changes in both relative and absolute errors.

**Algorithm Processes**

1. Initialize the quantization parameters `scale` and `offset` through a `MinMax` observer.
2. Calculate the optimal `scale` and `offset` by using the least squares method.
3. Compare the quantization error of the old and new parameters and retain the superior version.
4. Repeat steps 2 and 3 until convergence occurs or the maximum number of iterations is reached to obtain the final quantization parameters.

### Implementation

#### Code Implementation

The algorithm is implemented in [msmodelslim/core/quantizer/impl/ssz.py](../../../../../msmodelslim/core/quantizer/impl/ssz.py), and the core function is `ssz_calculate_qparam`.

#### Initialization Phase

- The MinMax observer calculates the weight statistics, including the minimum and maximum values.
- The algorithm calculates the initial quantization parameters (`scale` and `offset`) based on these statistics.

#### Iterative Optimization Phase

- Symmetric quantization: The `offset` is fixed at `0`, and the algorithm optimizes only the `scale`.
- Asymmetric quantization: The algorithm optimizes both the `scale` and the `offset` simultaneously.
- The algorithm calculates the optimal parameters by using the least squares method.
- Greedy update strategy: The algorithm retains only parameters that minimize quantization error.

#### Convergence Determination Phase

- Relative error change: `(best_mse - current_mse) / best_mse < threshold`.
- Absolute error change: `|best_mse - current_mse| < threshold`.
- The process exits early when all channels meet the convergence conditions.

## Application Requirements

- **High accuracy requirement**: This algorithm is suitable for model quantization scenarios that require high accuracy.
- **Uneven weight distribution**: This algorithm is especially suitable for linear layers with uneven weight distribution.
- **Computing cost**: The SSZ algorithm requires multiple iterations, which may result in high computing cost in some scenarios.
- **Initialization dependency**: The algorithm requires the MinMax observer to calculate the initial quantization parameters.
- **Usage limitations**:
    - Currently, `per_channel` symmetric quantization is supported for INT8 and INT4 scenarios.
    - `per_channel` asymmetric quantization for INT4 scenarios is not supported, but support will be added in later versions.
    - The `per_tensor` and `per_group` quantization granularities are not supported, but support will be added in later versions.
    - The weights must be 2D tensors.

## Function Description

### YAML Configuration Example

The following example shows a YAML configuration when the algorithm is used as a processor:

```yaml
spec:
  process:
    - type: "linear_quant"
      qconfig:
        weight:
          scope: "per_channel"
          dtype: "int8"
          symmetric: true
          method: "ssz"
```

### YAML Configuration Fields

#### `qconfig.weight` - Weight Quantization Configuration

| Field| Purpose| Optional Value| Description| Default Value|
|--------|------|--------|------|--------|
| scope | Specifies the quantization scope.| `"per_channel"` | `per_channel`: uses independent parameters for each channel.| `"per_channel"` |
| dtype | Specifies the quantization data type.| `"int8"`, `"int4"` | This parameter specifies 8-bit or 4-bit integer quantization.| `"int8"` |
| symmetric | Specifies whether to enable symmetric quantization.| `true`, `false` | `true`: enables symmetric quantization, with a zero point of `0`.<br>`false`: enables asymmetric quantization, with an adjustable zero point.| `true` |
| method | Specifies the quantization method.| `"ssz"` | `ssz`: SSZ weight quantization.| `"ssz"` |

### Algorithm Parameters

The SSZ algorithm uses the following internal parameters (which can be adjusted by modifying the source code within [msmodelslim/core/quantizer/impl/ssz.py](../../../../../msmodelslim/core/quantizer/impl/ssz.py)):

```python
SCALE_SEARCH_ITER_NUM = 20                    # Specifies the maximum number of iterations.
SCALE_SEARCH_CONVERGE_THRESHOLD = 1e-10       # Specifies the threshold used to determine convergence.
SCALE_SEARCH_MIN_SCALE = 1e-5                 # Specifies the minimum scaling factor.
```

## FAQ

### Incorrect Weight Dimension

**Symptom**: The input weight dimension is incorrect, which causes the quantization process to fail.

**Solution**: Verify that the weight dimension is correct and ensure that the weight is a 2D tensor.

### Incorrect Quantization Configuration

**Symptom**: The quantization configuration is incorrect, which causes the quantization process to fail.

**Solution**: Verify that the `dtype`, `scope`, `method`, and `symmetric` parameters are set correctly.

### Incorrect Initialization Sequence

**Symptom**: The initialization sequence is incorrect, which causes the quantization process to fail.

**Solution**: Call `init_weight` before calling `forward`.

### Convergence Issues

**Symptom**: If the algorithm does not converge, the `SCALE_SEARCH_CONVERGE_THRESHOLD` parameter may require adjustment.

**Solution**: Adjust the `SCALE_SEARCH_CONVERGE_THRESHOLD` parameter.
