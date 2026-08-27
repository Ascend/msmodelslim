# MinMax Quantization Algorithm

## Overview

- MinMax is the most basic and commonly used quantization algorithm. It determines the quantization range by collecting statistics on the minimum and maximum values in the tensor (weights or activation values) to calculate the quantization scale and offset.
- **Core idea**: This algorithm linearly maps the original floating-point range to the target numerical range (such as the INT8 range of [-128, 127] or the FP8 representation range). This algorithm is simple and efficient with extremely low computational overhead, making it the primary choice for most common quantization scenarios.

## Preparations

Install msModelSlim. For details, see [msModelSlim Installation Guide](../../../install_guide/install_guide.md).

## Principles and Implementation

### Principles

The MinMax algorithm calculates quantization parameters based on the following formulas:

1. **Range determination**
   - $V_{min} = \min(X)$
   - $V_{max} = \max(X)$

2. **Scale calculation**
   - For asymmetric quantization: $S = \frac{V_{max} - V_{min}}{Q_{max} - Q_{min}}$
   - For symmetric quantization: $S = \frac{\max(|V_{min}|, |V_{max}|)}{Q_{max}}$

3. **Offset calculation**:
   - For asymmetric quantization: $Z = Q_{min} - \text{round}(\frac{V_{min}}{S})$
   - For symmetric quantization: $Z = 0$

$Q_{max}$ and $Q_{min}$ are the maximum and minimum values of the target data type. For example, $Q_{max}=127$ for INT8 symmetric quantization.For FP8 floating-point quantization, these values correspond to the representation range of the specific format.

### Implementation

The algorithm is implemented in [`msmodelslim/core/quantizer/impl/minmax.py`](../../../../../msmodelslim/core/quantizer/impl/minmax.py).

## Function Description

### YAML Configuration Example

MinMax is used as the activation or weight quantization method in the [`linear_quant`](linear_quant.md) processor and is specified by the `qconfig.act.method` or `qconfig.weight.method` field. Here is a YAML configuration example:

```yaml
spec:
  process:
    - type: "linear_quant"
      qconfig:
        act:
          scope: "per_tensor"   # Specifies the activation quantization scope.
          dtype: "int8"         # Specifies the activation quantization data type.
          symmetric: false      # Enables asymmetric activation quantization.
          method: "minmax"      # Use the MinMax algorithm for activation quantization.
        weight:
          scope: "per_channel"  # Specifies the weight quantization scope.
          dtype: "int8"         # Specifies the weight quantization data type.
          symmetric: true      # Enables symmetric quantization.
          method: "minmax"      # Use the MinMax algorithm for weight quantization.
      include: ["*"]            # Specifies the layers to be included.
      exclude: []               # Specifies the layers to be excluded.
```

### YAML Configuration Fields

The following fields are part of the `linear_quant` processor configuration. For a complete description of the `linear_quant` fields, see [Linear Quantization Configuration Fields](linear_quant.md#yaml-configuration-fields).

| Parameter| Configuration| Optional Value| Description|
|--------|----------|--------|------|
| `qconfig.act.method` | Activation quantization| `"minmax"` | Specify the MinMax algorithm for activation quantization.|
| `qconfig.weight.method` | Weight quantization| `"minmax"` | Specify the MinMax algorithm for weight quantization.|

## FAQ

### Accuracy Issues

**Symptom**: Accuracy of MinMax decreases significantly in low-bit (such as INT8 or INT4) scenarios.

**Solution**: The MinMax algorithm is highly sensitive to outliers. If a tensor contains even a few values with extreme magnitudes, MinMax expands the overall quantization range to cover these points. This expansion results in a loss of quantization precision for the majority of normal values. For low-bit or limited bit-width floating-point quantization, the system recommends using outlier suppression algorithms (such as SmoothQuant) or more advanced quantization algorithms (such as SSZ).
