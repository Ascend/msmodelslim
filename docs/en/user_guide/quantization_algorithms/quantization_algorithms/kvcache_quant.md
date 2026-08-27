# KVCache Quantization: Cache Quantization Algorithm

## Overview

- **Introduction**: This topic describes the KVCache quantization mechanism. KVCache quantization is typically used in combination with [linear quantization](linear_quant.md) to implement a full quantization scheme.
- **Problem**: In large model inference, the key and value states stored in the KVCache occupy a significant amount of GPU memory. This memory usage increases linearly with sequence length, creating an inference bottleneck.
- **Objective**: Quantize the `key_states` and `value_states` written to the KVCache to significantly reduce cache memory usage while maintaining generation quality.

## Preparations

Install msModelSlim. For details, see [msModelSlim Installation Guide](../../../install_guide/install_guide.md).

## Principle and Implementation

### Principle

**Core Concepts**

1. **Quantization objective**: Perform INT8 quantization on the `key_states` and `value_states` written to the KVCache within the attention mechanism.
2. **Quantization timing**: Intercept the key and value states to apply quantization calibration when `DynamicCache.update()` is called.
3. **Quantization strategy**: `per_channel`, which calculates quantization parameters based on the hidden layer dimension to balance accuracy and efficiency.
4. **Memory optimization**: Theoretically, this algorithm can reduce cache memory usage by approximately 50% after quantization (from FP16 to INT8).

### Implementation

#### Code Implementation

- The algorithm is implemented in [msmodelslim/processor/quant/attention.py](../../../../../msmodelslim/processor/quant/attention.py), and the processing flow consists of three phases:

#### Detection phase

  - Phase: `pre_run`.
  - Automatically detect attention layers in the model and identify `self_attn` modules based on module naming rules.
  - Create a `DynamicCacheQuantizer` for each attention layer and configure the quantization parameters.
  - Install a trigger hook on the first target attention layer to detect the start of inference.

#### Calibration phase

  - Phase: `run`.
  - Use the hook mechanism to intercept key and value states when `DynamicCache.update()` is called.
  - Use `DynamicCacheQuantizer` to perform fake-quantization on cache states and collect quantization statistics.
  - Incremental calibration is supported to adapt to dynamic sequence length changes.

#### Fake-quantization deployment phase

  - Phase: `postprocess`.
  - Convert the calibrated quantizer into an inference-optimized `FakeQuantDynamicCache` intermediate representation (IR).
  - Maintain compatibility with the original cache mechanism without modifying upper-layer inference logic.

### Core Components

#### Quantizer Implementation

```python
# DynamicCacheQuantizer: quantizer in the calibration phase
class DynamicCacheQuantizer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Transposition: (batch, heads, seq, head_dim) -> (batch, seq, heads, head_dim)
        x = x.transpose(-2, -3)
        # 2. Reshape: Adapt to the input format of the quantization algorithm.
        x = x.reshape(-1, x.shape[-1] * x.shape[-2])
        # 3. Quantization: Apply fake-quantization to collect statistics.
        x = self.input_quantizer(x)
        # 4. Restoration: Restore to the original shape.
        return x.transpose(-2, -3)

# FakeQuantDynamicCache: Fake-quantization IR in the deployment phase
class FakeQuantDynamicCache(AutoFakeQuantDynamicCache):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Transposition: (batch, heads, seq, head_dim) -> (batch, seq, heads, head_dim)
        x = x.transpose(-2, -3)
        x_shape = x.shape
        # 2. Reshape: Adapt to the input format of the quantization algorithm.
        x = x.reshape(-1, x.shape[-1] * x.shape[-2])
        # 3. Quantization: Perform fake-quantization by using fixed quantization parameters.
        x_q_dq = fake_quantize(QStorage(QDType.FLOAT, x), self.x_q_param).value
        # 4. Restoration: Restore to the original shape.
        x_q_dq = x_q_dq.reshape(x_shape)
        x_q_dq = x_q_dq.transpose(-2, -3)
        return x_q_dq
```

## Application Requirements

- **Model structure constraints**
  - The `forward` function of the attention module must accept a `DynamicCache` object and call `cache.update()`.
  - The `layer_idx` parameter must be correctly passed to the update API to distinguish quantizers of different layers.
  - Currently, module class names are used for pattern matching (`*self_attn*`). Custom naming must be adapted to this pattern.

- **Quantization mode constraints**
  - Currently, only INT8 quantization is supported.
  - Only the KVCache states are quantized. `query_states` and attention weights are not affected.

- **Memory management constraints**
  - Original-precision memory is still required during the fake-quantization phase. Actual memory savings require support from underlying operators.
  - Quantization parameters themselves occupy a small amount of additional memory.

- **Cache type constraints**
  - **DynamicCache**: Transformers standard dynamic cache, which is fully supported.
  - **Custom cache**: The `update(key_states, value_states, layer_idx)` API must be implemented in the custom model as follows:
  1. **API requirements**

      ```python
      class CustomCache:
          def update(self, key_states: torch.Tensor, value_states: torch.Tensor,
                      layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
              # Returns the updated key_states and value_states.
              pass
      ```

  2. **Hook registration**

     - The cache object must be passed as a parameter to the `forward` method of the attention module.
     - The system automatically detects the `cache.update` call and registers quantization hooks.

  3. **Parameter passing**

     - The attention module must use the `layer_idx` parameter to indicate the current layer index.
     - Nested calls and recursive quantization are supported.

## Function Description

### Supported Models

- Qwen2.5 series
- Qwen3 series

### YAML Configuration Example

The following example shows a YAML configuration when the algorithm is used as a processor.

```yaml
- type: "dynamic_cache" # Specifies the processor type. The value is fixed to dynamic_cache.
  qconfig: # Specifies the quantization configuration. per_channel quantization is supported.
    scope: "per_channel" # Specifies the quantization granularity. Only per_channel is supported.
    dtype: "int8" # Specifies the quantization data type. Currently only int8 is supported.
    symmetric: True # Specifies whether to enable symmetric quantization. Default value: True.
    method: "minmax" # Specifies the quantization method. Currently only minmax is supported.
  include: # Specifies name matching patterns for attention modules to be included. Wildcard character * is supported. By default, all attention modules are included.
    - "*"
  exclude: # Specifies name matching patterns for attention modules to be excluded. Wildcard character * is supported. By default, this parameter is left unspecified.
    - "model.layers.0.self_attn"
```

### YAML Configuration Fields

| Field| Purpose| Data Type| Default Value| Description|
|--------|------|----------|--------|------|
| type | Specifies the processor type identifier.| string | - | The value is fixed to `dynamic_cache`, which identifies the object as the KVCache quantization processor.|
| qconfig | Specifies the KVCache quantization configuration.| object | - | This section contains parameters for KVCache quantization.|
| scope | Specifies the quantization granularity.| string | "per_channel" | Currently, only `per_channel` is supported, which specifies that quantization parameters are calculated by hidden layer dimension.|
| dtype | Specifies the quantization data type.| string | "int8" | Data type after quantization. Currently, only `int8` is supported.|
| symmetric | Specifies whether to enable symmetric quantization.| boolean | true | Specifies whether to enable symmetric quantization. `true` enables symmetric quantization, while `false` enables asymmetric quantization.|
| method | Specifies the quantization method.| string | "minmax" | Currently only the `minmax` algorithm is supported.|
| include | Specifies attention layers included.| array[string] | ["*"] | Wildcard matching is supported to specify the attention layers for KVCache quantization.|
| exclude | Specifies attention layers excluded.| array[string] | [] | Wildcard matching is supported, and this field has a higher priority than `include`.|

## FAQ

### **Cache not Quantized**

**Symptom**: The cache is not quantized.

**Solution**: Ensure that the `forward` method of the attention module takes a `cache` parameter and that `cache.update()` is correctly called.

### New Cache Type Not Supported

**Symptom**: The new cache type is not supported.

**Solution**: Ensure that the custom cache implements the standard `update` API and correctly processes the return value.
