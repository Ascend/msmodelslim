# FA3 Quantization: Flash Attention 3 Activation Quantization Algorithm

## Overview

- **Background**: In long sequences, the intermediate activation tensors `Q`, `K`, and `V` of the attention mechanism occupy a large proportion of the GPU memory. Quantizing these tensors can effectively reduce GPU memory usage and improve computational efficiency. However, the activation dynamic ranges of `Q`, `K`, and `V` are large and unevenly distributed. Performing direct global quantization may cause a significant accuracy drop.
- **Core idea**: Flash Attention 3 (FA3) is a per-head quantization algorithm for the activations of the attention mechanism. It performs per-head INT8 quantization on the `Q`, `K`, and `V` activations to improve inference performance and reduce GPU memory usage while maintaining model accuracy. FA3 quantization is typically used together with [linear quantization](linear_quant.md) to implement a full quantization scheme.

## Preparations

Install msModelSlim. For details, see [msModelSlim Installation Guide](../../getting_started/install_guide.md).

## Principle and Implementation

### Principle

**Core Concepts**

1. **Quantization objective**: Perform `per-head` INT8 quantization on the `Q`, `K`, and `V` activation values within the attention mechanism.
2. **Quantization granularity**: `per-head`. This granularity indicates that quantization parameters are calculated independently for each attention head to adapt to the activation distribution differences across different heads.
3. **Quantization timing**: Insert quantization nodes at the critical positions of Multi-head Latent Attention (MLA) computation.
4. **Quantization strategy**: `per-head`, which improves accuracy by using independent quantization parameters calculated for each attention head.

**Algorithm Process**

1. Collect activation statistics for each head.
   - Input: activation tensor x, with shape (B, H, S, D),
     where B = batch_size, H = num_heads, S = seq_len, and D = head_dim.
   - Reshape x to `(H, N)`, where `N = B * S * D`.
   - Collect N data points for each head independently.

2. Identify the minimum quantization range for each head by using the Recall Window algorithm.
   - Input: head_data (N,), ratio (default: 0.9999)
   - Sort the N data points: sorted_data = sort(head_data).
   - Calculate the number of target elements: target_num = int(ratio * N).
   - Search for the minimum range by using a sliding window.
     * Traverse all possible window start points i from 0 to (N – target_num).
     * Window range: [sorted_data[i], sorted_data[i + target_num – 1]]
     * Calculate the window length: window_length = sorted_data[i + target_num – 1] – sorted_data[i].
     * Retain the window with the smallest window length.
   - Output: (min_val, max_val) for the head

3. Perform cross-batch accumulation of statistics.
   - Calculate the minimum and maximum values (min_val, max_val) for each calibration batch.
   - Update the accumulated statistics to ensure the quantization range covers all calibration data.
     * min_values[h] = min(min_values[h], current_min[h])
     * max_values[h] = max(max_values[h], current_max[h])

4. Calculate quantization parameters for each head.
   - Symmetric quantization formula:
     * abs_max[h] = max(abs(min_values[h]), abs(max_values[h]))
     * scale[h] = abs_max[h] / 127
   - Output: quantization parameter q_param

### Implementation

#### Code Implementation

- FA3 quantization is implemented in [processor.py](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/processor/quant/fa3/processor.py). The processing flow consists of three phases.

#### Injection phase

  - Phase: `preprocess`.
  - Call the `inject_fa3_placeholders()` method of the model adapter.
  - The adapter inserts the placeholder `FA3QuantPlaceHolder` at key positions in the MLA computation process.
  - Selective injection is supported by using `include/exclude` configurations.

#### Calibration phase

  - Phase: `process`.
  - The placeholder is replaced with the `_FA3PerheadObserver` listener.
  - When calibration data flows through the attention layer, the listener collects activation statistics for each head.
  - The algorithm identifies the minimum numerical distribution interval containing a specified data proportion by using a sliding window approach.

#### Fake-quantization deployment phase

  - Phase: `postprocess`.
  - Extract the `min` and `max` values for each head from the listener.
  - Call `calculate_qparam()` to calculate the symmetric quantization parameters.
  - Create an intermediate representation (IR) replacement listener.

## Application Requirements

- **Model structure requirements**
  - A model adapter that supports FA3 must implement `FA3QuantAdapterInterface`.
  - The requirements are applicable to the MLA-based attention mechanism.
  - The activation computation paths for `Q`, `K`, and `V` must be clearly defined to insert quantization nodes.

- **Quantization mode constraints**
  - Currently, only INT8 symmetric quantization is supported.
  - Quantization parameters are fixed after calibration. Dynamic adjustment is not supported.

## Function Description

### Supported Models

- DeepSeek-R1-0528
- DeepSeek-V3.1

### YAML Configuration Example

The following example shows a YAML configuration when the algorithm is used as a processor:

```yaml
spec:
  process:
    - type: "fa3_quant"
      include: [ "*" ]                           # Specifies the attention layers to be included for quantization.
      exclude: [ "model.layers.0.self_attn" ]   # Specifies the attention layers to be excluded from quantization.
```

### YAML Configuration Fields

| Field | Purpose              | Data Type     | Default Value| Description                                              |
| ------- | ------------------ | ------------- | ------ | -------------------------------------------------- |
| type    | Specifies the processor type identifier.    | string        | -      | The value is fixed to `fa3_quant`, which identifies the object as the FA3 quantization processor.|
| include | Specifies attention layers included.| array[string] | ["*"]  | Wildcard matching is supported to specify the attention layers for FA3 quantization.     |
| exclude | Specifies attention layers excluded.| array[string] | []     | Wildcard matching is supported, and this field has a higher priority than `include`.              |

## Model Adaptation

### Interface and Data Structure

Currently, DeepSeek-V3 series models are supported. For other MLA-based models, the corresponding adapters are required.

```python
# Model adaptation for the FA3 quantization interface
class ModelAdapter(FA3QuantAdapterInterface):
    def inject_fa3_placeholders(
            self,
            root_name: str,
            root_module: nn.Module,
            should_inject: Callable[[str], bool],
    ) -> None: ...
```

### Adaptation Procedure

- **Prerequisites**
    - The model is based on the Transformer architecture and contains explicit attention layers.
    - The `Q`, `K`, and `V` activation values of the attention layers can be located within the computation process.
    - The adapter can access the attention module of the model and modify the `forward` method.

- **Procedure**
  For details, see the implementation of [model_adapter.py](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/model/deepseek_v3/model_adapter.py) for DeepSeek.
    1. Inherit the `FA3QuantAdapterInterface` for the model adapter.
    2. Traverse the model and selectively inject the `FA3QuantPlaceHolder` as a submodule within the attention layer by using `should_inject`.
    3. Locate the critical position where the `Q`, `K`, and `V` activation values flow into attention computation. This position indicates the node where FA3 quantization must be inserted.
    4. Wrap the `forward` method of the attention layer and insert the call to FA3 quantization at the identified critical position.
