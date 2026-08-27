# KVSmooth: Outlier Suppression Algorithm for KVCache Quantization

## Overview

- **Problem**: In KVCache quantization, a small number of outliers of the key significantly increase the quantization scale. This leads to insufficient effective bits for most channels, which causes attention score degradation and generation quality deterioration.
- **Objective**: Compress the dynamic range of `K` to make it easier to quantize, while maintaining numerical stability and accuracy, without changing the expected value of the attention score `QK^T`.

## Preparations

Install msModelSlim. For details, see [msModelSlim Installation Guide](../../../install_guide/install_guide.md).

## Principle and Implementation

### Principle

**Core Logic**

- Smooth the activation values `key_states` of KVCache by using the method of fusing the scaling coefficient `s` into the Q/K projection or normalization weight before RoPE:
    - `K' = K / s`
    - `Q' = Q × s`
    - With `Q'K'^T = QK^T`, the attention score remains unchanged, while the dynamic range of `K` is compressed, and the quantization is more robust.
- Outliers are migrated from `key_states` to `query_states`. During inference, only `key_states` written to the KVCache are quantized, while `query_states` are not.
  This migration is acceptable and does not introduce additional quantization error.
- RoPE rotates channels in pairs, and the channel dimensions have a pairwise relationship. The algorithm first takes the maximum between paired channels, and then restores the paired structure for scaling.

### Implementation

#### Code Implementation

The algorithm is implemented in [msmodelslim/processor/kv_smooth](../../../../../msmodelslim/processor/kv_smooth/), and the processing flow consists of two phases.

#### Observation phase

  - Phase: `preprocess`.
  - Encapsulate `past_key_values` by using the method of injecting an observer, and capture `key_states` when the attention module calls `Cache.update()`.
  - Aggregate min/max in the `[batch, seq]` dimension by using the observer to obtain the maximum absolute value of each channel at each layer, which is used as the statistical basis for scaling.

#### Smoothing phase

  - Phase: `postprocess`.
  - Calculate the scaling vector based on the maximum value of `|key_states|`, and rewrites the `weight` (and optional `bias`) of the corresponding module located before RoPE based on the fusion method, so that `key_states` written to the KVCache after RoPE are smoothed. Meanwhile, `query_states` are amplified accordingly.
    - `state-rope-linear`: Fold the scaling into `k_proj`/`q_proj` along the path of `Linear → RoPE → KVCache`.
    - `state-rope-norm`: Fold the scaling into `k_norm`/`q_norm` along the path of `Norm → RoPE → KVCache`.

## Application Requirements

- **Data dependency on the calibration dataset**: Observe the suppression scaling factor by using inference calibration. If the data distribution of the calibration dataset deviates from the actual service, the effect will be affected.
- **Model implementation constraints**: The attention forward process must accept and use `past_key_values` or `past_key_value`. Otherwise, the suppression scaling factor cannot be observed.
- **Fusion point constraints**: Currently, fusion is supported for two types of paths: `Linear/Norm → RoPE → KVCache`.
- **Fusion modules constraints**: The target `Linear` or `Norm` submodule must exist and have a writable `weight` (and optional `bias`). Other custom modules are not supported for now.
- **RoPE assumption**: By default, the algorithm performs reduction and restoration on paired channels by using RoPE. Evaluate and verify non-RoPE structures with caution.
- **Quantization method assumption**: The algorithm is based on the assumption of quantizing only `key_states` and `value_states` of the KVCache without quantizing `query_states`.
  When quantizing `query_states`, evaluate the applicability of this method with caution.

## Function Description

### YAML Configuration Example

The following example shows a YAML configuration when the algorithm is used as a processor:

```yaml
spec:
  process:
    - type: "kv_smooth"
      smooth_factor: 1.0                    # Specifies the degree of smoothing aggressiveness. The value must be greater than 0, and a larger value indicates more aggressive smoothing.
      include: ["*"]                        # Specifies the layers to be included. Wildcard matching is supported.
      exclude: ["model.layers.0.self_attn"] # Specifies the layers to be excluded. Wildcard matching is supported.
```

### YAML Configuration Fields

| Field            | Purpose       | Type       | Default Value        | Description             | Example                            |
|-----------------|-----------|-----------|-------------|-----------------|--------------------------------|
| `type`          | Specifies the processor type.  | str       | "kv_smooth" | The value is fixed to `kv_smooth`.| `"kv_smooth"`                  |
| `smooth_factor` | Specifies the degree of smoothing aggressiveness. | float     | 1.0         | The value must be greater than 0, and a larger value indicates more aggressive smoothing.    | `1.5`                          |
| `include`       | Specifies the modules to be included for smoothing.| List[str] | ["*"]       | Wildcard matching is supported.          | `["model.layers.*.self_attn"]` |
| `exclude`       | Specifies the modules to be excluded from smoothing.| List[str] | []          | Wildcard matching is supported.          | `["model.layers.0.self_attn"]` |

**Note**:

- `smooth_factor` must be greater than 0.
- `include` and `exclude` support wildcard matching, for example, `model.layers.*.self_attn`.
- `exclude` has higher priority than `include`. That is, if a module matches both `include` and `exclude`, the module will be excluded.

## Model Adaptation

### Interface and Data Structure

```python
# Fusion mode enumeration
class KVSmoothFusedType(Enum):
    StateViaRopeToNorm = 'state-rope-norm'  # Supports the fusion of key_states/query_states → Norm.
    StateViaRopeToLinear = 'state-rope-linear'  # Supports the fusion of key_states/query_states → Linear.


# Information about the KVSmooth unit, describing the model substructure and fusion mode
class KVSmoothFusedUnit(BaseModel):
    attention_name: str  # Specifies the full module name, such as "model.layers.0.self_attn".
    layer_idx: int  # Specifies the layer index, such as 0.
    fused_from_query_states_name: str  # Specifies the name of the module on the query_states branch before RoPE, such as "q_proj" or "q_norm".
    fused_from_key_states_name: str # Specifies the name of the module on the key_states branch before RoPE, such as "k_proj" or "k_norm".
    fused_type: KVSmoothFusedType  # Specifies the fusion type.


# Interface for adapting models to the KVSmooth algorithm
class KVSmoothFusedInterface(ABC):
    # Retrieve a list of all units in the model that can be processed by using KVSmooth.
    def get_kvsmooth_fused_subgraph(self) -> List[KVSmoothFusedUnit]: ...

    # Obtain the head_dim information.
    def get_head_dim(self) -> int: ...

    # Obtain the num_key_value_groups information.
    def get_num_key_value_groups(self) -> int: ...

    # Obtain the num_key_value_heads information.
    def get_num_key_value_heads(self) -> int: ...
```

### Adaptation Procedure

- **Prerequisites**
    - The attention forward process must accept `past_key_values` or `past_key_value` through `kwargs` and call `Cache.update()` internally. Otherwise, the observer cannot work.
    - The target path must comply with the `Linear/Norm → RoPE → KVCache` structure.
- **Procedure**
    1. Inherit the `KVSmoothFusedInterface` and implement all methods by using the model adapter. For reference, see
       [msmodelslim/model/qwen3/model_adapter.py](../../../../../msmodelslim/model/qwen3/model_adapter.py).
    2. In `get_kvsmooth_fused_subgraph()`, return `KVSmoothFusedUnit` for each layer and specify the following: parameters:
        - `attention_name`: specifies the full path (for example, `model.layers.{i}.self_attn`) that is consistent with `named_modules()`.
        - `layer_idx`: specifies the layer index used for `Cache.update()`.
        - `fused_from_query_states_name`: specifies the name of the `norm` or `linear` submodule on the `query_states` branch before RoPE, such as `q_proj`.
        - `fused_from_key_states_name`: specifies the name of the `norm` or `linear` submodule on the `key_states` branch before RoPE, for example, `k_proj`.
        - `fused_type`: enumerates the fusion modes. Valid values: `StateViaRopeToNorm` or `StateViaRopeToLinear`.
    3. Provide the global structure information of the model by using `get_head_dim()`, `get_num_key_value_heads()`, and `get_num_key_value_groups()`.

## FAQ

### Fallback Mismatch

**Symptom**: The alarm log contains the description `are not matched any module`.

**Solution**: Check the complete module names to confirm whether `include` or `exclude` is incorrectly set.

### Missing Header Dimension Information

**Symptom**: `UnsupportedError` is thrown, indicating that `get_head_dim`, `get_num_key_value_groups`, and `get_num_key_value_heads` are missing.

**Solution**: Ensure that the corresponding model adapter implements the `KVSmoothFusedInterface`. Otherwise, the model is not applicable to the algorithm.

### Attention Not Applicable

**Symptom**: The log contains the alarm `past_key_values and past_key_value both are None`.

**Solution**: Check the model file in `Transformers` to ensure that `past_key_values` and `past_key_value` are passed to the `forward` process of the `Attention` layer.
      Otherwise, the model is not applicable to the algorithm.

### Inconsistent Module Names

**Symptom**: `ToDoError` is thrown, indicating `has no submodule`.

**Solution**: Check the model adapter to ensure that the values of `fused_from_query_states_name` and `fused_from_key_states_name` are consistent with the actual names of the fused submodules.
