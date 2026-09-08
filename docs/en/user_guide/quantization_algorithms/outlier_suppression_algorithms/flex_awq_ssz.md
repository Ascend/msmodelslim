# Flex AWQ SSZ: Flexible Activation-Aware Weight Quantization Smoothing Algorithm

## Overview

- **Introduction**: The flexible activation-aware weight quantization smoothing (Flex AWQ SSZ) algorithm is designed to suppress activation outliers during the quantization of large language models (LLMs). This algorithm combines the concepts of activation-aware weight quantization (AWQ) and smooth scale zero (SSZ). By using an actual quantizer to evaluate parameter effectiveness, this algorithm automatically searches for the optimal `alpha` parameter to effectively reduce quantization error while maintaining model accuracy. Unlike traditional smoothing algorithms, Flex AWQ SSZ uses a physical quantizer for parameter evaluation, which more accurately reflects the actual effects of quantization.
- **Core idea**: Flex AWQ SSZ evaluates quantization error across different alpha parameters by using an actual quantizer (`LinearQuantizer`), automatically searching for the optimal `alpha` parameter that minimizes quantization error. The algorithm fixes `beta` at `0` and uses the `mean` value instead of the `max` value to calculate the activation scale, which balances accuracy and quantization efficiency across various scenarios.

## Preparations

Install msModelSlim. For details, see [msModelSlim Installation Guide](../../../install_guide/install_guide.md).

## Principle and Implementation

### Principle

**Core Logic**

- Calculate the activation scaling factor by using the mean of activation values: `Act_Mean_Abs = mean(abs(act))`.
- Evaluate the quantization error under various `alpha` parameters by using the actual quantizer (`LinearQuantizer`).
- Fix the `beta` parameter at `0` and either automatically search for or apply the configured `alpha` parameter.

**Formula**

```text
scales = (Act_Mean_Abs**alpha / Weight_Max_Abs**beta).clamp(min=1e-5)
```

where

- `Act_Mean_Abs` is the mean absolute value of the activation (`mean(abs(act))`).
- `Weight_Max_Abs` is the maximum absolute value of the weight, calculated using the maximum value of each column.
- `alpha` is the coefficient for activation scaling, which controls the influence of the activation on the scaling factor. This value ranges from 0 to 1 and can be determined through automatic search or manual configuration.
- `beta` is the coefficient for weight scaling, which is fixed at `0`.

**Key Features**

1. **Actual quantizer evaluation**: Unlike Flex Smooth Quant, Flex AWQ SSZ uses the actual quantizer (`LinearQuantizer`) to evaluate quantization error across different `alpha` parameters rather than relying on simple simulated quantization.
2. **Activation scale calculation**: The algorithm calculates the activation scale using the `mean` instead of the `max` of activation values, making it better suited for low-bit quantization scenarios.
3. **Fixed `beta` at `0`**: The algorithm fixes `beta` at `0` to simplify the parameter search space and focus exclusively on `alpha` parameter optimization.
4. **Automatic parameter search**: If no `alpha` parameter is provided, the algorithm searches for the optimal `alpha` within the range [0.0, 1.0] with a step size of 0.05, and selects the parameter that minimizes the mean squared error (MSE).

**Alpha Parameter Search Process**

1. **Initialization**: Create a `FlexAWQSSZAlphaBetaSearcher` and use the configured `qconfig`.
2. **Grid search**: Traverse `alpha` values in the range of [0.0, 1.0] with a step of 0.05.
3. **Quantization error evaluation**: For each `alpha` value:
   - Calculate the scaling factor: `scale = max(abs(Act_Mean_Abs)) ** alpha`
   - Apply scaling: `scaled_act = act / scale`, `scaled_weight = weight * scale`
   - Create and deploy the actual quantizer (`LinearQuantizer`).
   - Calculate the normalized MSE error between the quantized result and the floating-point result.
4. **Select the optimal parameter**: Select the `alpha` value that minimizes the MSE.

### Supported Subgraph Types

#### NormLinearSubgraph

This type applies to structures containing a normalization layer followed by multiple linear layers, such as:

```python
x = norm(x)
y = torch.cat([linear(x) for linear in linears], dim=-1)
```

**Process**

- Calculate the weight scaling factor by using the maximum column value across all linear layer weights.
- Calculate the activation scaling factor by using the mean of the activation values.
- Apply forward scaling to each linear layer.
- Apply inverse scaling (1/scales) to the normalization layer.

#### LinearLinearSubgraph

This type applies to structures containing a normalization layer followed by multiple linear layers, such as:

```python
y = linear2(linear1(x))
```

**Process**

- Calculate the scaling factor based on the weights of `linear2`.
- Calculate the activation scaling factor by using the mean of activation values output by `linear1`.
- Apply forward scaling to `linear2`.
- Apply inverse scaling (1/scales) to `linear1`.

#### OVSubgraph (Attention Output-Value Subgraph)

This type applies to the output projection and value projection in attention mechanisms, including support for:

- Multi-headed attention (MHA)
- Multi-query attention (MQA)
- Grouped-query attention (GQA)

**Process**

- Calculate the scaling factor based on the weights of `o_proj`.
- Calculate the activation scaling factor by using the mean of activation values output by `v_proj`.
- Apply forward scaling to `o_proj`.
- Apply inverse scaling (1/scales) to `v_proj`.

#### UpDownSubgraph (Up Projection – Down Projection Subgraph)

This type applies to MLP gating mechanisms, such as:

```python
y = down_proj(ReLU(gate_proj(x)) * up_proj(x))
```

**Process**

- Calculate the scaling factor based on the weights of `down_proj`.
- Calculate the activation scaling factor by using the mean of activation values output by `up_proj`.
- Apply forward scaling to `down_proj`.
- Apply inverse scaling (1/scales) to `up_proj`.

### Implementation

#### Code Implementation

The algorithm is implemented in [msmodelslim/processor/anti_outlier/flex_smooth/processor.py](../../../../../msmodelslim/processor/anti_outlier/flex_smooth/processor.py), and the processing flow is divided into two phases:

#### Preprocessing

**Subgraph Discovery and Construction**

- Retrieve global subgraph information by using the `SubgraphProcessor` to identify four subgraph types: `norm-linear`, `linear-linear`, `ov`, and `up-down`.
- Filter subgraphs based on the configured `include` and `exclude` patterns.

**Statistics Collection**

- Install forward hooks for all linear modules within the subgraphs.
- Collect activation statistics across the `[batch, seq, hidden_dim]` dimensions through these hooks.
  - **Activation tensor data**: Collect the full activation tensor to support subsequent smoothing computations.
  - **First linear layer statistics**: Use the activation statistics from the first linear layer in the subgraph `targets`.

#### Postprocessing

**Subgraph Processing by Priority**

- Process subgraphs according to the default priority order: `up-down` (highest) → `ov` (high) → `norm-linear` (medium) → `linear-linear` (low).
- Call the corresponding smoothing method for each specific subgraph type.

**Subgraph Smoothing**

- **NormLinearSubgraph**: Apply smoothing to the normalization layer and subsequent linear layers. If there are more than three linear layers, use only the first two for the `alpha` search.
- **LinearLinearSubgraph**: Apply smoothing to both linear layers and adjust their weights and biases.
- **OVSubgraph**: Process the connection between the output projection and value projection in the attention mechanism, including support for QKV fusion modes.
- **UpDownSubgraph**: Process the MLP gating mechanism and apply smoothing to both the up-projection and down-projection layers.

**Resource Cleanup**

- Remove all installed statistics hooks.
- Free the memory allocated for storing statistics.
- Restore the model to its original state.

## Application Requirements

- **Model architecture**: The model must implement the `FlexSmoothQuantInterface` and have its subgraph mapping relationships correctly defined.
- **Module naming**: Module names must exactly match the full paths returned by the `named_modules()` method.
- **Subgraph support**: The algorithm currently supports four standard subgraph types: `norm-linear`, `linear-linear`, `ov`, and `up-down`.
- **Module attributes**: Target modules must exist and possess a writable `weight` attribute. Other custom modules are currently not supported.
- **Model structure assumptions**: The algorithm is designed for the standard Transformer architecture. Exercise caution and carefully evaluate applicability when using non-standard structures.
- **Quantization configuration**: A valid `qconfig` must be provided, defining the quantization methods for both activations and weights. Typically, the SSZ method is used for weight quantization.

## Function Description

### YAML Configuration Example

The following example shows a YAML configuration when the algorithm is used as a processor:

```yaml
spec:
  process:
    - type: "flex_awq_ssz"                # Specifies the processor type. The value is fixed to flex_awq_ssz.
      alpha: 0.8                          # Specifies the activation scaling coefficient (0.0–1.0). The default value is None for automatic search.
      qconfig:                            # (Mandatory) Quantization configuration
        act:                              # Activation quantization configuration
          scope: "per_token"             # Specifies the quantization scope. Valid values: per_token or per_tensor.
          dtype: "int8"                   # Specifies the quantization data type: int8.
          symmetric: True                # Specifies whether to enable symmetric quantization. Valid values: True or False.
          method: "minmax"                # Specifies the quantization method: minmax.
        weight:                           # Weight quantization configuration
          scope: "per_channel"           # Specifies the quantization scope: per_channel.
          dtype: "int4"                   # Specifies the quantization data type. Valid values: int4 or int8.
          symmetric: True                # Enables symmetric quantization.
          method: "ssz"                   # Specifies the quantization method: ssz (Smooth Scale Zero).
          ext:                            # (Optional) Extended configuration
            step: 10                      # Specifies the step size parameter for the SSZ method.
      enable_subgraph_type:               # Specifies the list of enabled subgraph types. All four types are enabled by default.
        - 'norm-linear'
        - 'linear-linear'
        - 'ov'
        - 'up-down'
      include: ["*"]                      # Specifies the layers to be included. Wildcard matching is supported.
      exclude: ["*self_attn*"]            # Specifies the layers to be excluded. Wildcard matching is supported.
```

### YAML Configuration Fields

**Overall Field Description**

| Field| Purpose| Description|
|--------|------|------|
| `type` | Specifies the processor type identifier.| The value is fixed to `"flex_awq_ssz"`, which identifies the object as a flexible activation-aware quantization smoothing processor.|
| `alpha` | Specifies the activation scaling coefficient.| The value is a floating point number ranging from 0 to 1. It controls the influence of activations on the scaling factor. The default value is `None` for automatic search.|
| `qconfig` | Specifies the quantization configuration.| Mandatory. This section contains activation (`act`) and weight (`weight`) settings used for actual quantizer evaluation.|
| `qconfig.act` | Specifies the activation quantization configuration.| This section includes fields such as `scope`, `dtype`, `symmetric`, and `method` to define the activation quantization method.|
| `qconfig.weight` | Specifies the weight quantization configuration.| This section includes fields such as `scope`, `dtype`, `symmetric`, and `method` to define the weight quantization method. It is typically used in combination with the SSZ method.|
| `enable_subgraph_type` | Specifies the enabled subgraph types.| The values are a list of supported subgraph types, including `"norm-linear"`, `"linear-linear"`, `"ov"` and `"up-down"`.|
| `include` | Specifies the layers to be included.| Wildcard matching is supported to identify the names of the modules to be processed.|
| `exclude` | Specifies the layers to be excluded.| Wildcard matching is supported to identify the module names to be excluded from the processing flow.|

#### `qconfig.act` (Activation Quantization Configuration)

**Purpose**: Configures quantization parameters for activation values.

| Parameter| Purpose| Valid Values                                      | Description                                                                                                              | Default Value|
|--------|------|-------------------------------------------|------------------------------------------------------------------------------------------------------------------|--------|
| `scope` | Specifies the quantization scope.| `"per_tensor"`, `"per_token"`, `"pd_mix"`| `per_tensor`: The entire tensor uses the same parameters.<br>`per_token`: Each token uses independent parameters (dynamic quantization).<br>`pd_mix`: `per_token` is used during prefilling and `per_tensor` during decoding.| `"per_tensor"` |
| `dtype` | Specifies the quantization data type.| `"int8"`, `"int4"`                       | This parameter specifies 8-bit or 4-bit integer quantization.                                                                                                 | `"int8"` |
| `symmetric` | Specifies whether to enable symmetric quantization.| `true`, `false`                          | `true`: enables symmetric quantization, with a zero point of `0`.<br>`false`: enables asymmetric quantization, with an adjustable zero point.                                                               | `false` |
| `method` | Specifies the quantization method.| `"minmax"`, `"histogram"`                | `minmax`: specifies MinMax quantization.<br>`histogram`: specifies histogram activation quantization.                                                                  | `"minmax"` |

#### `qconfig.weight` (Weight Quantization Configuration)

**Purpose**: Configures quantization parameters for weights.

| Parameter| Purpose| Valid Values| Description| Default Value|
|--------|------|--------|------|--------|
| `scope` | Specifies the quantization scope.| `"per_tensor"`, `"per_channel"`| `per_tensor`: The entire tensor uses the same parameters.<br>`per_channel`: Each channel uses independent parameters.| `"per_channel"` |
| `dtype` | Specifies the quantization data type.| `"int8"`, `"int4"`| This parameter specifies 8-bit or 4-bit integer quantization.| `"int8"` |
| `symmetric` | Specifies whether to enable symmetric quantization.| `true`, `false`| `true`: enables symmetric quantization, with a zero point of `0`.<br>`false`: enables asymmetric quantization, with an adjustable zero point.| `true` |
| `method` | Specifies the quantization method.| `"minmax"`, `"ssz"`, `"gptq"`| `minmax`: specifies MinMax quantization.<br>`ssz`: specifies SSZ quantization.<br>`gptq`: specifies GPTQ quantization.| `"minmax"` |

## Model Adaptation

### Interface and Data Structure

Flex AWQ SSZ uses the same interface as Flex Smooth Quant:

```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod

@dataclass
class MappingConfig:
    """Module mapping configuration"""
    source: str  # Specifies the source module name, such as "model.layers.0.input_layernorm".
    targets: List[str] # Specifies a list of target module names, such as ["model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj"].

@dataclass
class FusionConfig:
    """Fusion configuration (supporting advanced functions such as QKV fusion)"""
    fusion_type: str = "none"  # Specifies the fusion type, such as none, qkv, or custom.
    num_attention_heads: Optional[int] = None  # Specifies the number of attention heads.
    num_key_value_heads: Optional[int] = None  # Specifies the number of key-value heads.
    custom_config: Optional[Dict[str, Any]] = None  # Specifies the custom configuration.

@dataclass
class AdapterConfig:
    """Subgraph adapter configuration"""
    subgraph_type: str  # Specifies the subgraph type, such as norm-linear, linear-linear, ov, or up-down.
    mapping: Optional[MappingConfig] = None  # Specifies the module mapping relationships.
    fusion: FusionConfig = field(default_factory=lambda: FusionConfig())  # Specifies the fusion configuration.

# Interface for adapting models to the Flex AWQ SSZ algorithm (same as Flex Smooth Quant)
class FlexSmoothQuantInterface(ABC):
    @abstractmethod
    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        """
        Return all subgraph configurations within the model eligible for Flex AWQ SSZ processing.

        Returns:
            List[AdapterConfig]: A list of subgraph configurations, each containing:
                - subgraph_type: indicates the subgraph type.
                - mapping: indicates the mapping from the source module to the target modules.
                - fusion: indicates fusion configuration (such as QKV fusion).
        """
        pass
```

### Adaptation Procedure

**Prerequisites**

- The model must inherit the `FlexSmoothQuantInterface` (same as Flex Smooth Quant).
- Module names must exactly match the full paths returned by `named_modules()`.
- Supported subgraph types include `norm-linear`, `linear-linear`, `ov`, and `up-down`.
- The `subgraph_type` and `mapping` parameters are mandatory in the configuration.
- When `FusionConfig` is used and `fusion_type` is set to `qkv`, `num_attention_heads` and `num_key_value_heads` must be specified.

**Procedure**

1. **Inherit the interface**: Inherit the `FlexSmoothQuantInterface` in the model adapter and implement the `get_adapter_config_for_subgraph()` method.
2. **Configure subgraph mappings**: Configure four types of subgraph mapping relationships for each layer.
   - **NormLinearSubgraph**: Map the normalization layer to its subsequent linear layers.
   - **OVSubgraph**: Map the value (V) projection to the output (O) projection within the attention mechanism.
   - **UpDownSubgraph**: Map the up-projection to the down-projection within the MLP gating mechanism.
   - **LinearLinearSubgraph**: Map consecutive linear layers.
3. **Specify module paths**: Use absolute module paths, such as `model.layers.{i}.self_attn.q_proj`.

**Reference implementation**: For details, see the `Qwen3ModelAdapter` implementation in [msmodelslim/model/qwen3/model_adapter.py](../../../../../msmodelslim/model/qwen3/model_adapter.py).

### Configuration Example

The following example shows the configuration of a typical Transformer layer (same as Flex Smooth Quant):

```python
def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
    adapter_config = []
    for layer_idx in range(self.config.num_hidden_layers):
        # 1. Norm-Linear mapping from the input layer normalization to the QKV projection
        norm_linear_config1 = AdapterConfig(
            subgraph_type="norm-linear",
            mapping=MappingConfig(
                source=f"model.layers.{layer_idx}.input_layernorm",
                targets=[
                    f"model.layers.{layer_idx}.self_attn.q_proj",
                    f"model.layers.{layer_idx}.self_attn.k_proj",
                    f"model.layers.{layer_idx}.self_attn.v_proj"
                ]
            )
        )

        # 2. Norm-Linear mapping from the post-attention layer normalization to the MLP projection
        norm_linear_config2 = AdapterConfig(
            subgraph_type="norm-linear",
            mapping=MappingConfig(
                source=f"model.layers.{layer_idx}.post_attention_layernorm",
                targets=[
                    f"model.layers.{layer_idx}.mlp.gate_proj",
                    f"model.layers.{layer_idx}.mlp.up_proj"
                ]
            )
        )

        # 3. OV mapping within the attention mechanism
        ov_config = AdapterConfig(
            subgraph_type="ov",
            mapping=MappingConfig(
                source=f"model.layers.{layer_idx}.self_attn.v_proj",
                targets=[f"model.layers.{layer_idx}.self_attn.o_proj"]
            )
        )

        # 4. Up-Down mapping for the MLP gating mechanism
        up_down_config = AdapterConfig(
            subgraph_type="up-down",
            mapping=MappingConfig(
                source=f"model.layers.{layer_idx}.mlp.up_proj",
                targets=[f"model.layers.{layer_idx}.mlp.down_proj"]
            )
        )

        adapter_config.extend([norm_linear_config1, norm_linear_config2, ov_config, up_down_config])

    return adapter_config
```

## FAQ

### Module Name Mismatch

**Symptom**: When `include` and `exclude` patterns do not hit, the log indicates that no pattern was matched.

**Solution**: Verify that the complete module name exactly matches the path returned by `named_modules()`.

### Subgraph Configuration Error

**Symptom**: The configuration returned by `get_adapter_config_for_subgraph()` is incorrect.

**Solution**: Check whether the `source` and `targets` fields in the configuration are correct.

### Module Does Not Exist

**Symptom**: A module name specified in the configuration does not exist within the model.

**Solution**: Verify the existence of the module by using `model.named_modules()`.

### Unsupported Subgraph Type

**Symptom**: The configured subgraph type is not supported.

**Solution**: Ensure that the configured subgraph type is in the supported list. Currently, four subgraph types are supported: `norm-linear`, `linear-linear`, `ov` and `up-down`.

### Missing `qconfig` Configuration

**Symptom**: An error occurs, indicating that the `qconfig` parameter is mandatory.

**Solution**: Add the `qconfig` field to the YAML configuration, including the activation (`act`) and weight (`weight`) quantization configurations.

### Incorrect Mapping Relationship

**Symptom**: The `source` and `targets` in `MappingConfig` point to incorrect modules.

**Solution**: Check whether the `source` and `targets` in the `MappingConfig` point to the correct modules.
