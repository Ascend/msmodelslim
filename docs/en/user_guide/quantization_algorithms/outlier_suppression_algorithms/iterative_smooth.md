# Iterative Smooth: Outlier Suppression Algorithm

## Overview

- **Introduction**: Iterative Smooth is an algorithm designed to suppress activation outliers during the quantization of large language models. This algorithm dynamically adjusts the scaling factors of weights and activations to effectively reduce quantization error while maintaining model accuracy.
- **Core idea**: The Iterative Smooth algorithm makes the distribution of activation values more even by redistributing quantization error between adjacent layers, thereby reducing the impact of outliers on quantization accuracy.

## Preparations

Install msModelSlim. For details, see [msModelSlim Installation Guide](../../../install_guide/install_guide.md).

## Principle and Implementation

### Principle

**Core Logic**

- Calculate per_channel scaling factors based on collected activation statistics.
- Optimize subgraphs by using the `iter_smooth` algorithm for iterative smooth quantization.
- Support configurable smoothing parameters: `alpha` (smoothing strength), `scale_min` (minimum scaling factor), and `symmetric` (symmetric quantization).

**Formula**

```text
scales = (A_scale**α / W_scale**(1-α)).clamp(min=scale_min)
```

where

- `A_scale` is the scaling factor of the activation values.
- `W_scale` is the scaling factor of the weights, calculated using the maximum value of each column.
- `α` is the balance coefficient that controls the relative importance of activations and weights (default value: `0.9`)
- `scale_min` is the minimum value of the scaling factor (default value: `1e-5`)

### Supported Subgraph Types

#### NormLinearSubgraph

This type applies to structures containing a normalization layer followed by multiple linear layers, such as:

```python
x = norm(x)
y = torch.cat([linear(x) for linear in linears], dim=-1)
```

**Process**

- Calculate the weight scaling factor by using the maximum column value across all linear layer weights.
- Apply forward scaling to each linear layer.
- Apply inverse scaling (1/scales) to the normalization layer.

#### LinearLinearSubgraph

This type applies to structures containing a normalization layer followed by multiple linear layers, such as:

```python
y = linear2(linear1(x))
```

**Process**

- Calculate the scaling factor based on the weights of `linear2`.
- Apply forward scaling to `linear2`.
- Apply inverse scaling (1/scales) to `linear1`.

#### OVSubgraph (Attention Output-Value Subgraph)

This type applies to the output projection and value projection in attention mechanisms, including support for:

- Multi-headed attention (MHA)
- Multi-query attention (MQA)
- Grouped-query attention (GQA)

**Process**

- Calculate the scaling factor based on the weights of `o_proj`.
- Apply forward scaling to `o_proj`.
- Apply inverse scaling (1/scales) to `v_proj`.

#### UpDownSubgraph (Up Projection – Down Projection Subgraph)

This type applies to MLP gating mechanisms, such as:

```python
y = down_proj(ReLU(gate_proj(x)) * up_proj(x))
```

**Process**

- Calculate the scaling factor based on the weights of `down_proj`.
- Apply forward scaling to `down_proj`.
- Apply inverse scaling (1/scales) to `up_proj`.

#### NonFusionSubgraph

This type applies to scenarios where **several linear layers are smoothed without being fused into the preceding layer**. When `source` in the mapping configuration is set to `None` and only `targets` is specified, the processor executes the non-fusion branch by using the linear layers in `targets` to form a NonFusionSubgraph for processing.

**Typical Scenarios**

- There are no fusible preceding normalization or linear layers, and outlier suppression is only required for independent linear layers.
- The structure is unique and cannot be categorized as `norm-linear`, `linear-linear`, `ov`, or `up-down`, but these linear layers still require smoothing.

**Process**

- Treat multiple linear layers in `targets` as a single group, and compute a unified per_channel scaling factor by using the maximum column value of the weights and activation statistics for this group.
- Apply forward scaling to the weights of each linear layer.
- Register a forward pre-hook on each linear layer. Apply the corresponding inverse scaling (1/scales) to the **input** during inference by using the forward hook. This operation maintains numerical consistency with the weight scaling and is numerically equivalent to dividing the input by the scale and then multiplying it back.
- **Shift (bias translation) is not supported**. If shift is configured, the non-fusion branch ignores the configuration and generates a log message.

**Configuration method:** In the `AdapterConfig` returned by `get_adapter_config_for_subgraph()`, configure the `mapping` by using `MappingConfig(source=None, targets=[...])`. The `targets` parameter specifies the list of full paths for the linear layers to be smoothed, and you can set `subgraph_type` to `norm-linear` (indicating that all linear names are recorded when multiple objectives are present) or other supported types. This configuration affects only the internal naming and does not change the non-fusion behavior of the processor.

### Implementation

#### Code Implementation

The algorithm is implemented in [msmodelslim/processor/anti_outlier/iter_smooth/processor.py](../../../../../msmodelslim/processor/anti_outlier/iter_smooth/processor.py), and the processing flow consists of two phases.

#### Preprocessing

**Subgraph Discovery and Construction**

- Retrieve global subgraph information by using the `SubgraphProcessor` to identify four subgraph types: `norm-linear`, `linear-linear`, `ov`, and `up-down`.
- Filter subgraphs based on the configured `include` and `exclude` patterns.

**Statistics Collection**

- Install forward hooks for all linear modules within the subgraphs.
- Collect activation statistics across the `[batch, seq, hidden_dim]` dimensions by using these hooks. These statistics include:
  - Maximum and minimum values of each channel
  - Per-channel absolute maximum value used to calculate the smooth scaling factor.
  - Channel offset used for symmetric quantization.
- Aggregate statistics in distributed training environments.

#### Postprocessing

**Subgraph Processing by Priority**

- Process subgraphs according to the default priority order: `up-down` (highest) → `ov` (high) → `norm-linear` (medium) → `linear-linear` (low).
- Call the corresponding smoothing method for each specific subgraph type.

**Subgraph Smoothing**

- **NormLinearSubgraph**: Apply smoothing to the normalization layer and subsequent linear layers. Support the adjustment of the `RMSNorm` bias by using the algorithm.
- **LinearLinearSubgraph**: Apply smoothing to both linear layers and adjust their weights and biases.
- **OVSubgraph**: Process the connection between the output projection and value projection in the attention mechanism, including support for QKV fusion modes.
- **UpDownSubgraph**: Process the MLP gating mechanism and apply smoothing to both the up-projection and down-projection layers.
- **NonFusionSubgraph**: When `mapping.source` is `None` and `mapping.targets` is not empty, group the target linear layers into a `NonFusionSubgraph`. Scale the weights and register an input-side scale pre-hook on each layer without fusing these layers into the preceding layer. Bias translation (shift) is not supported.

**Resource Cleanup**

- Remove all installed statistics hooks.
- Free the memory allocated for storing statistics.
- Restore the model to its original state.

## Application Requirements

- **Model architecture**: The model must implement the `IterSmoothInterface` and have its subgraph mapping relationships correctly defined.
- **Module naming**: Module names must exactly match the full paths returned by the `named_modules()` method.
- **Subgraph support**: The algorithm currently supports four standard subgraph types: norm-linear, linear-linear, ov and up-down. It also supports **NonFusionSubgraph** (with a mapping configuration where `source` is set to `None` and only a list of linear layers is provided for `targets`).
- **Module attributes**: Target modules must exist and possess a writable `weight` attribute (and optional `bias`). Other custom modules are currently not supported.
- **Model structure assumptions**: The algorithm is designed for the standard Transformer architecture. Exercise caution and carefully evaluate applicability when using non-standard structures.

## Function Description

### YAML Configuration Example

The following example shows a YAML configuration when the algorithm is used as a processor:

```yaml
spec:
  process:
    - type: "iter_smooth"                 # Specifies the processor type. The value is fixed to iter_smooth.
      alpha: 0.9                           # Specifies the balance parameter that controls the relative importance of activations and weights. Default value: 0.9.
      scale_min: 1e-5                      # Specifies the minimum value of the scaling factor to prevent numerical instability. Default value: 1e-5.
      symmetric: True                      # Specifies whether to enable symmetric quantization. Default value: True.
      enable_subgraph_type:                # Specifies the enabled subgraph types.
        - 'norm-linear'
        - 'linear-linear'
        - 'ov'
        - 'up-down'
      include: ["*"]                       # Specifies the layers to be included. Wildcard matching is supported.
      exclude: ["*self_attn*"]             # Specifies the layers to be excluded. Wildcard matching is supported.
```

### YAML Configuration Fields

| Field| Purpose     | Description|
|--------|---------|------|
| type | Specifies the processor type identifier.| The value is fixed to `iter_smooth`, which identifies the processor as an iterative smoothing processor.|
| alpha | Specifies the balance parameter.   | This parameter is a floating-point number greater than 0 that controls the relative importance of activations and weights. Default value: `0.9`.|
| scale_min | Specifies the minimum value of the scaling factor.| This parameter is a floating-point number greater than 0 that prevents numerical instability. Default value: `1e-5`.|
| symmetric | Specifies whether to enable symmetric quantization. | This parameter is a boolean value. `True` enables symmetric quantization, while `False` enables asymmetric quantization. Default value: `True`.|
| enable_subgraph_type | Specifies the enabled subgraph types.| The values are a list of supported subgraph types, including `"norm-linear"`, `"linear-linear"`, `"ov"` and `"up-down"`.|
| include | Specifies the layers to be included. | Wildcard matching is supported.|
| exclude | Specifies the layers to be excluded. | Wildcard matching is supported.|

## Model Adaptation

### Interface and Data Structure

```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod

@dataclass
class MappingConfig:
    """Module mapping configuration"""
    source: Optional[str]  # Specifies the source module name, such as "model.layers.0.input_layernorm". If this parameter is set to None, the processor performs smoothing only on the linear layers in targets as a NonFusionSubgraph.
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
    subgraph_type: str  # Specifies the subgraph type. Valid values: norm-linear, linear-linear, ov, or up-down. You can set it to a value such as norm-linear for a NonFusionSubgraph, which affects only the internal naming.
    mapping: Optional[MappingConfig] = None  # Specifies the module mapping relationships.
    fusion: FusionConfig = field(default_factory=lambda: FusionConfig())  # Specifies the fusion configuration.

# Interface for adapting models to the Iterative Smooth algorithm
class IterSmoothInterface(ABC):
    @abstractmethod
    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        """
        Return all subgraph configurations within the model eligible for smoothing.

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

- The model must inherit the `IterSmoothInterface`.
- Module names must exactly match the full paths returned by `named_modules()`.
- The following subgraph types are supported: `norm-linear`, `linear-linear`, `ov` and `up-down`. **NonFusionSubgraph** is also supported (with a mapping configuration where`mapping.source` is set to `None` and only `targets` is specified).
- The `subgraph_type` and `mapping` parameters are mandatory in the configuration.
- When `FusionConfig` is used and `fusion_type` is set to `qkv`, `num_attention_heads` and `num_key_value_heads` must be specified.

**Procedure**

1. **Inherit the interface**: Inherit the `IterSmoothInterface` in the model adapter and implement the `get_adapter_config_for_subgraph()` method.
2. **Configure subgraph mappings**: Configure subgraph mapping relationships for each layer.
   - **NormLinearSubgraph**: Map the normalization layer (`source`) to its subsequent linear layers (`targets`).
   - **OVSubgraph**: Map the value (V) projection to the output (O) projection within the attention mechanism.
   - **UpDownSubgraph**: Map the up-projection to the down-projection within the MLP gating mechanism.
   - **LinearLinearSubgraph**: Map consecutive linear layers.
   - **NonFusionSubgraph**: Set `source` to `None` and `targets` to a list of paths for linear layers to be smoothed (which do not require fusion into the preceding layer).
3. **Specify module paths**: Use absolute module paths, such as `model.layers.{i}.self_attn.q_proj`.

**Reference implementation**: For details, see the Qwen3ModelAdapter implementation in [msmodelslim/model/qwen3/model_adapter.py](../../../../../msmodelslim/model/qwen3/model_adapter.py).

### Configuration Example

The following example shows the configuration of a typical Transformer layer:

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

        # 5. Example of a NonFusionSubgraph: Smoothing is performed only on several linear layers, without fusion into the preceding layer (source is set to None).
        # non_fusion_config = AdapterConfig(
        #     subgraph_type="norm-linear",
        #     mapping=MappingConfig(
        #         targets=[
        #             f"model.layers.{layer_idx}.some_module.linear_a",
        #             f"model.layers.{layer_idx}.some_module.linear_b",
        #         ]
        #     )
        # )
        # adapter_config.append(non_fusion_config)

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

**Solution**: Ensure that the configured subgraph type exists in the `ENABLE_SUBGRAPH_TYPES` list.

### Incorrect Mapping Relationship

**Symptom**: The `source` and `targets` in `MappingConfig` point to incorrect modules.

**Solution**: Check whether the `source` and `targets` in the `MappingConfig` point to the correct modules.

### NonFusionSubgraph Does Not Take Effect

**Symptom**: `source=None` and `targets` have been configured, but non-fusion smoothing is not performed.

**Solution**: Confirm that `mapping.source` is `None` (explicitly passed as `None` through Python) and `mapping.targets` is not empty. Confirm that these target modules are within the `include` scope and are not filtered out by using `exclude`.
