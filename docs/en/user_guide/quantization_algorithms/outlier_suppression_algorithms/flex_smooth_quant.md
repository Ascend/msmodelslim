# Flex Smooth Quant: Flexible Smooth Quantization Algorithm

## Overview

- **Introduction**: Flex Smooth Quant is an algorithm designed to suppress activation outliers during the quantization of large language models. This algorithm dynamically adjusts the scaling factors of weights and activations to effectively reduce quantization error while maintaining model accuracy. Unlike traditional smoothing algorithms, Flex Smooth Quant provides more flexible parameter configurations, enabling adaptive adjustments based on different model architectures and quantization requirements.
- **Core idea**: Flex Smooth Quant achieves a finer balance between activations and weights by using a two-stage grid search to automatically search for the optimal `alpha` and `beta` parameters, thereby achieving a balance between accuracy and quantization efficiency across various quantization scenarios.

## Preparations

Install msModelSlim. For details, see [msModelSlim Installation Guide](../../../install_guide/install_guide.md).

## Principle and Implementation

### Principle

**Core Logic**

- Calculate per_channel scaling factors based on collected activation statistics.
- Optimize subgraphs by using the `flex_smooth_quant` algorithm for flexible smooth quantization.
- Configure smoothing parameters (`alpha` for activation scaling and `beta` for weight scaling), or employ a two-stage grid search to determine the optimal `alpha` and `beta` parameters if they are not provided.

**Formula**

```text
scales = (A_scale**alpha / W_scale**beta).clamp(min=1e-5)
```

where

- `A_scale` is the scaling factor of the activation values.
- `W_scale` is the scaling factor of the weights, calculated using the maximum value of each column.
- `alpha` is the coefficient for activation scaling, which controls the influence of the activation on the scaling factor (ranging from 0 to 1).
- `beta` is the coefficient for weight scaling, which controls the influence of the weight on the scaling factor (ranging from 0 to 1).

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

This type applies to scenarios where **several linear layers are smoothed without being fused into the preceding layer**. When `source` in the mapping configuration is set to `None` and only `targets` is specified, the processor executes the non-fusion branch by using the linear layers in `targets` to form a NonFusionSubgraph for Flex Smooth processing.

**Typical Scenarios**

- There are no fusible preceding normalization or linear layers, and outlier suppression is only required for independent linear layers.
- The structure is unique and cannot be categorized as `norm-linear`, `linear-linear`, `ov`, or `up-down`, but these linear layers still require smoothing.

**Process**

- Treat multiple linear layers in `targets` as a single group. Perform an **`alpha` or `beta` search** based on collected activations and weights (or use the configured `alpha` or `beta`) to determine the optimal scaling, and compute a unified per_channel scale.
- Apply forward scaling to the weight of each linear layer using `NonFusionSubgraphFusion` of `SubgraphFusionFactory`).
- Register a forward pre-hook (`NonFusionSmoothQuantHookIR`) on each linear layer. Apply the corresponding inverse scaling (1/scales) to the **input** during inference to maintain numerical consistency with the weight scaling.

**Configuration method**: In the `AdapterConfig` returned by `get_adapter_config_for_subgraph()`, set `mapping` to `MappingConfig(source=None, targets=[...])`, where `targets` is the list of full paths of the linear layers to be smoothed. `subgraph_type` can be set to `"norm-linear"` or other supported types. It only affects the internal naming and does not change the non-fusion behavior.

### Implementation

#### Code Implementation

The algorithm is implemented in `msmodelslim/processor/anti_outlier/flex_smooth/processor.py`. The processing flow consists of two phases:

#### Preprocessing

**Subgraph Discovery and Construction**

- Retrieve global subgraph information by using the `SubgraphProcessor` to identify four subgraph types: `norm-linear`, `linear-linear`, `ov`, and `up-down`.
- Filter subgraphs based on the configured `include` and `exclude` patterns.

**Statistics Collection**

- Install forward hooks for all linear modules within the subgraphs.
- Collect activation statistics across the `[batch, seq, hidden_dim]` dimensions by using these hooks. These statistics include:
  - **Activation tensor data**: Collect the full activation tensor to support subsequent smoothing computations.
  - **Absolute maximum value per channel**: Calculate the absolute maximum value of the activations per channel, and use it as the basis for calculating the smoothing scaling factor.

#### Postprocessing

**Subgraph Processing by Priority**

- Process subgraphs according to the default priority order: `up-down` (highest) → `ov` (high) → `norm-linear` (medium) → `linear-linear` (low).
- Call the corresponding smoothing method for each specific subgraph type.

**Subgraph Smoothing**

- **NormLinearSubgraph**: Apply smoothing to the normalization layer and subsequent linear layers.
- **LinearLinearSubgraph**: Apply smoothing to both linear layers and adjust their weights and biases.
- **OVSubgraph**: Process the connection between the output projection and value projection in the attention mechanism, including support for QKV fusion modes.
- **UpDownSubgraph**: Process the MLP gating mechanism and apply smoothing to both the up-projection and down-projection layers.
- **NonFusionSubgraph**: When `mapping.source` is `None` and `mapping.targets` is not empty, group the target linear layers into a NonFusionSubgraph. Perform an `alpha` or `beta` search (or use the configured values), scale the weights, and register an input-side scale pre-hook on each layer.

**Resource Cleanup**

- Remove all installed statistics hooks.
- Free the memory allocated for storing statistics.
- Restore the model to its original state.

## Application Requirements

- **Model architecture**: The model must implement the `FlexSmoothQuantInterface` and have its subgraph mapping relationships correctly defined.
- **Module naming**: Module names must exactly match the full paths returned by the `named_modules()` method.
- **Subgraph support**: The algorithm currently supports four standard subgraph types: `norm-linear`, `linear-linear`, `ov` and `up-down`. It also supports **NonFusionSubgraph** (with a mapping configuration where `source` is set to `None` and only a list of linear layers is provided for `targets`).
- **Module attributes**: Target modules must exist and possess a writable `weight` attribute. Other custom modules are currently not supported.
- **Model structure assumptions**: The algorithm is designed for the standard Transformer architecture. Exercise caution and carefully evaluate applicability when using non-standard structures.

## Function Description

### YAML Configuration Example

The following example shows a YAML configuration when the algorithm is used as a processor:

```yaml
spec:
  process:
    - type: "flex_smooth_quant"  # Specifies the processor type. The value is fixed to flex_smooth_quant.
      alpha: 0.8                 # Specifies the coefficient for activation scaling, ranging from 0 to 1. The default value is None. The algorithm automatically searches for the optimal alpha. Manual configuration is also supported.
      beta: 0.7                  # Specifies the coefficient for weight scaling, ranging from 0 to 1. The default value is None. The algorithm automatically searches for the optimal beta. Manual configuration is also supported.
      enable_subgraph_type:      # Specifies the list of enabled subgraph types. All four types are enabled by default.
        - 'norm-linear'
        - 'linear-linear'
        - 'ov'
        - 'up-down'
      include: ["*"]             # Specifies the layers to be included. Wildcard matching is supported.
      exclude: ["*self_attn*"]   # Specifies the layers to be excluded. Wildcard matching is supported.
```

### YAML Configuration Fields

| Field| Purpose| Description|
|--------|------|------|
| type | Specifies the processor type identifier.| The value is fixed to `"flex_smooth_quant"`, which identifies the object as a flexible smooth quantization processor.|
| alpha | Specifies the activation scaling coefficient.| The value is a floating point number ranging from 0 to 1. It controls the influence of activations on the scaling factor. The default value is `None` for automatic search.|
| beta | Specifies the weight scaling coefficient.| The value is a floating point number ranging from 0 to 1. It controls the influence of weights on the scaling factor. The default value is `None` for automatic search.|
| enable_subgraph_type | Specifies the enabled subgraph types.| The values are a list of supported subgraph types, including `"norm-linear"`, `"linear-linear"`, `"ov"` and `"up-down"`.|
| include | Specifies the layers to be included.| Wildcard matching is supported.|
| exclude | Specifies the layers to be excluded.| Wildcard matching is supported.|

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

# Interface for adapting models to the Flex Smooth Quant algorithm
class FlexSmoothQuantInterface(ABC):
    @abstractmethod
    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        """
        Return all subgraph configurations within the model eligible for Flex Smooth Quant processing.

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

- The model must inherit the `FlexSmoothQuantInterface`.
- Module names must exactly match the full paths returned by `named_modules()`.
- The following subgraph types are supported: `norm-linear`, `linear-linear`, `ov` and `up-down`. **NonFusionSubgraph** is also supported (with a mapping configuration where`mapping.source` is set to `None` and only `targets` is specified).
- The `subgraph_type` and `mapping` parameters are mandatory in the configuration.
- When `FusionConfig` is used and `fusion_type` is set to `qkv`, `num_attention_heads` and `num_key_value_heads` must be specified.

**Procedure**

1. **Inherit the interface**: Inherit the `FlexSmoothQuantInterface` in the model adapter and implement the `get_adapter_config_for_subgraph()` method.
2. **Configure subgraph mappings**: Configure subgraph mapping relationships for each layer.
   - **NormLinearSubgraph**: Map the normalization layer (`source`) to its subsequent linear layers (`targets`).
   - **OVSubgraph**: Map the value (V) projection to the output (O) projection within the attention mechanism.
   - **UpDownSubgraph**: Map the up-projection to the down-projection within the MLP gating mechanism.
   - **LinearLinearSubgraph**: Map consecutive linear layers.
   - **NonFusionSubgraph**: Set `source` to `None` and `targets` to a list of paths for linear layers to be smoothed (which do not require fusion into the preceding layer).

3. **Specify module paths**: Use absolute module paths, such as `model.layers.{i}.self_attn.q_proj`.

**Reference implementation**: For details, see the implementation of `Qwen3ModelAdapter` in `msmodelslim/model/qwen3/model_adapter.py`.

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

        # 5. Example of a NonFusionSubgraph: Flex Smooth smoothing is performed only on several linear layers, without fusion into the preceding layer (source is set to None).
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
