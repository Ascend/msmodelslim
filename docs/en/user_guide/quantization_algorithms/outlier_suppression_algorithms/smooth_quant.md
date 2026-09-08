# SmoothQuant: Outlier Suppression Algorithm

## Overview

- **Source**: The SmoothQuant algorithm proposed by MIT.
- **Introduction**: SmoothQuant is an algorithm used to suppress activation outliers during the quantization of large language models. This algorithm smooths outliers in activation values into weights through cooperative scaling between the normalization layers and the linear layers, making the activation values easier to quantize.
- **Core idea**: SmoothQuant uses mathematical equivalence transformation to divide activation values by a smoothing factor and multiply weights by the same factor. This ensures that the distribution of activation values is more uniform without changing the output of the model, reducing the impact of outliers on quantization accuracy.

## Preparations

Install msModelSlim. For details, see [msModelSlim Installation Guide](../../../install_guide/install_guide.md).

## Principle and Implementation

### Principle

**Core Logic**

- Use the `smooth_quant` algorithm to smooth subgraphs.
- Support configurable smoothing parameters: `alpha` (smoothing strength) and `symmetric` (symmetric quantization).
- The lower bound of the scaling factor is fixed at `1e-5`.

**Formula**

The SmoothQuant algorithm is based on the following mathematical equivalence transformation:

```text
Y = XW = (X · diag(s)^(-1)) · (diag(s) · W) = X̂ · Ŵ
```

where

- `X` is the activation value.
- `W` is the weight.
- `s` is the smoothing scaling factor.
- `X̂ = X · diag(s)^(-1)` is the smoothed activation value.
- `Ŵ = diag(s) · W` is the smoothed weight.

The formula for calculating the smoothing scaling factor is as follows:

```text
scales = (A_scale**α / W_scale**(1-α)).clamp(min=1e-5)
```

where

- `A_scale` is the per_channel absolute maximum activation value.
- `W_scale` is the per-column absolute maximum weight value.
- `α` is the balance parameter that controls the relative importance of activations and weights (default value: 0.5).
- `1e-5` is the minimum value of the scaling factor to prevent numerical instability.

### Supported Subgraph Types

#### NormLinearSubgraph

SmoothQuant supports only the NormLinearSubgraph type.

This type applies to structures containing a normalization layer followed by multiple linear layers, such as:

```python
x = norm(x)
y = torch.cat([linear(x) for linear in linears], dim=-1)
```

**Process**

- Calculate the weight scaling factor by using the maximum column value across all linear layer weights.
- Perform forward scaling on each linear layer by multiplying the weights by `scales`.
- Perform inverse scaling on the normalization layer by dividing the weights by `scales`.
- Compute and apply the offset if asymmetric quantization is enabled.

### Implementation

#### Code Implementation

The algorithm is implemented in [msmodelslim/processor/anti_outlier/smooth_quant/](../../../../../msmodelslim/processor/anti_outlier/smooth_quant). The processing flow consists of two phases.

#### Preprocessing

**Subgraph Discovery and Construction**

- Obtain subgraph information through `get_adapter_config_for_subgraph()` of the model adapter.
- Process only subgraphs of the `norm-linear` type. Other types are filtered automatically.
- Filter subgraphs based on the configured `include` and `exclude` patterns.

**Normalization Layer Replacement**

- Replace the original `RMSNorm` module with the `RMSNormBias` module to correctly process the offset in asymmetric quantization mode.

**Statistics Collection**

- Install forward hooks for all linear modules within the subgraphs.
- Collect activation statistics across the `[batch, seq, hidden_dim]` dimensions by using these hooks. These statistics include:
  - Per-channel absolute maximum value used to calculate the smooth scaling factor.
  - Channel offset used for asymmetric quantization.

#### Postprocessing

**Subgraph Smoothing**

- Traverse all `norm-linear` subgraphs to apply the smoothing algorithm in sequence.
- Calculate the smooth scaling factor based on the collected activation statistics and weight information.
- Apply inverse scaling and forward scaling to the normalization layers and linear layers, respectively.

**Resource Cleanup**

- Remove all installed statistics hooks.
- Free the memory allocated for storing statistics.
- Restore the model to its original state.

## Application Requirements

- **Model architecture**: The model must implement the `SmoothQuantInterface` and have its subgraph mapping relationships correctly defined.
- **Module naming**: The module names must exactly match the full paths returned by the `named_modules()` method.
- **Supported subgraph types**: SmoothQuant supports only the `norm-linear` subgraph type.
- **Module attributes**: Target modules must exist and possess a writable `weight` attribute (and optional `bias`).
- **Model structure assumptions**: The algorithm is designed for the standard Transformer architecture. Exercise caution and carefully evaluate applicability when using non-standard structures.

## Function Description

### YAML Configuration Example

The following example shows a YAML configuration when the algorithm is used as a processor:

```yaml
spec:
  process:
    - type: "smooth_quant"                 # Specifies the processor type. The value is fixed to smooth_quant.
      alpha: 0.5                           # Specifies the balance parameter that controls the relative importance of activations and weights. The value is a floating point number ranging from 0 to 1. Default value: 0.5.
      symmetric: True                      # Specifies whether to enable symmetric quantization. Valid values: True or False.
      include: ["*"]                       # Specifies the layers to be included. Wildcard matching is supported. Default value: ["*"] (all layers are included).
      exclude: ["*self_attn*"]             # Specifies the layers to be excluded. Wildcard matching is supported. Default value: empty.
```

**Note**: SmoothQuant supports only the `norm-linear` subgraph type and does not support other subgraph types, such as `ov`, `up-down`, and `linear-linear`. Therefore, the `enable_subgraph_type` field cannot be specified.

### YAML Configuration Fields

| Field| Purpose     | Description|
|--------|---------|------|
| type | Specifies the processor type identifier.| The value is fixed to `smooth_quant`, which identifies the object as the SmoothQuant processor.|
| alpha | Specifies the balance parameter.   | This parameter is a floating-point number that controls the relative importance of activations and weights. The value is a float ranging from 0 to 1. Default value: `0.5`.|
| symmetric | Specifies whether to enable symmetric quantization. | This parameter is a boolean value. `True` enables symmetric quantization, while `False` enables asymmetric quantization. Default value: `True`.|
| include | Specifies the layers to be included. | The value is a string list. Wildcard matching is supported. Default value: ["*"], which matches all layers.|
| exclude | Specifies the layers to be excluded. | The value is a string list. Wildcard matching is supported. By default, it is not specified.|

## Model Adaptation

### Interfaces and Data Structures

```python
from dataclasses import dataclass, field
from typing import List, Optional
from abc import ABC, abstractmethod

@dataclass
class MappingConfig:
    """Module mapping configuration"""
    source: str  # Specifies the source module name, such as "model.layers.0.input_layernorm".
    targets: List[str]  # Specifies a list of target module names, such as ["model.layers.0.self_attn.q_proj", ...].

@dataclass
class AdapterConfig:
    """Subgraph adapter configuration"""
    subgraph_type: str  # Specifies the subgraph type. SmoothQuant supports only "norm-linear".
    mapping: Optional[MappingConfig] = None  # Specifies the module mapping relationships.

# Interface for adapting models to the SmoothQuant algorithm
class SmoothQuantInterface(ABC):
    @abstractmethod
    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        """
        Return all subgraph configurations within the model eligible for SmoothQuant processing.

        Returns:
            List[AdapterConfig]: A list of subgraph configurations, each containing:
                - subgraph_type: indicates the subgraph type (must be "norm-linear")
                - mapping: indicates the mapping from the source module to the target modules.
        """
        pass
```

### Adaptation Procedure

**Prerequisites**

- The model must inherit the `SmoothQuantInterface`.
- Module names must exactly match the full paths returned by `named_modules()`.
- SmoothQuant supports only the `norm-linear` subgraph type.
- The `subgraph_type` and `mapping` parameters are mandatory in the configuration.

**Procedure**

1. **Inherit the interface**: Inherit the `SmoothQuantInterface` in the model adapter and implement the `get_adapter_config_for_subgraph()` method.
2. **Configure subgraph mappings**: Configure `norm-linear` subgraph mapping relationships for each layer.
3. **Specify module paths**: Use absolute module paths, such as `model.layers.{i}.input_layernorm`.

**Reference implementation**: For details, see the `Qwen3ModelAdapter` implementation in [msmodelslim/model/qwen3/model_adapter.py](../../../../../msmodelslim/model/qwen3/model_adapter.py).

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

        adapter_config.extend([norm_linear_config1, norm_linear_config2])

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

### Incorrect Mapping Relationship

**Symptom**: The `source` and `targets` in `MappingConfig` point to incorrect modules.

**Solution**: Check whether `source` in `MappingConfig` is a normalization layer and whether `targets` are its subsequent linear layers.
