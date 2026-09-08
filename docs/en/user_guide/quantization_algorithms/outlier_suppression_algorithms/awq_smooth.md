# AWQ: Activation-aware Weight Quantization Algorithm

## Overview

- **Description**: Activation-aware Weight Quantization (AWQ) is an algorithm used to suppress activation outliers during large language model quantization. By observing the statistical characteristics of activation values, this algorithm automatically searches for the optimal scaling factor and scales the weights prior to weight quantization, effectively reducing quantization error while maintaining model accuracy. The core philosophy of AWQ is that not all weights are equally important to the model output. By identifying and protecting critical weight channels based on the activation value distribution, the algorithm achieves superior accuracy performance in low-bit quantization scenarios.
- **Core idea**: The AWQ algorithm evaluates the importance of each weight channel by calculating the mean value of the activations. The tool then performs a grid search to determine the optimal scaling factor that minimizes the mean squared error (MSE) between the quantized output and the floating-point baseline result.

## Preparations

Install msModelSlim. For details, see [msModelSlim Installation Guide](../../../install_guide/install_guide.md).

## Principle and Implementation

### Principle

**Formula**

```python
scales = act_mean.pow(ratio).clamp(min=1e-4)
scales = scales / sqrt(scales.max() * scales.min())
```

where

- `act_mean` is the channel-wise mean value of the absolute activation values (`mean(abs(act))`), which reflects the importance of each channel.
- `ratio` is the scaling ratio factor, which is determined via a grid search over the range `[0, 1)` with a step size of `1 / n_grid`.
- `n_grid` is the number of grid search steps, which defaults to `20` and controls the search accuracy.

**Key Features**

1. **Importance evaluation based on activation mean values**: The tool measures the importance of each weight channel using the channel-wise mean value of the absolute activation values. Channels with larger mean values receive more protection during quantization.
2. **Grid search for optimal scaling**: The tool traverses different `ratio` values within the `[0, 1)` range to evaluate the quantization effect of each candidate scaling factor.
3. **Real quantizer evaluation**: The tool uses the actual weight quantizer to quantize the scaled weights. It then selects the optimal parameters based on the mean squared error between the quantization output and the floating-point baseline result.
4. **Block-level error evaluation**: Instead of only comparing the weight error of a single linear layer, the tool evaluates the quantization error at the block level by automatically discovering the lowest common ancestor (LCA) of the target module and caching its input parameters.

**Scaling Factor Search Process**

1. **Initialization**: Collect the activation value mean and the input parameter cache of the ancestor module.
2. **Benchmark inference**: Perform inference on the ancestor module by using the original floating-point weights to obtain the floating-point benchmark output.
3. **Grid search**: Traverse `ratio` values within the `[0, 1)` range.
   - Calculate the scaling factor: `scales = act_mean.pow(ratio).clamp(min=1e-4)`.
   - Normalize the scaling factor: `scales = scales / sqrt(scales.max() * scales.min())`.
   - Apply the scaling factor to the target linear layer weights.
   - Use the quantizer to quantize the scaled weights.
   - Perform inverse scaling on the quantized weights.
   - Perform inference on the ancestor module and calculate the mean squared error.
   - Restore the original weights.
4. **Select optimal parameters**: Select the scaling factor corresponding to the `ratio` that yields the minimum mean squared error.
5. `Apply scaling`: Integrate the optimal scaling factor into the subgraph weights by using the `SubgraphFusionFactory` interface.

### Supported Subgraph Types

#### NormLinearSubgraph

This type applies to structures containing a normalization layer followed by multiple linear layers, such as:

```python
x = norm(x)
y = torch.cat([linear(x) for linear in linears], dim=-1)
```

Processing workflow:

- The tool uses all target linear layers to search for the scaling factor.
- The tool evaluates block-level errors by using the automatically discovered lowest common ancestor module.
- After identifying the optimal scaling factor, the tool applies inverse scaling to the normalization layer and forward scaling to the linear layers.

#### LinearLinearSubgraph

This type applies to structures containing a normalization layer followed by multiple linear layers, such as:

```python
y = linear2(linear1(x))
```

Processing workflow:

- The tool searches for the scaling factor based on the `linear2` weights.
- The tool evaluates block-level errors by using the automatically discovered lowest common ancestor module.
- The tool applies forward scaling to `linear2` and inverse scaling to `linear1`.

#### `OVSubgraph` (Attention Output-Value Subgraph)

This subgraph applies to output projections and value projections within the attention mechanism, supporting the following architectures:

- Multi-head attention (MHA)
- Multi-query attention (MQA)
- Grouped-query attention (GQA)

Processing workflow:

- The tool searches for the scaling factor based on the `o_proj` weights.
- The tool evaluates block-level errors by using the automatically discovered lowest common ancestor module.
- The tool applies forward scaling to `o_proj` and inverse scaling to `v_proj`.

#### UpDownSubgraph (Up Projection – Down Projection Subgraph)

This subgraph applies to Multi-Layer Perceptron (`MLP`) gating architectures, such as:

```python
y = down_proj(ReLU(gate_proj(x)) * up_proj(x))
```

Processing workflow:

- The tool searches for the scaling factor based on the `down_proj` weights.
- The tool evaluates block-level errors by using the automatically discovered lowest common ancestor module.
- The tool applies forward scaling to `down_proj` and inverse scaling to `up_proj`.

### Implementation

#### Code Implementation

The code of the AWQ algorithm is organized in the [msmodelslim/processor/anti_outlier/awq/](../../../../../msmodelslim/processor/anti_outlier/awq/__init__.py) directory.

| File                                                                                               | Core Class/Function                                       | Role                                  |
| --------------------------------------------------------------------------------------------------- | -------------------------------------------------- | -------------------------------------- |
| [processor.py](../../../../../msmodelslim/processor/anti_outlier/awq/processor.py)                     | `AWQProcessor`, `AWQProcessorConfig`               | Processor entry point, which manages preprocessing and postprocessing workflows.|
| [api.py](../../../../../msmodelslim/processor/anti_outlier/awq/api.py)                                 | `awq()`                                            | AWQ dispatching and implementation for each subgraph type.           |
| [best_scales_search.py](../../../../../msmodelslim/processor/anti_outlier/awq/best_scales_search.py)   | `AWQSearcher`, `AWQBestScalesSearcher`             | Scaling factor grid search logic.                  |
| [awq_stats_collector.py](../../../../../msmodelslim/processor/anti_outlier/awq/awq_stats_collector.py) | `AWQStatsCollector`                                | Activation statistics collection and intermediate parameter caching.        |
| [common.py](../../../../../msmodelslim/processor/anti_outlier/awq/common.py)                           | `AWQConfig`, `AWQContext`, `offload()`, `onload()` | Algorithm configuration, runtime context, and tensor migration tools.  |
| [interface.py](../../../../../msmodelslim/processor/anti_outlier/awq/interface.py)                     | `AWQInterface`                                     | Abstract interface to be implemented by the model adapter.          |

#### Preprocessing

- The tool obtains the subgraph configuration through `AWQInterface.get_adapter_config_for_subgraph()`.
- The tool filters the target subgraphs based on the `include` and `exclude` configurations.
- The tool installs a forward hook for each target linear layer to collect the channel-wise mean of the absolute activation values.
- The tool automatically discovers the ancestor module for block-level evaluation through `LCA`, installs a forward pre-hook for it, and caches the input parameters.

#### Postprocessing

- The tool processes subgraphs based on the following priority order: `up-down`, `ov`, `norm-linear`, and `linear-linear`.
- The tool constructs the `AWQContext` from the collected statistics, including the activation mean, ancestor module instances, and the input parameter cache.
- The tool calls the `AWQBestScalesSearcher.search()` method to search for the optimal scaling factor.
- The tool fuses the optimal scaling factor into the subgraph through the `SubgraphFusionFactory` interface.
- The tool stops all hooks and clears the statistics.

## Application Requirements

- **Model interface**: The model adapter must implement the `AWQInterface` interface.
- **Module naming**: The module names in the configuration must match the full paths returned by the `named_modules()` method.
- **Subgraph type**: The supported values for the `enable_subgraph_type` parameter are `norm-linear`, `linear-linear`, `ov`, and `up-down`.
- **Module attribute**: The target module must exist and possess a writable `weight` attribute.
- **Runtime**: AWQ depends on `ContextManager` to provide the global context. The tool automatically creates and manages the related context objects at runtime.

## Function

### YAML Configuration Example

The following example shows a YAML configuration when the algorithm is used as a processor:

```yaml
- type: "awq"
  weight_qconfig:
    scope: "per_channel"
    dtype: "int8"
    symmetric: true
    method: "minmax"
  n_grid: 20
  enable_subgraph_type:
    - "norm-linear"
    - "linear-linear"
    - "ov"
    - "up-down"
  include:
    - "*"
  exclude: []
```

### YAML Configuration Fields

| Field                | Purpose          | Description                                                                                                                                                     |
| ---------------------- | -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `type`                 | Specifies the processor type identifier.| The value is fixed to `"awq"`.                                                                                                                                       |
| `weight_qconfig`       | Specifies the weight quantization configuration.  | The value is the weight quantization configuration used during the AWQ search phase. The field definitions match the `qconfig.weight` parameter described in [Linear Quantization Algorithm](../quantization_algorithms/linear_quant.md#yaml-configuration-fields).|
| `n_grid`               | Specifies the number of grid search steps.  | The value must be a positive integer, and defaults to `20`. A larger value improves search accuracy but increases execution time.                                                                                              |
| `enable_subgraph_type` | Specifies the enabled subgraph types.| The supported values are `norm-linear`, `linear-linear`, `ov`, and `up-down`.                                                                                         |
| `include`              | Specifies the layers to be included.      | Wildcard matching is supported.                                                                                                                                         |
| `exclude`              | Specifies the layers to be excluded.      | Wildcard matching is supported, and this field has a higher priority than `include`.                                                                                                                   |

### Model Adaptation

#### Interfaces and Data Structures

The AWQ model adaptation depends on the following interface and data structures:

```python
from dataclasses import dataclass
from typing import List, Optional
from abc import ABC, abstractmethod

@dataclass
class MappingConfig:
    targets: List[str]
    source: Optional[str] = None

@dataclass
class AdapterConfig:
    subgraph_type: str
    mapping: MappingConfig

class AWQInterface(ABC):
    @abstractmethod
    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        ...
```

#### Adaptation Procedure

1. Ensure that the model adapter inherits from `AWQInterface` and implements the `get_adapter_config_for_subgraph()` method.
2. Configure `subgraph_type` and `mapping` properties for the standard subgraphs within the model.
3. Provide the full module paths when defining the `source` and `targets` parameters.
4. The tool automatically discovers the `LCA` and caches the input parameters required for block-level evaluation.

For details about the reference implementation, see [msmodelslim/model/qwen2/model_adapter.py](../../../../../msmodelslim/model/qwen2/model_adapter.py).

#### Configuration Example

The following example demonstrates the AWQ subgraph mapping configuration for a typical Transformer layer:

```python
def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
    adapter_config = []
    for layer_idx in range(self.config.num_hidden_layers):
        norm_linear_mapping_config1 = MappingConfig(
            source=f"model.layers.{layer_idx}.input_layernorm",
            targets=[
                f"model.layers.{layer_idx}.self_attn.k_proj",
                f"model.layers.{layer_idx}.self_attn.q_proj",
                f"model.layers.{layer_idx}.self_attn.v_proj",
            ],
        )

        norm_linear_mapping_config2 = MappingConfig(
            source=f"model.layers.{layer_idx}.post_attention_layernorm",
            targets=[
                f"model.layers.{layer_idx}.mlp.gate_proj",
                f"model.layers.{layer_idx}.mlp.up_proj",
            ],
        )

        ov_mapping_config = MappingConfig(
            source=f"model.layers.{layer_idx}.self_attn.v_proj",
            targets=[f"model.layers.{layer_idx}.self_attn.o_proj"],
        )

        up_down_mapping_config = MappingConfig(
            source=f"model.layers.{layer_idx}.mlp.up_proj",
            targets=[f"model.layers.{layer_idx}.mlp.down_proj"],
        )

        adapter_config.extend([
            AdapterConfig(subgraph_type="norm-linear", mapping=norm_linear_mapping_config1),
            AdapterConfig(subgraph_type="norm-linear", mapping=norm_linear_mapping_config2),
            AdapterConfig(subgraph_type="ov", mapping=ov_mapping_config),
            AdapterConfig(subgraph_type="up-down", mapping=up_down_mapping_config),
        ])

    return adapter_config
```

### Differences from Flex AWQ SSZ

| Feature        | AWQ                                   | [Flex AWQ SSZ](flex_awq_ssz.md)       |
| ------------ | ------------------------------------- | ------------------------------------- |
| Scaling factor calculation| `act_mean.pow(ratio)` grid search       | `A_scale**alpha / W_scale**beta`      |
| Error evaluation method| Block-level evaluation                             | Uses a quantizer to evaluate different parameter combinations           |
| Parameter search space| `ratio` ∈ `[0, 1)`, step size `1 / n_grid`| `alpha` ∈ `[0, 1]`, `beta` is usually set to `0`|
| Configuration interfaces    | `AWQInterface`                        | `FlexSmoothQuantInterface`            |
| Quantization configurations    | Requires only the weight quantization configuration.                     | Requires both the activation and weight quantization configurations.               |

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

**Solution**: Set this parameter to a supported subgraph type that has been adapted to your model. The supported values are `norm-linear`, `linear-linear`, `ov`, and `up-down`. Unless you have specific requirements, retain the default configuration.

### Ancestor Module Not Found

**Symptom**: The log displays "No name found for inspect module of subgraph", and the tool skips the target subgraph.

**Solution**: Check whether the module names in the `targets` parameter share a common path prefix. Ensure that their lowest common ancestor module exists within the model architecture.

### Missing Activation Statistics

**Symptom**: The log displays "No activation mean for target module", and the tool skips the target subgraph.

**Solution**: Ensure that your calibration data is sufficient and that model forward inference runs normally, allowing the hook to collect activation statistics correctly.

### Empty Intermediate Parameter Cache

**Symptom**: The log displays "No kwargs cache for parent module", and the tool skips the target subgraph.

**Solution**: Ensure that the ancestor module automatically discovered through `LCA` is correctly triggered during forward inference, and verify that the module paths specified in `targets` are correct.
