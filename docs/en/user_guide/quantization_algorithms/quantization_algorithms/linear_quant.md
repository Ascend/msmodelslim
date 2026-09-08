# Linear Quantization Algorithm

## Overview

Linear quantization is the most fundamental and widely used algorithm category in deep learning model compression.

In msModelSlim, linear quantization is implemented by using the `linear_quant` processor, which is dedicated to quantizing the linear layers of a model (such as `torch.nn.Linear`). It supports flexible quantization configurations for the weights and activation values of linear layers.

## Preparations

Install msModelSlim. For details, see [msModelSlim Installation Guide](../../../install_guide/install_guide.md).

## Principle and Implementation

### Principle

The core of linear quantization is the mapping of a continuous range of floating-point values to a discrete set of values. The basic formula is as follows:

$$Q = \text{clamp}(\text{round}(\frac{V}{S}) + Z, Q_{min}, Q_{max})$$

where

- $V$: original floating-point value
- $S$ (Scale): scaling factor, which determines the quantization step
- $Z$ (Zero-point): offset (zero point) used to handle asymmetric distributions
- $Q$: quantized value

### Implementation

In msModelSlim, linear quantization is implemented through the `linear_quant` processor, which supports flexible quantization configurations for linear or dense layers of a model. The algorithm is implemented in [msmodelslim/processor/quant/linear.py](../../../../../msmodelslim/processor/quant/linear.py).

### Algorithm Classification

Based on parameter statistics and computation timing, linear quantization is classified into the following types:

1. **Static Quantization**
   - **Principles**: Before inference, the activation value distribution is collected using a calibration dataset to calculate and fix the scale and zero point.
   - **Advantages**: Deliver optimal performance because quantization parameters do not need to be dynamically calculated during inference.
   - **Application scenarios**: Production environments that are extremely sensitive to inference latency.

2. **Dynamic Quantization**
   - **Principles**: During inference, the scale and zero point of activations are dynamically calculated for each input group (such as each token or batch).
   - **Advantages**: This method can adapt to distribution changes in input data. The accuracy is usually significantly higher than that of static quantization.
   - **Application scenarios**: Scenarios that require high accuracy and where the activation value distribution of the model changes drastically with the input (such as large language models).

3. **Mixed/Hybrid Quantization**
   - **Principle**: This method combines the benefits of static and dynamic quantization. For example, the `PDMIX` algorithm uses dynamic quantization in the prefilling phase to ensure the accuracy of the first token and uses static quantization in the decoding phase to improve generation speed.

## Function Description

### Supported Quantization Algorithms

In the `linear_quant` processor, you can select different quantization algorithms by setting the `method` parameter.

| Algorithm Name| Application Scenario| Brief Description| Detailed Description|
| :--- | :--- | :--- | :--- |
| **MinMax** | Weight/Activation quantization| Determines the quantization range by determining the maximum and minimum values. As the most fundamental quantization algorithm, it is simple, efficient, and the primary choice for INT8 scenarios.| [MinMax Quantization Algorithm](minmax.md)|
| **Histogram** | Activation quantization| Analyzes the activation value distribution through histograms and automatically truncates outliers to optimize the quantization range. The accuracy is usually better than that of MinMax.| [Histogram: Activation Quantization Algorithm](histogram_activation_quantization.md)|
| **SSZ** | Weight quantization| Minimizes quantization error by computing the optimal scaling factor through iterative search. It is specifically designed for low-bit weight quantization (such as INT4).| [SSZ: Weight Quantization Algorithm](ssz.md)|
| **GPTQ** | Weight quantization| Minimizes quantization error by using a column-wise optimization method to compensate for errors in subsequent unquantized weights.| [GPTQ: Weight Quantization Algorithm](gptq.md)|

### Core Differences Between Static and Dynamic Quantization

In quick quantization, the `qconfig.act.scope` field distinguishes between **static** and **dynamic** quantization:

- **Static quantization (`per_tensor`)**: calculates and fixes quantization parameters (`scale` and `offset`) during the quantization calibration phase for use during inference. **Characteristics**: This mode provides optimal inference performance and minimum computational overhead. However, accuracy may be compromised when the distribution changes drastically.
- **Dynamic quantization (`per_token`)**: calculates quantization parameters in real time for each token during inference. **Characteristics**: This mode provides a finer quantization granularity and can better capture the dynamic distribution of activation values. Generally, the **accuracy of dynamic quantization is higher than that of static quantization**, but this mode introduces some real-time computational overhead.
- **PDMIX hybrid quantization (`pd_mix`)**: uses `per_token` in the prefilling phase and `per_tensor` in the decoding phase. **Characteristics**: This mode aims to balance accuracy and performance and is especially suitable for accelerating the inference of generative models. For details, see [PDMIX: Activation Phase Hybrid Quantization Algorithm](pdmix.md).

### YAML Configuration Example

#### W8A8 Static Quantization Configuration

```yaml
- type: "linear_quant"         # Specifies the linear quantization processor.
  qconfig:
    act:                       # Activation quantization configuration
      scope: "per_tensor"      # Specifies the quantization scope: per_tensor, which indicates static quantization where the entire tensor shares the same quantization parameters.
      dtype: "int8"            # Specifies the quantization data type: int8.
      symmetric: false         # Disables symmetric quantization (performs asymmetric quantization, which is recommended for static quantization).
      method: "minmax"         # Specifies the quantization algorithm: minmax.
    weight:                    # Weight quantization configuration
      scope: "per_channel"     # Specifies the weight quantization granularity: per_channel quantization.
      dtype: "int8"            # Specifies the quantization data type: int8.
      symmetric: true          # Specifies whether to enable symmetric quantization: true.
      method: "minmax"         # Specifies the quantization algorithm: minmax.
  include: [ "*" ]             # Specifies the layers to be included. Default value: ["*"].
  exclude: [ "*down_proj*" ]   # Specifies the layers to be excluded. Default value: [].
```

#### W8A8 Dynamic Quantization Configuration

```yaml
- type: "linear_quant"         # Specifies the linear quantization processor.
  qconfig:
    act:                       # Activation quantization configuration
      scope: "per_token"       # Specifies the quantization scope: per_token, which indicates dynamic quantization and uses independent quantization parameters for each token.
      dtype: "int8"            # Specifies the quantization data type: int8.
      symmetric: false         # Specifies whether to enable symmetric quantization: false.
      method: "minmax"         # Specifies the quantization algorithm: minmax.
    weight:                    # Specifies the weight quantization configuration.
      scope: "per_channel"     # Specifies the weight quantization granularity: per_channel quantization.
      dtype: "int8"            # Specifies the quantization data type: int8.
      symmetric: true          # Specifies whether to enable symmetric quantization: true.
      method: "minmax"         # Specifies the quantization algorithm: minmax.
  include: [ "*mlp*" ]         # Includes only layers containing MLP.
```

#### W4A8 Dynamic Quantization Configuration

```yaml
- type: "linear_quant"         # Specifies the linear quantization processor.
  qconfig:
    act:                       # Activation quantization configuration
      scope: "per_token"       # Specifies the quantization scope: per_token, which indicates dynamic quantization and uses independent quantization parameters for each token.
      dtype: "int8"            # Specifies the quantization data type: int8.
      symmetric: true          # Specifies whether to enable symmetric quantization: true.
      method: "minmax"         # Specifies the quantization algorithm: minmax.
    weight:                    # Specifies the weight quantization configuration.
      scope: "per_channel"     # Specifies the weight quantization granularity: per_channel quantization.
      dtype: "int4"            # Specifies the data type: int4 (low-bit weight quantization).
      symmetric: true          # Specifies whether to enable symmetric quantization: true.
      method: "ssz"            # Specifies the quantization algorithm: ssz.
  include: [ "*" ]             # Includes all layers.
```

#### W8A8 PDMIX Configuration

```yaml
- type: "linear_quant"         # Specifies the linear quantization processor.
  qconfig:
    act:                       # Activation quantization configuration
      scope: "pd_mix"          # Specifies the PDMIX quantization identifier. pd_mix uses per_token in the Prefilling phase, and per_tensor in the Decoding phase.
      dtype: "int8"            # Specifies the quantization data type: int8.
      symmetric: false         # Specifies whether to enable symmetric quantization: false.
      method: "minmax"         # Specifies the quantization algorithm: minmax.
    weight:                    # Specifies the weight quantization configuration.
      scope: "per_channel"     # Specifies the weight quantization granularity: per_channel quantization.
      dtype: "int8"            # Specifies the quantization data type: int8.
      symmetric: true          # Specifies whether to enable symmetric quantization: true.
      method: "minmax"         # Specifies the quantization algorithm: minmax.
  include: [ "*mlp*" ]         # Includes only layers containing MLP.
```

### YAML Configuration Fields

| Field| Purpose| Type| Description| Example Value|
|--------|------|------|------|--------|
| type | Specifies the processor type identifier.| `string` | Fixed value used to identify a linear layer quantization processor.| `"linear_quant"` |
| qconfig | Specifies the quantization configuration parameters.| `object` | This section contains parameters for activation quantization and weight quantization.| See the detailed configuration below.|
| include | Specifies the layers to be included.| `array[string]` | Wildcard matching is supported to specify layers for quantization.| `["*"]`, `["*self_attn*"]` |
| exclude | Specifies the layers to be excluded.| `array[string]` | Wildcard matching is supported, and this field has a higher priority than `include`.| `["*down_proj*"]` |

#### `qconfig.act` (Activation Quantization Configuration)

**Purpose**: Configures quantization parameters for activation values.

| Field| Purpose| Optional Value                                      | Description                                                                                                              | Default Value|
|--------|------|-------------------------------------------|------------------------------------------------------------------------------------------------------------------|--------|
| scope | Specifies the quantization scope.| `"per_tensor"`, `"per_token"`, `"pd_mix"` | `per_tensor`: The entire tensor uses the same parameters.<br>`per_token`: Each token uses independent parameters (dynamic quantization).<br>`pd_mix`: `per_token` is used during prefilling and `per_tensor` during decoding.| `"per_tensor"` |
| dtype | Specifies the quantization data type.| `"int8"`, `"int4"`, `"float"` | `int8`: 8-bit integer quantization.<br>`int4`: 4-bit integer quantization.<br>`float`: 16-bit floating-point activation quantization. In this case, set `scope` to `per_tensor`, `symmetric` to `true`, and `method` to `none`.| `"int8"` |
| symmetric | Specifies whether to enable symmetric quantization.| `true`, `false` | `true`: enables symmetric quantization, with a zero point of `0`.<br>`false`: enables asymmetric quantization, with an adjustable zero point.| `false` |
| method | Specifies the quantization method.| `"minmax"`, `"histogram"` | `minmax`: specifies MinMax quantization.<br>`histogram`: specifies histogram activation quantization.| `"minmax"` |

#### `qconfig.weight` (Weight Quantization Configuration)

**Purpose**: Configures quantization parameters for weights.

| Field| Purpose| Optional Value| Description| Default Value|
|--------|------|--------|------|--------|
| scope | Specifies the quantization scope.| `"per_tensor"`, `"per_channel"`, `"per_group"` | `per_tensor`: The entire tensor uses the same parameters.<br>`per_channel`: Each channel uses independent parameters.<br>`per_group`: Each group uses independent parameters.| `"per_channel"` |
| dtype | Specifies the quantization data type.| `"int8"`, `"int4"` | This parameter specifies 8-bit or 4-bit integer quantization.| `"int8"` |
| symmetric | Specifies whether to enable symmetric quantization.| `true`, `false` | `true`: enables symmetric quantization, with a zero point of `0`.<br>`false`: enables asymmetric quantization, with an adjustable zero point.| `true` |
| method | Specifies the quantization method.| `"minmax"`, `"ssz"`, `"gptq"` | `minmax`: specifies MinMax quantization.<br>`ssz`: specifies SSZ weight quantization.<br>`gptq`: specifies GPTQ weight quantization.| `"minmax"` |

### Layer Filtering Mechanism

#### Filtering Rules

1. **`include`**: defines the layers to be included. Only layers matching the `include` pattern are processed.
2. **`exclude`**: defines the layers to be excluded. Layers matching the `exclude` pattern are skipped.
3. **Precedence**: `exclude` rules take precedence over `include` rules.

#### Pattern Matching (Unix Wildcards)

| Wildcard| Purpose| Example|
|--------|------|------|
| `*` | Matches any sequence of characters.| `*self_attn*` matches any string containing "self_attn".|
| `?` | Matches any single character.| `layer?` Matches layer1, layerA, and so on.|
| `[abc]` | Matches any single character in the character set.| `layer[123]` matches layer1, layer2, or layer3.|
| `[!abc]` | Matches any character that is not in the character set.| `layer[!123]` matches any single string except for layer1, layer2, and layer3.|

#### Filtering Order

1. **Step 1**: Check whether the layer name matches the `include` pattern.
    - If `include` is empty or not set, all layers are included by default.
    - If the layer name does not match any `include` pattern, the layer is excluded.
2. **Step 2**: Check whether the layer name matches the `exclude` pattern.
    - If the layer name matches any `exclude` pattern, this layer is excluded
    - even if it matches any `include` pattern in Step 1.

#### Examples

**Example 1**: Basic filtering

```yaml
include: [ "*" ]
exclude: [ "*down_proj*" ]
```

- **Result**: All layers are included, except for those with `down_proj` in their names.

**Example 2**: Selective inclusion

```yaml
include: [ "*self_attn*", "*mlp*" ]
exclude: [ ]
```

- **Result**: Only layers with `self_attn` or `mlp` in their names are included.

**Example 3**: Complex filtering

```yaml
include: [ "*attention*", "*mlp*" ]
exclude: [ "*down_proj*", "*gate*" ]
```

- **Result**: Layers with `attention` or `mlp` in their names are included, except for those whose names contain `down_proj` or `gate`.

**Example 4**: Fine-grained filtering

```yaml
include: [ "model.layers.*.self_attn.*" ]
exclude: [ "model.layers.*.self_attn.down_proj" ]
```

- **Result**: Only layers with `self_attn` in their names are included, excluding their `down_proj` sub-layers.

#### Common Layer Name Patterns

**Common Layer Names in the Transformer Architecture**

| Pattern| Description| Typical Usage|
|------|------|----------|
| `*self_attn*` | Self-attention layer| Weights and biases for the attention mechanism.|
| `*mlp*` | Multi-layer perceptron (MLP) layer| Multi-layer perceptron layer|
| `*attention*` | Attention-related layer| Generalized attention-related components|
| `*ffn*` | Feed-forward network (FFN) layer| Feed forward network|
| `*gate*` | Gating layer| Layers related to the gating mechanism|
| `*down_proj*` | Down-projection layer| Dimension-reduction projection layer|
| `*up_proj*` | Up-projection layer| Dimension-expansion projection layer|

**Quantization Strategy Suggestions**

| Strategy| Configuration| Description|
|------|------|------|
| Full quantization| `include: ["*"]` | Quantizes all linear layers.|
| Attention layer quantization| `include: ["*self_attn*"]` | Quantizes only self-attention related layers.|
| MLP layer quantization| `include: ["*mlp*"]` | Quantizes only multi-layer perceptron layers.|
| Sensitive layer exclusion| `exclude: ["*down_proj*", "*gate*"]` | Excludes layers sensitive to precision.|

## FAQ

### Quantization Configuration Combination Validity

**Symptom**: Not all quantization configuration combinations are valid. When an invalid quantization configuration combination is detected, the system throws an `UnsupportedError` exception.

**Solution**: The exception message provides details regarding the specific cause of the configuration conflict. Adjust the configuration parameters based on the information provided in the exception.

### Layer Matching Alarms

**Symptom**: The tool issues an alarm when the `include` and `exclude` patterns do not match any layers. Pay close attention to these alarm messages.

**Common Causes of Matching Failures**

1. **Layer name mismatch**

   ```yaml
   # ❌ Possible matching failure: The layer name does not contain "self_attn".
   include: ["*self_attn*"]
   # The actual layer name might be "attention", "attn", or "self_attention".
   ```

2. **Incorrect path hierarchy**

   ```yaml
   # ❌ Possible matching failure: The path level does not match.
   include: ["model.layers.*.attention"]
   # The actual path might be "layers.*.attention" or "transformer.layers.*.attention".
   ```

3. **Case sensitivity**

   ```yaml
   # ❌ Possible matching failure: The case does not match.
   include: ["*SelfAttn*"]
   # The actual layer name might be "*self_attn*".
   ```

4. **Spelling error**

   ```yaml
   # ❌ Possible matching failure: There is a spelling error.
   include: ["*self_atttn*"]  # Spelling error: There is an extra "t".
   # The actual layer name might be "*self_attention*".

   include: ["*mlp*"]  # Possible matching failure: Different models use different naming conventions.
   # The actual layer name might be "*ffn*" or "*feed_forward*".
   ```

5. **Non-nn.Linear path**

   ```yaml
   # ❌ Possible matching failure: The pattern matches a non-linear layer.
   include: ["*gate.weight"]
   # The actual path is nn.Parameter.
   # LinearQuantProcess processes only the nn.Linear layers. Other layers are ignored.
   ```

**Solution**

1. **Check the layer names**: Use a model analysis tool to view the actual layer structure.
2. **Verify the patterns**: Use a simple wildcard pattern for testing.
3. **Debug step by step**: Start with `include: ["*"]` and gradually narrow the scope.
4. **Review the logs**: Focus on the matching results and alarm information issued by the tool.
    - Matching failure alarm: The message `patterns are not matched any module, please ensure this is as expected` is displayed.
