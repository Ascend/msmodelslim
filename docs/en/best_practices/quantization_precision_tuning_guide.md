# Quantization Accuracy Tuning Guide

## Overview

This document provides a systematic methodology for tuning quantization accuracy. It follows a progressive path: confirm that the accuracy issue is real, adjust the outlier suppression algorithm, adjust the quantization algorithm configuration, adjust the calibration set, and finally use quantization rollback.

**Key points:**

- **Prefer outlier suppression algorithms such as Iterative Smooth and Flex Smooth Quant to suppress activation outliers.**
- **Choose the minmax, ssz, gptq, or autoround quantization method according to the scenario.**
- **Improve accuracy further by optimizing the calibration set and rolling back sensitive layers.**

This document explains the procedures, algorithm comparisons, and configuration examples for each step so that you can deploy models efficiently with acceptable accuracy loss.

**Tuning path overview:**

Accuracy tuning is a systematic process that follows this path:

```text
Step 1: Confirm that the accuracy issue is real.
    ↓
Step 2: Adjust the outlier suppression algorithm (primary step).
    ↓
Step 3: Adjust the quantization strategy.
    ↓
Step 4: Adjust the calibration dataset.
    ↓
Step 5: Perform quantization rollback (last resort).
```

**objective:**
Deploy the model efficiently while keeping accuracy loss within an acceptable range.

## Preparation

Install msModelSlim. For details, see [msModelSlim Installation Guide](../install_guide/install_guide.md).

## Tuning Steps in Detail

### Step 1: Confirming That the Accuracy Issue Is Real

Before tuning, first rule out environmental interference and make sure the issue is real:

| Verification Item   | Operation                       |
|--------|-----------------------------|
| Inference engine verification| First evaluate the floating-point model on the target inference engine to confirm that it reproduces the original accuracy. |
| Verification result check| Inspect the evaluation output of the quantized model and confirm that there is no context truncation, timeout, or other non-quantization issue.|
| Fluctuation range determination| Understand the accuracy fluctuation range of the evaluation dataset itself and judge whether the current accuracy loss is abnormal.|

### Step 2: Adjusting the Outlier Suppression Algorithm (Primary Step)

**Core issue**: Outliers in activations can greatly expand the quantization range, consume effective quantization bits, and cause accuracy loss.

**Solution**: Use an outlier suppression algorithm to move part of the activation quantization difficulty to the weights.

[Outlier Suppression Algorithms](../user_guide/quantization_algorithms/README.md#outlier-suppression-algorithms)

#### Summary and Suggestions

1. **Prefer Iterative Smooth.** It is fast and accurate, and it is the first choice in most scenarios.
2. **Symmetry choice.** Asymmetric outlier suppression usually performs better than symmetric schemes, but you must first confirm that the inference engine already supports the asymmetric algorithm.
3. **Parameter tuning.** If Iterative Smooth does not reach the expected accuracy, tune the `alpha` parameter.
4. **Advanced options.** If accuracy still does not meet requirements, try Flex Smooth Quant or combine it with QuaRot for joint optimization. INT4 and W4A8 can use the Flex AWQ SSZ algorithm.

### Step 3: Choosing the Quantization Algorithm

Choose the appropriate algorithm according to the quantized object, either weights or activations, and the bit width. This step includes both weight quantization and activation quantization.

[Quantization Algorithms](../user_guide/quantization_algorithms/README.md#quantization-algorithms)

#### Weight Quantization

##### Configuration Example (YAML)

In the quantization configuration file, weight quantization is usually configured in `qconfig.weight` for the `linear_quant` processor. The `autoround` workflow uses a dedicated `autoround_quant` processor.

```yaml
- type: "linear_quant"         # Specifies the linear quantization processor.
  qconfig:
    weight:                    # Weight quantization configuration
      scope: "per_channel"     # Specifies the weight quantization granularity: per_channel quantization.
      dtype: "int8"            # Specifies the quantization data type. Default value: int8.
      symmetric: true          # Specifies whether to enable symmetric quantization. Default value: true.
      method: "minmax"         # Specifies the quantization method. Default value: minmax.
```

##### Summary and Suggestions

1. **INT8 weight quantization**: Prefer `minmax`, which gives the fastest speed while preserving accuracy.
2. **Low-bit weight quantization such as INT4**: Prefer `ssz`. If accuracy is not sufficient, try `autoround` or `gptq`.
3. **Quantization granularity (`scope`)**: Use per_channel (`per_channel`) weight quantization. It is finer than per_tensor (`per_tensor`) quantization and usually gives higher accuracy.
4. **Symmetry (`symmetric`)**: Weight quantization usually uses `true` for symmetric quantization, which is simpler and more efficient.
5. **AutoRound**: This offers the highest accuracy ceiling. It is suitable for scenarios that are extremely sensitive to accuracy. It runs more slowly, but its result is closest to floating point.

#### Activation Quantization

##### Activation Quantization Method Comparison

| Quantization Method| Features| Quantization Accuracy| Quantization Speed| Application Scenarios and Suggestions|
|---------|------|-----|-----|---------------|
| minmax | Uses the minimum and maximum activation values to determine the quantization range. It is simple and fast.| ★ | ★★ | **General first choice.** Suitable for most model quantization scenarios.|
| histogram | Analyzes the histogram distribution to search for the optimal clipping interval and filter activation outliers.| ★★ | ★  | Try this when the activation distribution has a long tail and minmax does not work well. It can help improve W8A8 accuracy.|
| fa3_quant | INT8 quantization at the per-head granularity for Q/K/V in the MLA architecture.| ★★ | ★★ | **Exclusive to the DeepSeek-V3/R1 series.** It adapts to distribution differences across attention heads and requires the dedicated `fa3_quant` processor.|

##### Choosing Activation Quantization Granularity

Activation quantization supports several granularities, and they directly affect accuracy and performance.

| Granularity| Features| Application Scenarios|
|---------|------|---------|
| per_tensor | Uses one set of quantization parameters for the entire tensor. This is static quantization. It is simple, fast, and supported by almost all hardware. However, if the data distribution varies significantly within the tensor, quantization error becomes noticeable.| Use when you want the best performance.|
| per_token | Uses one set of quantization parameters per token. This is dynamic quantization. It gives finer granularity and higher accuracy, but it also increases computation complexity and reduces performance.| Use when you want higher accuracy.|
| pd_mix | Uses `per_token` during prefill and `per_tensor` during decode. This is a hybrid strategy designed to balance accuracy and performance.| Use when you need to balance accuracy and performance. It helps improve W8A8 accuracy.|

##### Configuration Example (YAML)

In the quantization config file, activation quantization is usually configured in `qconfig.act` for the `linear_quant` processor:

```yaml
- type: "linear_quant"         # Specifies the linear quantization processor.
  qconfig:
    act:                       # Activation quantization configuration
      scope: "per_tensor"      # Specifies the quantization scope: per_tensor, which indicates static quantization where the entire tensor shares the same quantization parameters.
      dtype: "int8"            # Specifies the quantization data type. Default value: int8.
      symmetric: false      # Specifies whether to enable symmetric quantization. Default value: false.
      method: "minmax"         # Specifies the quantization method. Default value: minmax.
    weight:                    # Weight quantization configuration
      scope: "per_channel"     # Specifies the weight quantization granularity: per_channel quantization.
      dtype: "int8"            # Specifies the quantization data type. Default value: int8.
      symmetric: true          # Specifies whether to enable symmetric quantization. Default value: true.
      method: "minmax"         # Specifies the quantization method. Default value: minmax.
```

##### Summary and Suggestions

1. Quantization method: Prefer `minmax`. If accuracy is not enough, try `histogram`.
2. **Quantization granularity (`scope`)**:
   - Use `per_tensor` (static quantization) for performance.
   - Use `per_token` (dynamic quantization) for accuracy.
   - Use `pd_mix` (hybrid policy) when you need to balance accuracy and performance.
3. **Symmetry (`symmetric`)**: Activation quantization usually uses `false` for asymmetric quantization so that it better fits non-zero-centered data distributions.
4. `pd_mix`: This balanced strategy aims to trade off activation quantization accuracy and computation performance during inference.

### Step 4: Adjusting the Calibration Set

When algorithm tuning has limited effect, improve the quantized model accuracy by optimizing the calibration data. The quality of the calibration set directly affects the accuracy of the quantization parameters.

#### Calibration Set Optimization Strategy

| Adjustment Strategy| Operation| Optimization Objective|
|---------|---------|---------|
| Increase the amount of data.| Increase the data volume appropriately. 10 to 50 samples is recommended.| Improve the accuracy of quantization parameter estimation.|
| Match the application scenario.| Use data that matches the model application scenario. For example, use Chinese data for a Chinese model and code data for a code model.| Make the calibration data closer to the real application scenario.|
| Balance the data distribution.| Sample from multiple datasets and mix them together to balance the data distribution.| Improve the diversity and balance of the data distribution.|
| Remove abnormal data.| Remove calibration samples that cause a noticeable accuracy drop.| Reduces the interference of abnormal samples on quantization parameters.|
| Add bad cases.| Add bad-case samples from the model on that dataset so the calibration data better reflects the true input distribution.| Help the quantized model learn hard samples and improve accuracy.|

#### Summary and Suggestions

1. **Data volume**: Use 10 to 50 samples. Too few samples may not estimate the quantization parameters well, and too many may increase quantization time.
2. **Scenario matching**: Prefer data that matches the model application scenario so the calibration set can represent real usage.
3. **Data quality**: Remove abnormal data in time to avoid negative effects on quantization parameters.
4. **Hard samples**: Add bad-case samples as needed. This can improve quantization accuracy on difficult samples.

### Step 5: Perform Quantization Rollback (Last Resort)

When algorithm and calibration set tuning still cannot meet the accuracy target, roll back the most sensitive layers so they stay at high precision, either FP16 or BF16, to recover the accuracy loss caused by quantization.

**Use cases**:

- Accuracy still does not meet the target after steps 1 to 4.
- You need a finer balance between accuracy and performance.
- Certain layers are extremely sensitive to quantization and must stay at high precision.

#### Operation Process

##### Step 1: Analyzing Sensitive Layers

Use the sensitive layer analysis tool provided by msModelSlim to identify quantization-sensitive layers. For details, see [Quantization Sensitive Layer Analysis Guide](../user_guide/feature_guide/sensitive_layer_analysis/usage.md).

**Function description**:

- **Automatic evaluation**: The tool automatically evaluates the sensitivity of linear layers to quantization and generates a sensitivity score for each analyzable object.
- **Decision basis**: Use the generated sensitivity scores to decide which high-sensitivity layers should be rolled back.

##### Step 2: Configuring Rollback

Based on the analysis result from step 1, exclude the highly sensitive layers in the YAML config for the quantization algorithm configuration by using the `exclude` field.

#### Configuration Example

```yaml
apiversion: modelslim_v1       # Specifies the protocol version.

spec:
  process:                     # Specifies the processor list.
    - type: "linear_quant"     # Specifies the processor type: linear layer quantization.
      qconfig:
        act:                   # Activation quantization configuration
          scope: "per_tensor"  # Specifies the quantization scope: per_tensor, which indicates static quantization and shares quantization parameters on the entire tensor.
          dtype: "int8"        # Specifies the quantization data type. Default value: int8.
          symmetric: false     # Specifies whether to enable symmetric quantization. Default value: false.
          method: "minmax"     # Specifies the quantization method. Default value: minmax.
        weight:                # Weight quantization configuration
          scope: "per_channel" # Specifies the weight quantization granularity: per_channel quantization.
          dtype: "int8"        # Specifies the quantization data type. Default value: int8.
          symmetric: true      # Specifies whether to enable symmetric quantization. Default value: true.
          method: "minmax"     # Specifies the quantization method. Default value: minmax.
      include: ["*"]            # Specifies the layers to be included. Wildcard matching is supported. Default value: ["*"].
      exclude: ["*model.layers.*.mlp.down_proj*"] # Specifies layers to exclude. Default: []. Rolls back all `mlp.down_proj` layers here.
```

#### Summary and Suggestions

1. **Prefer rolling back `mlp.down_proj`.** Based on experience, `mlp.down_proj` is often one of the most quantization-sensitive layers, so it should be the first rollback candidate.
2. **Balance the cost.** Rollback reduces some of the performance and memory savings from quantization, so decide the rollback scope based on your business goal.
3. **Choose the rollback strategy.** Start from the most sensitive layers and roll them back step by step until you find the best balance between accuracy and performance.
