---
toc_depth: 3
---
# Quantization Algorithm Overview

msModelSlim supports multiple advanced quantization algorithms, covering all aspects from outlier suppression to low-bit optimization. The following tables summarize the currently supported core algorithms and their primary characteristics by category.

## Outlier Suppression Algorithms

Outlier suppression algorithms aim to smooth the distribution of activation values and reduce the accuracy drop caused by quantization.

| Algorithm| Core Idea| Application Scenario| Description|
| :--- | :--- | :--- | :--- |
| **QuaRot** | Uses an orthogonal rotation matrix to smooth the distribution of activation values.| Scenarios requiring activation outlier suppression to improve model accuracy| [View details](outlier_suppression_algorithms/quarot.md)|
| **Adapt Rotation** | Implements calibration-data-based iterative optimization of the Hadamard rotation matrix on top of QuaRot.| Scenarios where the rotation matrix must be optimized to further improve low-bit quantization accuracy| [View details](outlier_suppression_algorithms/adapt_rotation.md)|
| **SmoothQuant** | Smooths activation and weight scaling to mitigate the impact of outliers.| Scenarios requiring activation outlier suppression| [View details](outlier_suppression_algorithms/smooth_quant.md)|
| **Iterative Smooth** | Applies iterative smooth scaling for precise distribution adjustments.| Scenarios requiring accuracy optimization involving complex distributions| [View details](outlier_suppression_algorithms/iterative_smooth.md)|
| **Flex Smooth Quant** | Automatically searches for the optimal alpha and beta values across a two-phase grid search.| Scenarios requiring flexible adaptation to different architectures| [View details](outlier_suppression_algorithms/flex_smooth_quant.md)|
| **Flex AWQ SSZ** | Combines AWQ and SSZ to evaluate errors using real quantizers.| Scenarios requiring automated searches for fine-grained smoothing parameters| [View details](outlier_suppression_algorithms/flex_awq_ssz.md)|
| **KV Smooth** | Implements a smooth suppression algorithm tailored for the KV cache.| Scenarios requiring the reduction of GPU memory usage by the KV cache| [View details](outlier_suppression_algorithms/kv_smooth.md)|
|  **AWQ** | Performs a grid search for the optimal scaling factor based on the statistical characteristics of activation values.| Scenarios requiring automated searches for fine-grained smoothing parameters| [View details](outlier_suppression_algorithms/awq_smooth.md)|

## Quantization Algorithms

The following table summarizes the supported weight quantization, activation quantization, and structure-specific quantization solutions.

| Algorithm| Type| Core Idea| Application Scenario| Description|
| :--- | :--- | :--- | :--- | :--- |
| **AutoRound** | Weight quantization optimization| Optimizes the rounding offset based on SignSGD to reduce reconstruction error.| Scenarios requiring ultra-low-bit quantization such as 4-bit| [View details](quantization_algorithms/autoround.md)|
| **FA3 Quant** | Activation quantization| Performs per-head INT8 quantization for attention activations.| Scenarios involving long sequences or the MLA architecture| [View details](quantization_algorithms/fa3_quant.md)|
| **GPTQ** | Weight quantization optimization| Minimizes quantization error through column-wise optimization and error compensation.| Scenarios requiring high-precision weight quantization| [View details](quantization_algorithms/gptq.md)|
| **KVCache Quant** | KV cache quantization| Provides a quantization solution designed for the KV cache.| Scenarios requiring improved long-sequence inference efficiency| [View details](quantization_algorithms/kvcache_quant.md)|
| **Linear Quant** | Basic quantization| Performs weight quantization and activation quantization on linear layers.| Basic quantization scenarios| [View details](quantization_algorithms/linear_quant.md)|
| **PDMIX** | Phase-wise mixed quantization| Uses dynamic quantization for prefilling and static quantization for decoding.| Scenarios requiring large model inference acceleration to balance accuracy and performance| [View details](quantization_algorithms/pdmix.md)|
| **Histogram** | Activation quantization| Analyzes the histogram distribution to search for the optimal clipping interval| Scenarios requiring outlier filtering to improve accuracy| [View details](quantization_algorithms/histogram_activation_quantization.md)|
| **MinMax** | Basic quantization| Determines the quantization range based on the maximum and minimum statistics.| Basic quantization scenarios requiring low computational overhead| [View details](quantization_algorithms/minmax.md)|
| **SSZ** | Weight quantization| Iteratively searches for the optimal scaling factor and offset| Scenarios requiring accuracy optimization for uneven weight distributions| [View details](quantization_algorithms/ssz.md)|
| **LAOS** | Low-bit quantization| Optimizes ultra-low-bit scenarios such as W4A4| Scenarios requiring extreme compression| [View details](quantization_algorithms/laos.md)|
| **Float Sparse** | Sparsification| Implements model floating-point sparsification based on the ADMM algorithm| Scenarios requiring high compression ratios| [View details](quantization_algorithms/float_sparse.md)|

## Automatic Tuning Strategies

The following section summarizes the strategies that enable automated searches for the optimal quantization configuration.

| Algorithm| Core Idea| Application Scenario| Description|
| :--- | :--- | :--- | :--- |
| **Standing High** | Combines outlier suppression strategies with a binary search approach to minimize the number of fallback layers while meeting accuracy requirements.| Scenarios requiring fine-grained control over templates and strategies alongside a complete quantization configuration| [View details](auto_tuning_strategies/standing_high.md)|
| **Standing High With Experience** | Automatically generates a quantization configuration based on expert knowledge, requiring only the quantization type and structural layout.| Scenarios where users are familiar with the model architecture and prefer not to provide a complete quantization configuration| [View details](auto_tuning_strategies/standing_high_with_experience.md)|
| **Binary Fallback** | Binary-searches the minimum rollback prefix only; `template` is a full PracticeConfig.| Scenarios where outlier suppression is fixed in `template` and rollback path must be explicit| [View details](auto_tuning_strategies/binary_fallback.md)|

## Algorithm Selection Suggestions

- **Beginners**: Use [Quick Quantization (V1)](../feature_guide/quick_quantization_v1/usage.md), which integrates a suitable algorithm combination.
- **For ultimate model accuracy**: Use **QuaRot** and **AutoRound** in combination.
- **For long-sequence inference**: Enable **FA3 Quant** and **KVCache Quant**.
