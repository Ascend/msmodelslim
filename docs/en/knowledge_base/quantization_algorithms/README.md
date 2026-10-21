# Quantization Algorithm Overview

msModelSlim supports multiple advanced quantization algorithms, covering all aspects from outlier suppression to low-bit optimization. The following tables summarize the currently supported core algorithms and their primary characteristics by category.

> **Reading tip**: Each algorithm directory contains two types of documents: "Term" and "Usage Guide". A Term document only introduces the algorithm definition, principles, properties, applicable scenarios, and related knowledge. A Usage Guide targets new users and focuses on the meaning of the configuration parameters, recommended starting values, and when to adjust them. To look up complete field types, default values, and advanced task-level configuration, continue reading the *[modelslim_v1 Configuration Reference](../../api_reference/config/quant/modelslim_v1.md)*.

## 1. Outlier Suppression Algorithms

Outlier suppression algorithms aim to smooth the distribution of activation values and reduce the accuracy loss caused by quantization.

| Algorithm | Core Idea | Use Cases | Term | Usage Guide |
| :--- | :--- | :--- | :--- | :--- |
| **QuaRot** | Applies an orthogonal rotation matrix to smooth the distribution of activation values | Suppresses activation outliers to improve accuracy | [QuaRot Term](quarot/term_quarot.md) | [QuaRot Usage Guide](quarot/usage_quarot.md) |
| **Adapt Rotation** | Iteratively optimizes the Hadamard rotation matrix based on calibration data, building on QuaRot | Optimizes the rotation matrix to further improve low-bit quantization accuracy | [Adapt Rotation Term](adapt_rotation/term_adapt_rotation.md) | [Adapt Rotation Usage Guide](adapt_rotation/usage_adapt_rotation.md) |
| **SmoothQuant** | Scales activations and weights cooperatively to smooth outliers | Suppresses activation outliers | [SmoothQuant Term](smooth_quant/term_smooth_quant.md) | [SmoothQuant Usage Guide](smooth_quant/usage_smooth_quant.md) |
| **Iterative Smooth** | Iterative smooth scaling for finer distribution adjustment | Accuracy optimization under complex distributions | [Iterative Smooth Term](iterative_smooth/term_iterative_smooth.md) | [Iterative Smooth Usage Guide](iterative_smooth/usage_iterative_smooth.md) |
| **Flex Smooth Quant** | Two-phase grid search that automatically finds the optimal alpha/beta | Flexibly adapts to different architectures | [Flex Smooth Quant Term](flex_smooth_quant/term_flex_smooth_quant.md) | [Flex Smooth Quant Usage Guide](flex_smooth_quant/usage_flex_smooth_quant.md) |
| **Flex AWQ SSZ** | Combines AWQ and SSZ and evaluates errors with a real quantizer | Automatically searches for the optimal smoothing parameters | [Flex AWQ SSZ Term](flex_awq_ssz/term_flex_awq_ssz.md) | [Flex AWQ SSZ Usage Guide](flex_awq_ssz/usage_flex_awq_ssz.md) |
| **KV Smooth** | A smooth suppression algorithm tailored for the KV cache | Reduces KV cache memory usage | [KV Smooth Term](kv_smooth/term_kv_smooth.md) | [KV Smooth Usage Guide](kv_smooth/usage_kv_smooth.md) |
| **AWQ** | Grid-searches the optimal scaling factor based on activation statistics | Automatically searches for the optimal smoothing parameters | [AWQ Term](awq_smooth/term_awq_smooth.md) | [AWQ Usage Guide](awq_smooth/usage_awq_smooth.md) |
| **OASQ** | Identifies outliers by channel-wise z-score and splits the smoothing scale accordingly | Upfront suppression when channel-level activation outliers are obvious | [OASQ Term](oasq/term_oasq.md) | [OASQ Usage Guide](oasq/usage_oasq.md) |

## 2. Quantization Algorithms

Covers weight quantization, activation quantization, and quantization solutions for specific structures.

| Algorithm | Type | Core Idea | Use Cases | Term | Usage Guide |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **AutoRound** | Weight quantization optimization | Optimizes the rounding offset based on SignSGD to reduce reconstruction error | Ultra-low-bit quantization such as 4-bit | [AutoRound Term](autoround/term_autoround.md) | [AutoRound Usage Guide](autoround/usage_autoround.md) |
| **Trainable Linear Quant** | Weight quantization optimization | Block-level training plus pluggable OPs (range/rounding/optional smoothing) to optimize linear quantization parameters | Mainly for extremely low bit widths such as W4A4; higher memory usage and runtime | [Trainable Linear Quant Term](trainable_linear_quant/term_trainable_linear_quant.md) | [Trainable Linear Quant Usage Guide](trainable_linear_quant/usage_trainable_linear_quant.md) |
| **FA3 Quant** | Activation quantization | Per-head INT8 quantization for attention activations | Long sequences and MLA-architecture models | [FA3 Quant Term](fa3_quant/term_fa3_quant.md) | [FA3 Quant Usage Guide](fa3_quant/usage_fa3_quant.md) |
| **GPTQ** | Weight quantization optimization | Minimizes quantization error through column-wise optimization and error compensation | Scenarios requiring high-accuracy weight quantization | [GPTQ Term](gptq/term_gptq.md) | [GPTQ Usage Guide](gptq/usage_gptq.md) |
| **KVCache Quant** | KV cache quantization | A quantization solution designed for the KV cache | Improves long-sequence inference efficiency | [KVCache Quant Term](kvcache_quant/term_kvcache_quant.md) | [KVCache Quant Usage Guide](kvcache_quant/usage_kvcache_quant.md) |
| **Linear Quant** | Basic quantization | Performs weight quantization and activation quantization on linear layers | Basic quantization scenarios | [Linear Quant Term](linear_quant/term_linear_quant.md) | [Linear Quant Usage Guide](linear_quant/usage_linear_quant.md) |
| **PDMIX** | Phase-wise mixed quantization | Uses dynamic quantization for prefilling and static quantization for decoding | Large model inference acceleration that balances accuracy and performance | [PDMIX Term](pdmix/term_pdmix.md) | [PDMIX Usage Guide](pdmix/usage_pdmix.md) |
| **Histogram** | Activation quantization | Analyzes the histogram distribution to search for the optimal clipping interval | Outlier filtering to improve accuracy | [Histogram Term](histogram_activation_quantization/term_histogram_activation_quantization.md) | [Histogram Usage Guide](histogram_activation_quantization/usage_histogram_activation_quantization.md) |
| **MinMax** | Basic quantization | Determines the quantization range from the maximum and minimum statistics | Basic quantization scenarios with low computational overhead | [MinMax Term](minmax/term_minmax.md) | [MinMax Usage Guide](minmax/usage_minmax.md) |
| **HQQ** | Weight quantization | Half-quadratic splitting iteratively optimizes the quantization offset; calibration-free | Calibration-free W8A16 weight quantization without calibration data | [HQQ Term](hqq/term_hqq.md) | [HQQ Usage Guide](hqq/usage_hqq.md) |
| **SSZ** | Weight quantization | Iteratively searches for the optimal scaling factor and offset | Accuracy optimization for uneven weight distributions | [SSZ Term](ssz/term_ssz.md) | [SSZ Usage Guide](ssz/usage_ssz.md) |
| **LAOS** | Combined solution (LLM) | Adapt Rotation + AutoRound, targeting extremely low bit widths such as W4A4 for large language models | Extreme LLM compression | [LAOS Term](laos/term_laos.md) | [LAOS Usage Guide](laos/usage_laos.md) |
| **DAOS** | Combined solution (multimodal generation) | OASQ + TLQ; can also pair with FA3 for FA quantization to accelerate inference | Low-bit multimodal generation | [DAOS Term](daos/term_daos.md) | [DAOS Usage Guide](daos/usage_daos.md) |
| **Float Sparse** | Sparsification | Implements model floating-point sparsification based on the ADMM algorithm | Scenarios requiring high compression ratios | [Float Sparse Term](float_sparse/term_float_sparse.md) | [Float Sparse Usage Guide](float_sparse/usage_float_sparse.md) |
| **SVDQuant** | Comprehensive solution | Outlier migration + SVD low-rank residual + residual quantization | Low-bit quantization of diffusion models | [SVDQuant Term](svdquant/term_svdquant.md) | [SVDQuant Usage Guide](svdquant/usage_svdquant.md) |
| **MSE_Round** | Weight quantization | Selects, per block, between the ceil/floor shared exponent by MSE | Accuracy optimization for MXFP8 weight quantization | [MSE_Round Term](mse_round/term_mse_round.md) | [MSE_Round Usage Guide](mse_round/usage_mse_round.md) |
| **FouroverSix** | Weight quantization | Adaptively selects the block scaling (Scale-to-6 / Scale-to-4) | Quantization error optimization for MXFP4 | [FouroverSix Term](fouroversix/term_fouroversix.md) | [FouroverSix Usage Guide](fouroversix/usage_fouroversix.md) |
| **Ceil_X** | Weight quantization | Computes the shared exponent with ceil plus a configurable divisor | Suppression of large-value clipping in MXFP4 | [Ceil_X Term](ceil_x/term_ceil_x.md) | [Ceil_X Usage Guide](ceil_x/usage_ceil_x.md) |
| **DualScale** | Weight quantization | Two-level progressive scaling to mitigate abnormal channels | Low-bit scenarios such as W4A4 | [DualScale Term](dual_scale/term_dual_scale.md) | [DualScale Usage Guide](dual_scale/usage_dual_scale.md) |

## 3. Sensitive Layer Analysis Algorithms

Sensitive layer analysis is performed via `msmodelslim analyze` to measure the sensitivity of each layer or substructure to quantization on calibration data. The results are ranked to assist with fallback strategies and YAML parameter tuning.

| Algorithm | Analysis Scope | Core Idea | Use Cases | Term | Usage Guide |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Std** | linear | Characterizes sensitivity by the ratio of the activation dynamic range to the standard deviation | Coarse screening of linear layers before quantization; one of the default strategies | [Std Term](std/term_std.md) | [Std Usage Guide](std/usage_std.md) |
| **Quantile** | linear | Builds a score from quantiles and the IQR; relatively robust to outliers | Heavy activation tails where outlier dominance should be reduced | [Quantile Term](quantile/term_quantile.md) | [Quantile Usage Guide](quantile/usage_quantile.md) |
| **Kurtosis** | linear | Estimates activation kurtosis to capture the impact of peaks and extreme values | Distributions with sharp peaks; combined with fallback or mixed precision | [Kurtosis Term](kurtosis/term_kurtosis.md) | [Kurtosis Usage Guide](kurtosis/usage_kurtosis.md) |
| **Attention MSE (mse)** | attn (attention structures) | MSE of the attention output under floating-point and quantized weights | Sensitivity of attention weights to quantization (requires an adapter interface) | [Attention MSE Term](attention_mse/term_attention_mse.md) | [Attention MSE Usage Guide](attention_mse/usage_attention_mse.md) |
| **Layer-wise MSE (mse_layer_wise)** | layer (decoder blocks) | Mean MSE over the outputs of the selected submodules within each block | Fallback for entire layers or blocks (such as MLP or attention segments) | [Layer-wise MSE Term](mse_layer_wise/term_mse_layer_wise.md) | [Layer-wise MSE Usage Guide](mse_layer_wise/usage_mse_layer_wise.md) |
| **Model-wise MSE (mse_model_wise)** | layer (chained forward pass) | MSE of the **final model output** under layer-by-layer quantization perturbation | Layer sensitivity from the perspective of the final hidden state | [Model-wise MSE Term](mse_model_wise/term_mse_model_wise.md) | [Model-wise MSE Usage Guide](mse_model_wise/usage_mse_model_wise.md) |

## 4. Attention Head Analysis Algorithms

Attention head analysis uses `msmodelslim analyze` to measure the cross-segment behavior of attention heads on the repeated-segment input built by the algorithm itself, and outputs the list of KV heads to retain (`head.pt`) for delivery to the MindIE inference framework for long-sequence KV cache compression.

| Algorithm | Analysis Scope | Core Idea | Use Cases | Term | Usage Guide |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RA Compress** | attn_head (attention heads) | Measures the cross-segment attention strength of induction heads / echo heads based on the repeated-segment structure, and selects the critical KV heads | Selecting the KV heads to retain before long-sequence KV cache compression | [RA Compress Term](ra_compress/term_ra_compress.md) | [RA Compress Usage Guide](ra_compress/usage_ra_compress.md) |

## 5. Algorithm Selection Suggestions

Beginners can start with *[One-Click Quantization (V1)](../../user_guide/usage_one_click_quantization.md)*, which automatically integrates validated algorithm combinations. When an automatically searched configuration is needed, see the *[Auto Tuning Strategies Overview](../tuning_strategies/README.md)*. For practical configurations, also refer to the corresponding YAML files under `lab_practice/`.

### 5.1 Quantization Algorithms

- **W8A8**: The most common choice is **MinMax** — it determines the quantization range from the maximum and minimum statistics with low computational overhead, making it a good default starting point.
- **W8A16**: Use **HQQ** — half-quadratic splitting iteratively optimizes the offset and completes weight quantization without calibration (no calibration data required), suitable when calibration data is unavailable or quantization runtime is a concern.
- **W4A8**: Use **SSZ** on the weight side to iteratively search for the scaling factor and offset, while activations still use **MinMax** for A8; the two work together.
- **W4A4 MXFP**: Prefer **Ceil_X**, which tightens the shared exponent with ceil plus a configurable divisor to suppress the large-value clipping caused by floor scaling.
- **W4A4 (INT / MXFP)**: Optionally use the training-based **Trainable Linear Quant (TLQ)** to further improve accuracy, for both INT and MXFP; TLQ organizes range/rounding (and optional trainable smoothing) as a pluggable OP pipeline, but it demands more compute, and its quantization runtime is typically several times that of most other algorithms, so resource usage and latency must be weighed.
- **Combined low-bit for LLMs**: **LAOS** (Adapt Rotation + AutoRound) targets large language models; for practice, see the corresponding LLM recipe under `lab_practice`.
- **Combined low-bit for multimodal generation**: **DAOS** (OASQ + TLQ; pair it with FA3 for FA quantization to accelerate inference when needed). Do not mix its task protocol or calibration path with LAOS.
- **Long sequences / C8**: In the product, both **KVCache Quant** and **FA3 Quant** are grouped under C8. The former quantizes the Key/Value written to the cache, focusing on shrinking the KV cache and relieving inference memory pressure; the latter quantizes the Q/K/V activations along the attention path to accelerate the attention computation (MLA only). Because their mechanisms differ, in practice usually only one is enabled, chosen by whether memory or compute is the bottleneck.

### 5.2 Outlier Suppression Algorithms

- **Flex Smooth Quant** and **QuaRot** are the common choices: each can be used independently or chained together. The former runs a two-phase grid search over alpha/beta and adapts broadly; the latter uses an orthogonal rotation to smooth activation outliers and often yields a more noticeable accuracy gain, but places higher demands on model adaptation.
- **OASQ** fits scenarios where activations have obvious channel-level outliers and the smoothing scale should be split between outlier and normal channels; it can serve as a preprocessing step for subsequent linear quantization or trainable quantization.
- **Flex AWQ SSZ** performs well in 4-bit low-precision scenarios, but its search is relatively slow; it suits scenarios that prioritize accuracy and can accept a longer quantization time.

### 5.3 Sensitive Layer Analysis

Sensitive layer analysis currently supports measuring sensitivity on calibration data at three scopes — `linear` / `layer` / `attn` — and using the results for fallback and mixed-precision tuning at the corresponding granularity. Usage guides: *[Linear Layers](../../user_guide/usage_sensitive_linear_analysis.md)*, *[Layers](../../user_guide/usage_sensitive_layer_analysis.md)*, *[Attention](../../user_guide/usage_sensitive_attn_analysis.md)*.

- **linear** (linear layers): **Kurtosis** is preferred; it uses activation kurtosis to capture the impact of peaks and tails and helps identify linear layers that need fallback or a wider bit width.
- **layer** (decoder blocks): **mse_layer_wise** is preferred and suits fallback of entire layers or blocks (such as MLP or attention segments).
- **attn** (attention structures): **Attention MSE (mse)** is preferred, mainly used together with **FA3 Quant** to identify attention modules that need fallback (an adapter interface is required).

If the goal is attention head filtering before long-sequence KV cache compression (`attn_head`), use **RA Compress**: it filters induction heads / echo heads by the repeated-segment structure and produces the `head.pt` list of critical heads for delivery to the subsequent MindIE inference framework (LLM only; the calibration input is built by the algorithm itself at runtime, so `--calibration_dataset` is not required). This scope produces a deliverable rather than a sensitivity ranking and is not used for fallback or mixed-precision tuning. For the algorithm description, see *[4. Attention Head Analysis Algorithms](#4-attention-head-analysis-algorithms)*; for the usage guide, see *[Attention Head](../../user_guide/usage_sensitive_attn_head_analysis.md)*.
