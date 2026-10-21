# Post-Training Quantization (PTQ)

> **Entry Category**: Quantization Basic Concept ([Quantization Basics](../README.md))<br>
> **English Name**: Post-Training Quantization<br>
> **Abbreviation**: PTQ<br>
> **Application Domain**: Model compression, inference acceleration

---

## 1. Overview

Post-training quantization (Post-Training Quantization, [PTQ](./term_ptq.md)) is a technique that, after a deep learning model has **finished training**, requires no retraining of model parameters: it uses only a small amount of representative calibration data to gather statistics on the numerical distributions of model weights and activations and maps high-precision floating-point tensors (such as FP16 and BF16) to low-bit integers (such as INT8 and INT4) or low-precision floating point (such as FP8 and MXFP8). PTQ offers low engineering cost, short conversion time, and plug-and-play convenience, making it the most general quantization scheme for large model inference deployment.

---

## 2. Term Description

### 2.1 Basic Principles and Paradigms

The core idea of PTQ is to compute the optimal scaling factor (Scale) and zero point (Zero Point) by offline statistics of the tensor extrema, histograms, or mean squared error observed during the forward pass over the calibration set, thereby establishing the mapping between high-precision and low-precision values.

Depending on the quantization target and the compute objective, PTQ mainly has two paradigms:

- **Weight-Only Quantization**: only the model weights are compressed to a low bit width (such as W8A16), and they are dequantized to floating point at compute time. It mainly addresses limited memory capacity and the memory-bandwidth bottleneck (Memory-bound), greatly reducing the memory occupied by model weights; it is common in the LLM generation phase with a small batch size (Batch Size).
- **Weight-Activation Quantization**: both weights and activations are quantized to a low bit width (such as W8A8 and W4A8), directly invoking the hardware low-precision matrix multiplication operators (GEMM/MatMul). It reduces memory usage and accelerates computation (Compute-bound) at the same time, and delivers significant speedup gains in large-batch or multimodal inference.

### 2.2 Comparison of PTQ Characteristics Across Mainstream Large Model Architectures

Different model architectures have different compute patterns and data-flow characteristics, so PTQ shows highly differentiated quantization priorities:

| Architecture Type | Core Model Structure | Inference Compute Characteristics | Core Quantization Challenge and Response |
| :--- | :--- | :--- | :--- |
| **[LLM Quantization](./llm/term_large_language_model_quantization.md)** | Transformer Decoder-only | Autoregressive token-by-token generation, memory-bandwidth bound (Memory-bound); carries a KV cache | Activations have fixed outliers (Outliers) in specific channels, so outlier suppression or rotation algorithms such as [SmoothQuant](../quantization_algorithms/smooth_quant/term_smooth_quant.md) and [QuaRot](../quantization_algorithms/quarot/term_quarot.md) are required; KV cache quantization also needs to be combined in. |
| **[VLM Quantization](./vlm/term_vision_transformer_quantization.md)** | ViT vision encoder + text decoder | Cross-modal heterogeneity; image features are fused into the text space through a projection layer | The distribution characteristics of the vision encoder and the text decoder differ greatly, and the modality-alignment projection layer (Merger/Projection) is sensitive to quantization, so multimodal calibration data is required together with targeted sensitivity protection or cross-modal rotation coordination. |
| **[DiT Quantization](./dit/term_diffusion_transformer_quantization.md)** | Diffusion Transformer backbone | Multi-step iterative denoising, no KV cache, compute-intensive (Compute-bound) | The activation distribution is highly dynamic and non-stationary across timesteps (Timestep), so calibration must cover the denoising trajectory of multiple timesteps; multi-expert models need per-expert distribution statistics. |

### 2.3 Technical Challenges

- **Activation outlier suppression**: activations in deep Transformers often contain outlier channels whose magnitudes differ greatly from the other channels, which easily causes clipping error in conventional quantization or lets an extreme value dominate the scaling factor.
- **Representativeness of the calibration set distribution**: PTQ does not update model weights, so the statistical parameters depend heavily on calibration data. If the calibration samples do not cover the full domain, generalization accuracy tends to degrade.
- **Maintaining accuracy at ultra-low bit widths**: when moving to lower bit widths such as INT4 and INT3, the clipping loss of linear quantization rises exponentially, requiring advanced algorithms such as second-order error compensation or channel reordering.
- **Error accumulation across layers**: the local quantization error of a single layer spreads and amplifies layer by layer during the forward pass, so end-to-end sensitivity analysis is needed to protect the accuracy of critical layers.

---

## 3. Related Workflows

- [Large Language Model (LLM) Quantization Usage Guide](./llm/usage_large_language_model_quantization.md): the complete workflow for LLM quantization.
- [Multimodal Understanding Model (VLM) Quantization Usage Guide](./vlm/usage_vision_transformer_quantization.md): the complete workflow for VLM quantization.
- [Multimodal Generation Model (DiT) Quantization Usage Guide](./dit/usage_diffusion_transformer_quantization.md): the complete workflow for DiT quantization.
- [Weight Conversion Usage Guide](./convert/usage_weight_conversion.md): the complete workflow for weight conversion.
- *[One-Click Quantization Usage Guide](../../user_guide/usage_one_click_quantization.md)*: covers the overall one-click quantization workflow for various model types.

---

## 4. Related Terms

- [LLM Quantization](./llm/term_large_language_model_quantization.md): a subordinate concept; PTQ for autoregressive language models.
- [VLM Quantization](./vlm/term_vision_transformer_quantization.md): a subordinate concept; PTQ for multimodal vision-language models.
- [DiT Quantization](./dit/term_diffusion_transformer_quantization.md): a subordinate concept; PTQ for diffusion Transformer generation models.
- [Weight Conversion](./convert/term_weight_conversion.md): a companion term for offline reconstruction and format conversion of quantized weights.
- [SmoothQuant](../quantization_algorithms/smooth_quant/term_smooth_quant.md): a companion algorithm; a commonly used activation outlier suppression technique in PTQ.

---

## 5. Reference Documents

1. Gholami A, et al. "A Survey of Quantization Methods for Efficient Neural Network Inference." arXiv:2103.13630, 2021. <https://arxiv.org/abs/2103.13630>
2. Nagel M, et al. "A White Paper on Neural Network Quantization." arXiv:2106.08295, 2021. <https://arxiv.org/abs/2106.08295>
3. Xiao G, et al. "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models." arXiv:2211.10438, 2022. <https://arxiv.org/abs/2211.10438>
