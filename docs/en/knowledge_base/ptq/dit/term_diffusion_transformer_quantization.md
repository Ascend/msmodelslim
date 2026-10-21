# Diffusion Transformer Quantization

> **Entry Category**: Quantization Basic Concept ([Quantization Basics](../README.md))<br>
> **English Name**: Diffusion Transformer Quantization<br>
> **Abbreviation**: DiT Quantization<br>
> **Application Domain**: Multimodal generation model (text-to-image / text-to-video) inference acceleration, model compression

---

## 1. Overview

Diffusion Transformer quantization ([Diffusion Transformer Quantization](./term_diffusion_transformer_quantization.md)) is a technique that applies [post-training quantization (PTQ)](../term_ptq.md) to diffusion generation models with a Transformer backbone (such as FLUX, HunyuanVideo, and Wan2.2), converting weights and high-dimensional activation tensors to low-bit floating-point or integer formats (such as MXFP8, FP8, and INT8).

---

## 2. Term Description

### 2.1 Model and Inference Characteristics

Diffusion Transformer (DiT) uses the Transformer structure to completely replace the U-Net backbone of traditional diffusion models, and has the following characteristics:

- **Multi-step iterative denoising**: generating a single image or a continuous video requires dozens of denoising forward passes (such as 20 to 50 steps), with a different noise timestep (Timestep) embedding fed in at each step.
- **No autoregression and no KV cache**: DiT is a pure forward network of dense matrix computation and does not have the KV cache memory growth problem of autoregressive text generation; its inference latency is limited by high compute throughput (Compute-bound).
- **Multiple conditioning branches and mixture of experts**: advanced DiT models (such as Wan2.2 and FLUX) often contain text-encoding cross-attention channels, or multi-expert networks decoupled for different noise stages (such as a high-noise / low-noise expert).

### 2.2 Core Quantization Characteristics

Unlike the autoregressive or single-pass forward of LLMs/VLMs, the quantization focus of DiT models centres on adapting to dynamic timesteps and highly dense computation:

- **Timestep-Aware Calibration**: the dynamic range and magnitude distribution of activations differ strongly and non-stationarily between the early (high-noise) and late (low-noise) stages of denoising. Quantization calibration must uniformly sample the hidden-layer activations of different timesteps across the complete floating-point denoising trajectory to avoid global Scale distortion.
- **Decoupled independent statistics for multiple experts**: for diffusion architectures that use staged experts, the weight and activation distributions of different experts differ markedly, so the calibration activation distribution must be built and the quantization parameters solved independently for each expert.
- **High-throughput low-precision floating point preferred**: because denoising iterations exhibit an obvious error accumulation effect, industry practice favours wide-dynamic-range floating-point formats such as MXFP8 and FP8 (with micro-block block scaling), which greatly accelerate dense GEMM computation while tightly controlling the perceptual degradation of images/videos.

---

## 3. Related Workflows

- [Multimodal Generation Model (DiT) Quantization Usage Guide](./usage_diffusion_transformer_quantization.md): the complete workflow and command reference for DiT quantization.
- [DiT Model Integration Quantization Workflow Guide](./integration_guide_diffusion_transformer_quantization.md): guidance for integrating a new diffusion generation model into the quantization workflow.
- *[One-Click Quantization Usage Guide](../../../user_guide/usage_one_click_quantization.md)*: covers the one-click quantization workflow for various generation models.

---

## 4. Related Terms

- [PTQ](../term_ptq.md): the parent concept; Diffusion Transformer quantization is a concrete application of post-training quantization.
- [LLM Quantization](../llm/term_large_language_model_quantization.md): a comparison algorithm; quantization of autoregressive language models (which have a KV cache and outlier channels).
- [VLM Quantization](../vlm/term_vision_transformer_quantization.md): a sibling concept; quantization of multimodal understanding models.
- [Weight Conversion](../convert/term_weight_conversion.md): a companion term for offline reconstruction and format conversion of quantized weights.

---

## 5. Reference Documents

1. Peebles W, Xie S. "Scalable Diffusion Models with Transformers." ICCV 2023. <https://arxiv.org/abs/2212.09748>
2. Esser P, et al. "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis." arXiv:2403.03206, 2024. <https://arxiv.org/abs/2403.03206>
