# Large Language Model Quantization

> **Entry Category**: Quantization Basic Concept ([Quantization Basics](../README.md))<br>
> **English Name**: Large Language Model Quantization<br>
> **Abbreviation**: LLM Quantization<br>
> **Application Domain**: Large language model inference acceleration, model compression

---

## 1. Overview

Large language model quantization ([Large Language Model Quantization](./term_large_language_model_quantization.md)) is a technique that applies [post-training quantization (PTQ)](../term_ptq.md) to autoregressive text generation models with the mainstream Transformer Decoder-only architecture (such as LLaMA, Qwen, and DeepSeek), converting weights and activations from a high-precision floating-point format (FP16/BF16) to a low-bit format (such as INT8, INT4, and FP8). Its core characteristics are activation outlier suppression, low-overhead calibration, and a flexible weight-activation ratio; it aims to greatly reduce model memory usage and break through the memory-bandwidth bottleneck of the autoregressive decoding phase.

---

## 2. Term Description

### 2.1 Model and Inference Characteristics

Mainstream large language models have the following main architecture and runtime characteristics:

- **Deep Decoder-only stacking**: it is formed by cascading many homogeneous Transformer decoder blocks (containing the self-attention mechanism and the feed-forward network MLP), giving a regular and uniform network structure.
- **Two-phase autoregressive inference**: it consists of a prefill (Prefill) phase and a token-by-token decoding (Decode) phase. The Decode phase is limited by memory bandwidth (Memory-bound): generating each token requires reading the full set of weights.
- **KV cache memory growth**: the Key/Value states of historical tokens must be cached in memory, and as the batch size (Batch Size) and context length grow, the memory used by the KV cache can even exceed that of the weights themselves.

### 2.2 Core Quantization Characteristics

In response to the architectural and runtime bottlenecks of LLMs, their quantization schemes have the following key characteristics:

- **Weight quantization addresses the memory-access bottleneck**: in small-batch scenarios, Weight-only quantization significantly reduces the amount of weight traffic per forward step, greatly lowering memory usage and increasing generation speed.
- **Prominent outliers (Outliers) in specific channels**: among the activations of deep networks, a very small number of channels have magnitudes far higher than the others (up to hundreds of times). Direct quantization causes severe accuracy degradation, so distribution reshaping is usually required using [SmoothQuant](../../quantization_algorithms/smooth_quant/term_smooth_quant.md) (weight-activation smooth scaling) or [QuaRot](../../quantization_algorithms/quarot/term_quarot.md) (orthogonal rotation to remove outliers).
- **Lightweight calibration and data-free adaptation**: the calibration process requires no massive corpus; usually only tens to hundreds of generally representative text sequences are needed to derive stable quantization Scale parameters.
- **KV cache quantization coordination**: beyond the regular Linear layers (QKV, Dense, MLP), compressing the KV cache to INT8/FP8 format can improve the throughput of long-text inference.

---

## 3. Related Workflows

- [LLM Quantization Usage Guide](./usage_large_language_model_quantization.md): the complete workflow for LLM quantization.
- [LLM Model Integration Quantization Workflow Guide](./integration_guide_large_language_model_quantization.md): development guidance for integrating a new language model into the quantization workflow.
- *[One-Click Quantization Usage Guide](../../../user_guide/usage_one_click_quantization.md)*: covers the one-click quantization workflow for various model types.

---

## 4. Related Terms

- [PTQ](../term_ptq.md): the parent concept; large language model quantization is a concrete application of post-training quantization.
- [VLM Quantization](../vlm/term_vision_transformer_quantization.md): a sibling concept; quantization of multimodal understanding models (its text side reuses the principles of LLM quantization).
- [DiT Quantization](../dit/term_diffusion_transformer_quantization.md): a comparison algorithm; quantization of diffusion generation models (a non-autoregressive denoising paradigm).
- [SmoothQuant](../../quantization_algorithms/smooth_quant/term_smooth_quant.md): a companion algorithm; the classic smoothing algorithm for LLM activation outliers.
- [QuaRot](../../quantization_algorithms/quarot/term_quarot.md): a companion algorithm that removes activation outliers through an orthogonal rotation matrix.
- [Weight Conversion](../convert/term_weight_conversion.md): a companion term for offline reconstruction and format conversion of quantized weights.

---

## 5. Reference Documents

1. Xiao G, et al. "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models." arXiv:2211.10438, 2022. <https://arxiv.org/abs/2211.10438>
2. Frantar E, et al. "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." arXiv:2210.17323, 2022. <https://arxiv.org/abs/2210.17323>
3. Ashkboos S, et al. "QuaRot: Outlier-Free 4-Bit Inference in Large Language Models." arXiv:2404.00456, 2024. <https://arxiv.org/abs/2404.00456>
