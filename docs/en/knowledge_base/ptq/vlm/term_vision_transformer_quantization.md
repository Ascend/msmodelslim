# Vision-Language Model Quantization

> **Entry Category**: Quantization Basic Concept ([Quantization Basics](../README.md))<br>
> **English Name**: Vision-Language Model Quantization<br>
> **Abbreviation**: VLM Quantization<br>
> **Application Domain**: Multimodal understanding model inference acceleration, model compression

---

## 1. Overview

Vision-language model quantization ([Vision-Language Model Quantization](./term_vision_transformer_quantization.md)) is a technique that applies [post-training quantization (PTQ)](../term_ptq.md) to multimodal understanding models composed of a "vision encoder (ViT) + cross-modal projection layer + language decoder (LLM)" (such as Qwen-VL and GLM-4V), compressing the parameters and feature activations of the multimodal components into low-bit representations. It must handle the image-text bimodal distributions jointly, and focuses on applying error suppression or sensitivity protection to the vision alignment projection layer.

---

## 2. Term Description

### 2.1 Model Architecture Characteristics

Vision-language models (VLMs) adopt a cross-modal heterogeneous topology and mainly consist of three components:

- **Vision Encoder**: usually a high-resolution Vision Transformer (ViT); it encodes two-dimensional image patches (Patches) into a sequence of visual feature tokens, and its computation is mainly non-autoregressive forward passes.
- **Cross-modal projector / alignment layer (Projector/Merger)**: such as an MLP, Cross-Attention, or a pixel shuffle layer; it maps and aligns high-dimensional visual tokens into the text token embedding space.
- **Autoregressive language model (LLM Decoder)**: it receives the mixed image and text token sequence and generates the answer through autoregressive decoding.

### 2.2 Core Quantization Characteristics

Compared with pure language models, VLM quantization has the following notable characteristics:

- **Markedly different bimodal distributions**: the dynamic range and channel variance characteristics of visual activations and text activations are completely different, so quantization calibration must rely on a real multimodal dataset that mixes images and text (with diverse images and paired instruction text).
- **High sensitivity of the projection alignment layer to quantization error**: the projection layer performs the geometric coordinate transformation from visual to language space, and even fine numerical clipping can cause multimodal semantic alignment drift. In practice the projection layer is often kept at high precision (FP16/BF16) or quantized at an extremely fine granularity.
- **Spatial geometric alignment across modalities**: when a rotation-based algorithm (such as [QuaRot](../../quantization_algorithms/quarot/term_quarot.md)) is used to remove activation outliers on the text side, the corresponding orthogonal transform must be applied to the vision-side output features as well, so that the two feature spaces remain strictly aligned before being fed into the language decoder.
- **Tiered quantization strategy per component**: the vision encoder is mostly dense forward computation (limited by compute throughput), while long-sequence generation by the language decoder is limited by memory bandwidth, so different quantization bit widths and algorithm strategies must be configured for each.

---

## 3. Related Workflows

- [Multimodal Understanding Model (VLM) Quantization Usage Guide](./usage_vision_transformer_quantization.md): the complete workflow for VLM quantization.
- [VLM Model Integration Quantization Workflow Guide](./integration_guide_vision_transformer_quantization.md): development guidance for integrating a new vision-language model into the quantization workflow.
- *[One-Click Quantization Usage Guide](../../../user_guide/usage_one_click_quantization.md)*: covers the one-click quantization workflow for various model types.

---

## 4. Related Terms

- [PTQ](../term_ptq.md): the parent concept; vision-language model quantization is a concrete application of post-training quantization.
- [LLM Quantization](../llm/term_large_language_model_quantization.md): a companion concept; the language decoding part of a VLM reuses the LLM quantization scheme.
- [DiT Quantization](../dit/term_diffusion_transformer_quantization.md): a sibling concept; quantization of multimodal generation models.
- [QuaRot](../../quantization_algorithms/quarot/term_quarot.md): a companion algorithm involving rotation transforms on the modality-alignment projection layer.
- [Weight Conversion](../convert/term_weight_conversion.md): a companion term for offline reconstruction and format conversion of quantized weights.

---

## 5. Reference Documents

1. Dosovitskiy A, et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR 2021. <https://arxiv.org/abs/2010.11929>
2. Xiao G, et al. "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models." arXiv:2211.10438, 2022. <https://arxiv.org/abs/2211.10438>
3. Liu H, et al. "Visual Instruction Tuning (LLaVA)." NeurIPS 2023. <https://arxiv.org/abs/2304.08485>
