# 视觉语言模型（VLM）量化术语百科词条

> **词条类别**：[量化基础概念](../README.md)<br>
> **英文名称**：Vision-Language Model Quantization<br>
> **英文缩写**：VLM Quantization<br>
> **应用领域**：多模态理解模型推理加速、模型压缩

---

## 1. 概述

视觉语言模型（VLM）量化是指对"视觉编码器 + 语言解码器"的多模态理解模型（如 Qwen-VL、GLM-4V 等）进行[训练后量化（PTQ）](../term_ptq.md)的过程，将视觉编码器与文本解码器的权重和激活值映射到低比特表示，从而降低多模态推理的显存占用与延迟。与[LLM 量化](../llm/term_large_language_model_quantization.md)不同，VLM 量化需同时处理视觉与文本两种模态，视觉编码器全量加载、视觉组件需排除量化是其核心特征。

## 2. 模型特点

VLM 采用「视觉编码器 + 语言解码器」的双模态异构架构，视觉侧负责将图像转化为特征序列，语言侧负责跨模态理解与生成，两部分在结构、参数规模与数据流上均有明显差异。这种异构性意味着 VLM 既不能简单沿用 LLM 的量化假设，也不能将两个模态割裂处理，是理解 VLM 量化特殊性的出发点。具体而言：

- **双模态异构架构**：由视觉编码器（ViT）与语言解码器（LLM）两部分构成，视觉编码器负责图像特征提取，语言解码器负责跨模态推理生成。
- **视觉编码器参数可观**：如 Qwen3-VL 的视觉编码器约 675M 参数，在多模态模型显存占用中占比显著。
- **视觉编码器需整体加载**：视觉编码器一次性全量加载到显存，无法像文本解码器那样逐层按需加载，显存压力较高。
- **存在视觉特征投影层**：如 merger、linear_fc2、deepstack_merger_list 等，负责将图像 patch 特征映射到文本嵌入空间，对量化误差敏感。
- **视觉-语言特征空间耦合**：图像特征需与文本嵌入对齐，量化时需保持两个模态特征空间的几何一致性。

## 3. 量化特点

VLM 的量化方案围绕上述异构架构展开：语言解码器部分可以复用成熟的 LLM 逐层量化方案，而视觉编码器则因其全量加载方式、参数占比与对量化误差的敏感性，需要专门的量化策略。两个模态不是独立处理，而是在同一流程中协同校准、耦合量化，并保持视觉与文本特征空间的对齐。整体来看，VLM 量化呈现出以下特点：

- **多模态校准数据**：需使用图像校准数据（含图像与文本 prompt）而非纯文本统计视觉模态的激活分布。
- **视觉编码器整体量化**：ViT 编码器作为整体模块参与量化处理（而非逐层），其量化参数基于图像校准数据独立统计。
- **视觉组件排除量化**：视觉特征投影层（`*merger*`、`*linear_fc2`、`*deepstack_merger_list*`）通常从量化范围中排除，以保持视觉特征精度。
- **旋转对齐**：采用 QuaRot 旋转量化时，视觉输出投影层需左旋转（$W_{\text{new}} = R^T W$），使视觉特征进入与文本嵌入相同的旋转空间。
- **与文本解码器耦合量化**：ViT 量化通常与语言解码器量化作为同一 VLM 量化流程执行，而非独立量化。
- **逐层 + 全量混合**：文本解码器沿用 LLM 的逐层量化策略，视觉编码器采用全量加载策略，两种策略共存于同一流程。

## 4. 关联流程

- [《VLM 量化使用指南》](./usage_vision_transformer_quantization.md)：VLM 量化完整流程，含视觉编码器量化与视觉组件排除。
- [《VLM 模型接入量化流程指南》](./integration_guide_vision_transformer_quantization.md)：将新 VLM 模型接入量化流程，包含视觉编码器的适配说明。
- [《一键量化完整指南》](../../../user_guide/usage_one_click_quantization.md)：涵盖 VLM 与多模态模型的一键量化流程，默认集成 VLM 量化方案。

## 5. 关联词条

- [PTQ](../term_ptq.md)：上位概念，VLM 量化属于训练后量化的一种具体应用。
- [LLM 量化](../llm/term_large_language_model_quantization.md)：配套术语，VLM 的语言解码器部分采用与 LLM 相同的逐层量化方案。
- [DiT 量化](../dit/term_diffusion_transformer_quantization.md)：同类概念，多模态生成模型的量化，同样涉及视觉 Transformer 编码器。
- [权重转换](../convert/term_weight_conversion.md)：应用对象，VLM 量化权重可通过该工具调整格式/精度。

## 6. 参考文档

1. Dosovitskiy A, et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR 2021. https://arxiv.org/abs/2010.11929
2. Xiao G, et al. "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models." arXiv:2211.10438, 2022. https://arxiv.org/abs/2211.10438
