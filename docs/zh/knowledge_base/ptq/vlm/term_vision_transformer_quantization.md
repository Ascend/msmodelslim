# 视觉语言模型量化

> **词条类别**：[量化基础概念](../README.md)<br>
> **英文名称**：Vision-Language Model Quantization<br>
> **英文缩写**：VLM Quantization<br>
> **应用领域**：多模态理解模型推理加速、模型压缩

---

## 1. 概述

视觉语言模型量化（[Vision-Language Model Quantization](./term_vision_transformer_quantization.md)）是指针对由“视觉编码器（ViT）+ 跨模态投影层 + 语言解码器（LLM）”构成的多模态理解模型（如 Qwen-VL、GLM-4V 等），应用[训练后量化（PTQ）](../term_ptq.md)将多模态组件的参数和特征激活压缩为低比特表示的技术。该技术需协同处理图文双模态分布，并重点针对视觉对齐投影层实施误差抑制或敏感度保护。

---

## 2. 词条介绍

### 2.1 模型架构特征

视觉语言模型（VLM）采用跨模态异构拓扑，主要包含三大组件：

- **视觉编码器（Vision Encoder）**：通常为高分辨率 Vision Transformer（ViT），负责将二维图像切片（Patches）编码为视觉特征 Token 序列，计算以非自回归前向为主。
- **跨模态投影/对齐层（Projector/Merger）**：如 MLP、Cross-Attention 或像素重组层，将高维视觉 Token 映射并对齐到文本 Token 嵌入空间。
- **自回归语言模型（LLM Decoder）**：接收图像与文本混合 Token 序列，通过自回归解码生成答案。

### 2.2 量化核心特点

相较于纯语言模型，VLM 量化具有以下显著特点：

- **双模态分布差异显著**：视觉激活值与文本激活值的动态范围和通道方差特征截然不同，量化校准必须依赖图文混合（包含多样性图像和配对指令文本）的真实多模态数据集。
- **投影对齐层对量化误差高度敏感**：投影层承担视觉到语言空间的几何坐标转换，细微的数值截断都会导致多模态语义对齐漂移。工程中常将投影层保留为高精度（FP16/BF16）或采用极细粒度量化。
- **模态间空间几何对齐**：当使用旋转类算法（如 [QuaRot](../../quantization_algorithms/quarot/term_quarot.md)）消除文本侧激活离群值时，视觉侧输出特征必须同步施加相应的正交变换，以确保送入语言解码器前两者的特征空间依然严格对齐。
- **组件分级量化策略**：视觉编码器多为密集前向计算（受限于计算吞吐），语言解码器长序列生成受限于访存带宽，需针对性配置不同的量化比特与算法策略。

---

## 3. 关联流程

- [《多模态理解模型（VLM）量化使用指南》](./usage_vision_transformer_quantization.md)：VLM 量化的完整操作流程。
- [《VLM 模型接入量化流程指南》](./integration_guide_vision_transformer_quantization.md)：将新视觉语言模型接入量化流程的开发指导。
- [《一键量化完整指南》](../../../user_guide/usage_one_click_quantization.md)：涵盖各类模型的一键量化流程。

---

## 4. 关联词条

- [PTQ](../term_ptq.md)：上位概念，视觉语言模型量化属于训练后量化的具体应用。
- [LLM 量化](../llm/term_large_language_model_quantization.md)：配套概念，VLM 的语言解码部分沿用 LLM 量化方案。
- [DiT 量化](../dit/term_diffusion_transformer_quantization.md)：同类概念，多模态生成模型的量化。
- [QuaRot](../../quantization_algorithms/quarot/term_quarot.md)：配套算法，涉及模态对齐投影层的旋转变换。
- [权重转换](../convert/term_weight_conversion.md)：配套术语，量化权重的离线重构与格式转换。

---

## 5. 参考文档

1. Dosovitskiy A, et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR 2021. https://arxiv.org/abs/2010.11929
2. Xiao G, et al. "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models." arXiv:2211.10438, 2022. https://arxiv.org/abs/2211.10438
3. Liu H, et al. "Visual Instruction Tuning (LLaVA)." NeurIPS 2023. https://arxiv.org/abs/2304.08485
