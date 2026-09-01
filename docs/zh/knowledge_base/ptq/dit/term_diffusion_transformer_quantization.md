# Diffusion Transformer（DiT）量化术语百科词条

> **词条类别**：[量化基础概念](../README.md)<br>
> **英文名称**：Diffusion Transformer Quantization<br>
> **英文缩写**：DiT Quantization<br>
> **应用领域**：多模态生成模型（文生图/文生视频）推理加速

---

## 1. 概述

Diffusion Transformer（DiT）量化是指对以 Diffusion Transformer 为主干的文生图 / 文生视频模型（如 FLUX.1、HunyuanVideo、Wan2.2 等）进行[训练后量化（PTQ）](../term_ptq.md)的过程，将模型权重和激活值从浮点表示映射到低精度浮点格式（如 MXFP8）。与[LLM 量化](../llm/term_large_language_model_quantization.md)依赖文本校准数据集不同，DiT 量化通过运行完整浮点去噪推理管线生成校准数据，并针对多专家（multi-expert）模型按专家分别量化是其核心特征。

## 2. 模型特点

DiT 模型以 Transformer 替代传统 U-Net 作为扩散去噪主干，将图像或视频 patch 化后通过迭代去噪生成内容。与 LLM 的自回归范式不同，DiT 的推理是一个多步迭代过程，且部分模型采用多专家机制处理不同噪声水平，这些差异使 DiT 在模型结构与推理形态上自成一系，也从根本上决定了其量化路径的不同。具体而言：

- **Diffusion Transformer 主干**：以 Transformer 替代传统 U-Net 作为去噪主干，以 patch 化替代空间下采样，具有更好的可扩展性。
- **去噪推理管线**：推理时需多次迭代前向传播（通常 25~50 步），每次迭代计算量相当，整体推理延迟较高。
- **多专家架构**：部分模型（如 Wan2.2）使用 low_noise 和 high_noise 两个专家，分别处理不同噪声水平的去噪步，各专家需独立量化。
- **无 KV cache**：与自回归 LLM 不同，DiT 不缓存键值对，每次去噪步均独立计算完整注意力。
- **输出格式特定**：量化结果通常面向专有推理栈（如 MindIE-SD），而非通用 HuggingFace 格式。

## 3. 量化特点

DiT 的量化方案由其去噪推理形态直接塑造：多步迭代推理使校准数据必须通过运行完整浮点去噪管线来生成，激活分布随 timestep 演变的非平稳性要求量化参数覆盖整个去噪过程，而多专家架构则要求各专家独立统计与量化。与 LLM 量化相比，DiT 量化在数据处理、量化粒度与部署目标上均采取了一套独立的方案，具体特点如下：

- **去噪管线校准**：需运行完整浮点推理管线生成激活 dump 作为校准数据，而非使用文本或图像数据集，校准数据生成时间较长。
- **多专家独立量化**：对多专家模型，每个专家独立 dump 校准数据、独立量化、独立保存，各专家输出独立子目录。
- **Timestep 感知**：校准数据覆盖不同去噪步长（timestep），确保量化参数适应去噪过程中激活分布的非平稳性。
- **处理器链特异**：量化处理器链为 `online_quarot`（在线 QuaRot 旋转）→ `fa3_quant`（FA3 激活量化）→ `linear_quant`（线性层 MXFP8 量化），无 KV cache 量化环节。
- **per_block 量化粒度**：线性层采用 MXFP8 per_block 量化，权重与激活均按 block 粒度分组独立统计缩放因子。
- **仅支持单卡 LAYER_WISE**：DiT 量化不支持分布式 DP 或 MODEL_WISE runner，量化过程受单卡显存限制。
- **输出格式唯一**：仅支持 MindIE-SD 格式落盘，面向昇腾 NPU 推理栈。

## 4. 关联流程

- [《DiT 量化使用指南》](./usage_diffusion_transformer_quantization.md)：DiT 量化完整的操作流程与命令说明。
- [《DiT 模型接入量化流程指南》](./integration_guide_diffusion_transformer_quantization.md)：将新多模态生成模型接入量化流程的完整开发指南。
- [《一键量化完整指南》](../../../user_guide/usage_one_click_quantization.md)：涵盖多模态生成模型的一键量化流程，默认集成 DiT 量化方案。

## 5. 关联词条

- [PTQ](../term_ptq.md)：上位概念，DiT 量化属于训练后量化的一种具体应用。
- [LLM 量化](../llm/term_large_language_model_quantization.md)：对比算法，自回归语言模型的量化，校准策略与 DiT 的去噪管线校准完全不同。
- [VLM 量化](../vlm/term_vision_transformer_quantization.md)：同类概念，多模态理解模型的量化，同样涉及视觉 Transformer 结构。
- [权重转换](../convert/term_weight_conversion.md)：应用对象，DiT 量化权重可通过该工具调整格式/精度。

## 6. 参考文档

1. Peebles W, Xie S. "Scalable Diffusion Models with Transformers." ICCV 2023. https://arxiv.org/abs/2212.09748
