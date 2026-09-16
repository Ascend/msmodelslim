# 训练后量化（PTQ）

> **词条类别**：[量化基础概念](../README.md)<br>
> **英文名称**：Post-Training Quantization<br>
> **英文缩写**：PTQ<br>
> **应用领域**：模型压缩、推理加速

---

## 1. 概述

训练后量化（Post-Training Quantization，[PTQ](./term_ptq.md)）是指在深度学习模型**已完成训练**后，无需重新训练模型参数，仅通过少量代表性校准数据统计模型权重与激活值的数值分布，将高精度浮点张量（如 FP16、BF16）映射为低比特整数（如 INT8、INT4）或低精度浮点（如 FP8、MXFP8）的技术。PTQ 具有工程成本低、转换耗时短、即插即用等优势，是大模型推理部署中最通用的量化方案。

---

## 2. 词条介绍

### 2.1 基本原理与范式

PTQ 的核心思想是通过离线统计校准集前向传播中的张量极值、直方图或均方误差，计算出最佳的缩放因子（Scale）与零点（Zero Point），建立高精度与低精度数值之间的映射关系。

根据量化对象与算力目标，PTQ 主要包含两种范式：

- **权重量化（Weight-Only Quantization）**：仅将模型权重压缩为低比特（如 W8A16），计算时反量化为浮点执行。主要解决显存容量受限与内存带宽瓶颈（Memory-bound），大幅降低模型权重占用的显存空间，常见于批大小（Batch Size）较小的 LLM 生成阶段。
- **权重与激活全量化（Weight-Activation Quantization）**：权重与激活均量化为低比特（如 W8A8、W4A8），直接调用硬件低精度矩阵乘算子（GEMM/MatMul）。兼顾显存占用缩减与计算加速（Compute-bound），在大 Batch 或多模态推理中加速收益显著。

### 2.2 主流大模型架构的 PTQ 特点对比

不同模型架构的计算模式与数据流动特点各异，PTQ 呈现出高度差异化的量化重点：

| 架构类型 | 核心模型结构 | 推理计算特征 | 核心量化挑战与应对 |
| :--- | :--- | :--- | :--- |
| **[LLM 量化](./llm/term_large_language_model_quantization.md)** | Transformer Decoder-only | 自回归逐 token 生成，内存带宽受限（Memory-bound）；带 KV Cache | 激活值存在特定通道的固定离群点（Outliers），需采用 [SmoothQuant](../quantization_algorithms/smooth_quant/term_smooth_quant.md)、[QuaRot](../quantization_algorithms/quarot/term_quarot.md) 等离群值抑制或旋转算法；需结合 KV Cache 量化。 |
| **[VLM 量化](./vlm/term_vision_transformer_quantization.md)** | ViT 视觉编码器 + 文本 Decoder | 跨模态异构；图像特征经投影层融入文本空间 | 视觉编码器与文本解码器分布特性差异大，模态对齐投影层（Merger/Projection）对量化敏感，需配合多模态校准数据并进行针对性敏感度保护或模态间旋转协同。 |
| **[DiT 量化](./dit/term_diffusion_transformer_quantization.md)** | Diffusion Transformer 主干 | 多步迭代去噪，无 KV Cache，计算密集（Compute-bound） | 激活分布随时间步（Timestep）高度动态非平稳，需覆盖多时间步的去噪轨迹进行校准；多专家模型需按专家分别统计分布。 |

### 2.3 技术挑战

- **激活离群值抑制**：深层 Transformer 激活值常存在跨通道幅值悬殊的离群通道，易导致常规量化截断误差或缩放因子被极值主导。
- **校准集分布代表性**：PTQ 不更新模型权重，统计参数高度依赖校准数据。若校准样本覆盖域不全，易导致泛化精度下降。
- **超低比特精度维持**：进入 INT4、INT3 等更低比特时，线性量化截断损失呈指数级上升，需结合二阶误差补偿或通道重排等进阶算法。
- **误差跨层累积**：单层局部量化误差在前向传播过程中会逐层扩散与放大，需通过端到端敏感度分析进行关键层精度保护。

---

## 3. 关联流程

- [《大语言模型（LLM）量化使用指南》](./llm/usage_large_language_model_quantization.md)：LLM 量化的完整操作流程。
- [《多模态理解模型（VLM）量化使用指南》](./vlm/usage_vision_transformer_quantization.md)：VLM 量化的完整操作流程。
- [《多模态生成模型（DiT）量化使用指南》](./dit/usage_diffusion_transformer_quantization.md)：DiT 量化的完整操作流程。
- [《权重转换使用指南》](./convert/usage_weight_conversion.md)：权重转换的完整操作流程。
- [《一键量化完整指南》](../../user_guide/usage_one_click_quantization.md)：涵盖各类模型的一键量化总体流程。

---

## 4. 关联词条

- [LLM 量化](./llm/term_large_language_model_quantization.md)：下位概念，针对自回归语言模型的 PTQ。
- [VLM 量化](./vlm/term_vision_transformer_quantization.md)：下位概念，针对多模态视觉语言模型的 PTQ。
- [DiT 量化](./dit/term_diffusion_transformer_quantization.md)：下位概念，针对扩散 Transformer 生成模型的 PTQ。
- [权重转换](./convert/term_weight_conversion.md)：配套术语，量化权重的离线重构与格式转换。
- [SmoothQuant](../quantization_algorithms/smooth_quant/term_smooth_quant.md)：配套算法，PTQ 中常用的激活离群值抑制技术。

---

## 5. 参考文档

1. Gholami A, et al. "A Survey of Quantization Methods for Efficient Neural Network Inference." arXiv:2103.13630, 2021. https://arxiv.org/abs/2103.13630
2. Nagel M, et al. "A White Paper on Neural Network Quantization." arXiv:2106.08295, 2021. https://arxiv.org/abs/2106.08295
3. Xiao G, et al. "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models." arXiv:2211.10438, 2022. https://arxiv.org/abs/2211.10438
