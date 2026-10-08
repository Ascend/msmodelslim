# Diffusion Transformer 量化

> **词条类别**：[量化基础概念](../README.md)<br>
> **英文名称**：Diffusion Transformer Quantization<br>
> **英文缩写**：DiT Quantization<br>
> **应用领域**：多模态生成模型（文生图/文生视频）推理加速、模型压缩

---

## 1. 概述

Diffusion Transformer 量化（[Diffusion Transformer Quantization](./term_diffusion_transformer_quantization.md)）是指针对以 Transformer 为骨干网络的扩散生成模型（如 FLUX、HunyuanVideo、Wan2.2 等），应用[训练后量化（PTQ）](../term_ptq.md)将权重与高维激活张量转换为低比特浮点或整数格式（如 MXFP8、FP8、INT8）的技术。

---

## 2. 词条介绍

### 2.1 模型与推理特点

Diffusion Transformer（DiT）使用 Transformer 结构完全替代传统扩散模型中的 U-Net 骨干，具备以下特征：

- **多步迭代去噪机制**：生成单张图像或连续视频需执行数十步（如 20~50 步）去噪前向传播，每一步输入不同的噪声时间步（Timestep）嵌入。
- **无自回归与 KV Cache**：DiT 为密集矩阵计算的纯前向网络，不存在自回归文本生成的 KV Cache 显存膨胀问题，推理延迟受限于高算力吞吐（Compute-bound）。
- **多条件分支与专家混合**：先进 DiT 模型（如 Wan2.2、FLUX）常包含文本编码交叉注意力通道，或针对不同噪声阶段解耦的多专家（如 high-noise / low-noise expert）网络。

### 2.2 量化核心特点

与 LLM/VLM 的自回归或单次前向不同，DiT 模型的量化重点集中在动态时间步与高密集计算适配：

- **时间步感知校准（Timestep-Aware Calibration）**：去噪初期（高噪声）与后期（低噪声）激活值的动态范围和幅值分布具有强烈的非平稳性。量化校准必须在完整浮点去噪轨迹中均匀采样不同时间步的隐层激活，以避免全局 Scale 失真。
- **多专家解耦独立统计**：针对采用分阶段专家的扩散架构，不同专家的权重与激活分布差异显著，需按专家分别构建校准激活分布并独立求解量化参数。
- **高吞吐低精度浮点优先**：由于去噪迭代具有明显的误差累积效应，工业界多优先采用 MXFP8、FP8 等宽动态范围浮点格式（搭配微块 block 缩放），在大幅加速密集 GEMM 计算的同时严格控制图像/视频的感官退化。

---

## 3. 关联流程

- [《多模态生成模型（DiT）量化使用指南》](./usage_diffusion_transformer_quantization.md)：DiT 量化完整的操作流程与命令说明。
- [《DiT 模型接入量化流程指南》](./integration_guide_diffusion_transformer_quantization.md)：将新扩散生成模型接入量化流程的指导文档。
- [《一键量化完整指南》](../../../user_guide/usage_one_click_quantization.md)：涵盖各类生成模型的一键量化流程。

---

## 4. 关联词条

- [PTQ](../term_ptq.md)：上位概念，Diffusion Transformer 量化属于训练后量化的具体应用。
- [LLM 量化](../llm/term_large_language_model_quantization.md)：对比算法，自回归语言模型的量化（存在 KV Cache 与离群通道）。
- [VLM 量化](../vlm/term_vision_transformer_quantization.md)：同类概念，多模态理解模型的量化。
- [权重转换](../convert/term_weight_conversion.md)：配套术语，量化权重的离线重构与格式转换。

---

## 5. 参考文档

1. Peebles W, Xie S. "Scalable Diffusion Models with Transformers." ICCV 2023. <https://arxiv.org/abs/2212.09748>
2. Esser P, et al. "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis." arXiv:2403.03206, 2024. <https://arxiv.org/abs/2403.03206>
