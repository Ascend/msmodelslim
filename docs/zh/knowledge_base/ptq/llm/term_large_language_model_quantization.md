# 大语言模型量化

> **词条类别**：[量化基础概念](../README.md)<br>
> **英文名称**：Large Language Model Quantization<br>
> **英文缩写**：LLM Quantization<br>
> **应用领域**：大语言模型推理加速、模型压缩

---

## 1. 概述

大语言模型量化（[Large Language Model Quantization](./term_large_language_model_quantization.md)）是指针对以 Transformer Decoder-only 为主流架构的自回归文本生成模型（如 LLaMA、Qwen、DeepSeek 等），应用[训练后量化（PTQ）](../term_ptq.md)将权重和激活值从高精度浮点格式（FP16/BF16）转换为低比特格式（如 INT8、INT4、FP8）的技术。该技术以激活离群值抑制、低开销校准和权重-激活灵活配比为核心特征，旨在大幅降低模型显存占用并突破自回归解码阶段的内存带宽瓶颈。

---

## 2. 词条介绍

### 2.1 模型与推理特点

主流大语言模型具有以下主要架构与运行时特征：

- **Decoder-only 深度堆叠**：由多层同质的 Transformer 解码块（包含自注意力机制与前馈网络 MLP）级联而成，网络结构规整统一。
- **两阶段自回归推理**：包含预填充（Prefill）与逐 Token 生成的解码（Decode）阶段。Decode 阶段受限于访存带宽（Memory-bound），每生成一个 Token 均需完整读取全量权重。
- **KV Cache 显存膨胀**：历史 Token 的 Key/Value 状态需缓存在显存中，随着批大小（Batch Size）和上下文长度增长，KV Cache 显存占用甚至会超过权重本身。

### 2.2 量化核心特点

针对 LLM 的架构与运行时瓶颈，其量化方案具备以下关键特征：

- **权重量化解决访存瓶颈**：在小 Batch 场景下，通过 Weight-only 量化显著减少单步前向传播的权重搬运量，实现显存占用大幅降低与生成速度提升。
- **特定通道离群值（Outliers）显著**：深层网络激活值中，极少部分通道的幅值远高于其他通道（高达数百倍）。直接量化会导致巨大精度衰减，通常需结合 [SmoothQuant](../../quantization_algorithms/smooth_quant/term_smooth_quant.md)（权重-激活平滑缩放）或 [QuaRot](../../quantization_algorithms/quarot/term_quarot.md)（正交旋转消除离群值）进行分布整形。
- **轻量校准与无数据适配**：校准过程无需海量语料，通常仅需数十至数百条具有通用代表性的文本序列即可统计稳定的量化 Scale 参数。
- **KV Cache 量化协同**：除常规 Linear 层（QKV、Dense、MLP）外，将 KV Cache 压缩至 INT8/FP8 格式可提升长文本推理的吞吐量。

---

## 3. 关联流程

- [《LLM 量化使用指南》](./usage_large_language_model_quantization.md)：LLM 量化的完整操作流程。
- [《LLM 模型接入量化流程指南》](./integration_guide_large_language_model_quantization.md)：将新语言模型接入量化流程的开发指导。
- [《一键量化完整指南》](../../../user_guide/usage_one_click_quantization.md)：涵盖各类模型的一键量化流程。

---

## 4. 关联词条

- [PTQ](../term_ptq.md)：上位概念，大语言模型量化属于训练后量化的具体应用。
- [VLM 量化](../vlm/term_vision_transformer_quantization.md)：同类概念，多模态理解模型的量化（其文本端沿用 LLM 量化原理）。
- [DiT 量化](../dit/term_diffusion_transformer_quantization.md)：对比算法，扩散生成模型的量化（非自回归去噪范式）。
- [SmoothQuant](../../quantization_algorithms/smooth_quant/term_smooth_quant.md)：配套算法，解决 LLM 激活离群值的经典平滑算法。
- [QuaRot](../../quantization_algorithms/quarot/term_quarot.md)：配套算法，通过正交旋转矩阵消除激活离群值的算法。
- [权重转换](../convert/term_weight_conversion.md)：配套术语，量化权重的离线重构与格式转换。

---

## 5. 参考文档

1. Xiao G, et al. "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models." arXiv:2211.10438, 2022. https://arxiv.org/abs/2211.10438
2. Frantar E, et al. "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." arXiv:2210.17323, 2022. https://arxiv.org/abs/2210.17323
3. Ashkboos S, et al. "QuaRot: Outlier-Free 4-Bit Inference in Large Language Models." arXiv:2404.00456, 2024. https://arxiv.org/abs/2404.00456
