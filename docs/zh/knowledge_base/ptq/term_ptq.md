# 训练后量化（PTQ）术语百科词条

> **词条类别**：[量化基础概念](../README.md)<br>
> **英文名称**：Post-Training Quantization<br>
> **英文缩写**：PTQ<br>
> **应用领域**：模型压缩、推理加速

---

## 1. 概述

训练后量化（Post-Training Quantization，PTQ）是指在模型**已完成训练**后，仅通过少量校准数据统计模型权重与激活值的分布，将浮点参数映射到低比特整数或低精度浮点格式，从而压缩模型体积、降低推理时显存与算力开销的量化方式。PTQ 不需要重新训练模型，因此成本低、上线快，是大模型推理部署的主流方案。PTQ 的核心特征包括：无需训练、依赖校准数据、精度-压缩比的权衡，以及低比特硬件的加速适配。

## 2. 词条介绍

PTQ 是深度学习模型部署中最重要的模型压缩技术之一。随着大语言模型（LLM）、多模态模型（VLM、DiT）参数规模持续增长，模型推理对显存和算力的需求急剧攀升，PTQ 因其低成本、高回报的特性成为工业界首选的加速方案。与训练感知量化（QAT）相比，PTQ 无需访问完整训练集和训练基础设施，仅凭少量校准数据即可在数小时内完成千亿参数级模型的量化，是精度与工程成本折中的主流选择。

从量化格式看，PTQ 既支持 INT8、INT4 等低比特整数格式，也支持 FP8、MXFP8 等低精度浮点格式：整数格式压缩比高、面向通用整型算力，浮点格式动态范围宽、与新一代 AI 芯片的浮点算力天然匹配。从量化对象看，可分为仅量化权重的 weight-only 方案（如 W4A16，主打显存压缩）与权重、激活同时量化的方案（如 W8A8、W4A8，兼顾显存与算力加速）。从量化粒度看，权重支持 per-tensor / per-channel / per-group，激活支持 per-tensor / per-token，粒度越细则精度越高，但 scale 的存储与计算开销也相应增大。

PTQ 的典型流程为：准备与推理数据分布一致的校准集 → 前向统计权重与激活的分布，必要时做离群值抑制等分布整形 → 逐层选择量化算法与量化参数并完成量化 → 导出量化权重并适配目标推理框架。

PTQ 按模型架构可分为三大类：

- **[LLM 量化](./llm/term_large_language_model_quantization.md)**：针对 Decoder-only 自回归语言模型，支持逐层量化与离群值抑制，量化方案成熟。
- **[VLM 量化](./vlm/term_vision_transformer_quantization.md)**：涵盖视觉编码器与语言模型，需多模态校准数据，视觉组件对量化误差敏感。
- **[DiT 量化](./dit/term_diffusion_transformer_quantization.md)**：针对 Diffusion Transformer 主干的文生图/文生视频模型，通过去噪管线生成校准数据，支持多专家独立量化。

此外，PTQ 生成的量化权重可通过[权重转换](./convert/term_weight_conversion.md)工具做进一步的格式/精度变换，以适配不同推理框架。

PTQ 的技术挑战主要集中在以下几个方面：

- **离群值抑制**——激活值中的离群特征会显著放大量化误差，需配合 [SmoothQuant](../quantization_algorithms/smooth_quant/term_smooth_quant.md) 等预处理算法。
- **校准数据代表性**——校准集分布需与推理数据分布一致，分布失配会导致统计出的量化参数不准确，直接影响量化后精度。
- **低比特精度维持**——INT4 及更低比特量化在保持模型质量方面仍面临挑战，常需结合敏感度分析对关键层回退到更高精度。
- **量化误差跨层累积**——单层引入的量化误差会随前向传播逐层传递并放大，长序列、深网络场景下误差累积尤为明显。
- **校准与部署开销**——超大模型的校准过程本身需要加载浮点权重并执行前向传播，显存与时间开销可观，需借助逐层量化、分布式校准等手段控制。
- **硬件与推理框架适配**——量化格式需与目标芯片的算子支持、scale 融合方式及权重布局匹配，否则无法获得真实的加速收益。

## 3. 关联流程

- [《LLM 量化使用指南》](./llm/usage_large_language_model_quantization.md)：LLM 量化的完整操作流程。
- [《VLM 量化使用指南》](./vlm/usage_vision_transformer_quantization.md)：VLM 量化的完整操作流程。
- [《DiT 量化使用指南》](./dit/usage_diffusion_transformer_quantization.md)：DiT 量化的完整操作流程。
- [《权重转换使用指南》](./convert/usage_weight_conversion.md)：权重转换的完整操作流程。
- [《一键量化完整指南》](../../user_guide/usage_one_click_quantization.md)：涵盖 LLM、VLM、DiT 的一键量化总体流程。

## 4. 关联词条

- [LLM 量化](./llm/term_large_language_model_quantization.md)：下位概念，PTQ 在 Decoder-only 自回归模型上的具体应用。
- [VLM 量化](./vlm/term_vision_transformer_quantization.md)：下位概念，PTQ 在多模态理解模型上的具体应用。
- [DiT 量化](./dit/term_diffusion_transformer_quantization.md)：下位概念，PTQ 在多模态生成模型上的具体应用。
- [权重转换](./convert/term_weight_conversion.md)：配套术语，PTQ 产出的量化权重可通过该工具做格式/精度变换。

## 5. 参考文档

1. [《msModelSlim 一键量化完整指南》](../../user_guide/usage_one_click_quantization.md)：msModelSlim 一键量化总体流程。
2. Gholami A, et al. "A Survey of Quantization Methods for Efficient Neural Network Inference." arXiv:2103.13630, 2021. https://arxiv.org/abs/2103.13630
3. Nagel M, et al. "A White Paper on Neural Network Quantization." arXiv:2106.08295, 2021. https://arxiv.org/abs/2106.08295
