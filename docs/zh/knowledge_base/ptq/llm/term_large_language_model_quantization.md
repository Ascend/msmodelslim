# 大语言模型（LLM）量化术语百科词条

> **词条类别**：[量化基础概念](../README.md)<br>
> **英文名称**：Large Language Model Quantization<br>
> **英文缩写**：LLM Quantization<br>
> **应用领域**：大语言模型推理加速、模型压缩

---

## 1. 概述

大语言模型（LLM）量化是指对 Decoder-only 架构的自回归语言模型（如 LLaMA、Qwen、GLM、DeepSeek 等）进行[训练后量化（PTQ）](../term_ptq.md)的过程，将模型权重和激活值从浮点表示映射到低比特整数（如 INT8、INT4）或低精度浮点格式（如 FP8、MXFP8），从而降低模型部署时的显存占用与推理延迟。LLM 量化以逐层量化和离群值抑制为主要特征，是 PTQ 中技术最成熟的形态。

## 2. 模型特点

主流 LLM（如 LLaMA、Qwen、GLM、DeepSeek 等）虽然在注意力实现、位置编码、激活函数等细节上各有差异，但整体架构高度一致：均为 Transformer Decoder-only 结构，依靠逐 token 的自回归方式生成文本。这类模型参数规模庞大、层结构规整、推理过程高度串行，这些特性共同决定了 LLM 量化方案的走向。具体而言：

- **Decoder-only 自回归架构**：逐 token 生成，每个位置的预测依赖前序已生成 token，与编码器-解码器架构相比结构更简洁。
- **参数规模大**：从数十亿到数千亿参数，以 FP16/BF16 存储时显存需求极高（70B 权重约 140GB）。
- **结构高度同质化**：由大量结构相同的 DecoderLayer 堆叠而成（self-attention 的 QKV/O 投影 + MLP 的 gate/up/down 投影），量化逻辑可复用。
- **带 KV cache**：推理时缓存历史 token 的键值对，显存占用随序列长度增长，但 KV cache 不参与静态权重量化。
- **动态自回归推理**：前向传播高度串行（逐 token），计算密集，对低比特整数加速敏感。

## 3. 量化特点

LLM 的量化方案与上述模型特点紧密对应：DecoderLayer 结构同质化使量化逻辑可以逐层复用，超大参数规模带来的显存压力推动了逐层加载与分布式量化，激活值中的离群分布则决定了离群值抑制是精度关键，而自回归推理对算力的持续需求使权重与激活的低比特表示成为加速收益的主要来源。在长期的工程实践中，LLM 量化形成了以下鲜明特点：

- **逐层量化**：按 DecoderLayer 逐层加载、逐层量化、逐层落盘，显存压力小，且天然支持 DP 并行分布式量化。
- **权重量化 + 激活量化可解耦**：权重视显存压缩（W4A16 仅量化权重），激活视推理加速（W8A8 权重与激活同时量化），两者可独立配置。
- **离群值抑制是精度关键**：激活值中存在显著离群特征，直接量化精度损失大，需配合 SmoothQuant 等预处理算法；也可采用旋转量化（QuaRot）从几何上消除离群值。
- **校准数据轻量**：仅需 128~512 条文本样本即可完成校准，校准集质量直接决定量化精度。
- **量化粒度多样**：权重支持 per_channel / per_group，激活支持 per_tensor / per_token，可按精度-效率权衡灵活选择。
- **支持多种 runner 模式**：MODEL_WISE、LAYER_WISE、DP_LAYER_WISE，满足不同显存与并行的部署约束。

## 4. 关联流程

- [《LLM 量化使用指南》](./usage_large_language_model_quantization.md)：LLM 量化完整的操作流程与命令行说明。
- [《LLM 模型接入量化流程指南》](./integration_guide_large_language_model_quantization.md)：将新 Decoder-only 模型接入量化流程的完整开发指南。
- [《一键量化完整指南》](../../../user_guide/usage_one_click_quantization.md)：涵盖 LLM 与多模态模型的一键量化流程，默认集成 LLM 量化方案。

## 5. 关联词条

- [PTQ](../term_ptq.md)：上位概念，LLM 量化属于训练后量化的一种具体应用。
- [VLM 量化](../vlm/term_vision_transformer_quantization.md)：同类概念，多模态理解模型的量化，文本解码器部分采用与 LLM 相同的逐层量化方案。
- [DiT 量化](../dit/term_diffusion_transformer_quantization.md)：对比算法，去噪式生成模型的量化，校准策略与 LLM 完全不同。
- [权重转换](../convert/term_weight_conversion.md)：后续术语，对已有 LLM 量化权重做格式/精度变换。

## 6. 参考文档

1. Xiao G, et al. "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models." arXiv:2211.10438, 2022. https://arxiv.org/abs/2211.10438
2. Frantar E, et al. "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." arXiv:2210.17323, 2022. https://arxiv.org/abs/2210.17323
