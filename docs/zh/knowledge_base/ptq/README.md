# 训练后量化（PTQ）知识库

## 什么是训练后量化（PTQ）

训练后量化（Post-Training Quantization，PTQ）是指在模型**已完成训练**后，仅通过少量校准数据统计模型权重与激活值的分布，将浮点参数映射到低比特整数或低精度浮点格式，从而压缩模型体积、降低推理时内存与算力开销的量化方式。与训练感知量化（QAT）不同，PTQ 不需要重新训练模型，因此成本低、上线快，是大模型推理加速的主流方案。

## 基础概念词条

| 分类 | 说明 | 入口 |
|------|------|------|
| 基础概念 | 训练后量化（PTQ）的基本概念、技术特点与适用范围 | [《训练后量化（PTQ）词条》](term_ptq.md) |

## 量化词条

不同模型类型的量化关注点与实现路径差异较大，msModelSlim 将相关词条按量化对象和工具能力划分为以下四类：

| 分类 | 说明 | 入口 |
|------|------|------|
| LLM 量化 | 一般为 Decoder-only 自回归结构，权重与激活量化成熟 | [《LLM 量化词条》](llm/term_large_language_model_quantization.md) |
| VLM 量化 | 视觉编码器 + 语言模型，需多模态校准数据 | [《VLM 量化词条》](vlm/term_vision_transformer_quantization.md) |
| DiT 量化 | Diffusion Transformer 主干的文生图 / 文生视频 | [《DiT 量化词条》](dit/term_diffusion_transformer_quantization.md) |
| 权重转换 | 离线、data-free 的格式 / 精度变换 | [《权重转换词条》](convert/term_weight_conversion.md) |

## 使用指南

| 分类 | 说明 | 入口 |
|------|------|------|
| LLM 量化 | 大语言模型一键量化流程 | [《LLM 量化使用指南》](llm/usage_large_language_model_quantization.md) |
| VLM 量化 | 多模态理解模型量化流程 | [《VLM 量化使用指南》](vlm/usage_vision_transformer_quantization.md) |
| DiT 量化 | 多模态生成模型量化流程 | [《DiT 量化使用指南》](dit/usage_diffusion_transformer_quantization.md) |
| 权重转换 | 已有权重的格式 / 精度转换流程 | [《权重转换使用指南》](convert/usage_weight_conversion.md) |

## 模型接入指南

| 分类 | 说明 | 入口 |
|------|------|------|
| LLM 接入 | 新大语言模型（LLM）接入 msModelSlim 量化流程 | [《LLM 模型接入量化流程指南》](llm/integration_guide_large_language_model_quantization.md) |
| VLM 接入 | 新多模态视觉语言模型（VLM）接入 msModelSlim 量化流程 | [《VLM 模型接入量化流程指南》](vlm/integration_guide_vision_transformer_quantization.md) |
| DiT 接入 | 新多模态生成模型（DiT）接入 msModelSlim 量化流程 | [《DiT 模型接入量化流程指南》](dit/integration_guide_diffusion_transformer_quantization.md) |
