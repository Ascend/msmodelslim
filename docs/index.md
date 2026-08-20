# 欢迎来到 msModelSlim

<p align="center">
  <img src="assets/modelslim_slogan.png" alt="ModelSlim Slogan" width="340" />
</p>

MindStudio ModelSlim（昇腾模型压缩工具，msModelSlim），一个以加速为目标、压缩为技术、昇腾为根本的亲和压缩工具。包含量化和压缩等一系列推理优化技术，旨在加速大语言稠密模型、MoE模型、多模态理解模型、多模态生成模型等。 :computer:

昇腾 AI 开发者可通过调用 msModelSlim 提供的丰富 Python API，灵活实现算法适配与模型压缩，支持多种格式的权重导出及全流程的精度、性能调优。经过优化后的模型可无缝接入 MindIE、vLLM Ascend 等主流推理框架，在昇腾 AI 处理器上实现高效部署。 :zap:

## :star: 核心优势

- 🚀 **推理加速**：量化后显著降低显存占用，提升推理吞吐与部署成本效率（例如 DeepSeek-V3.2 单卡 64GB 显存、100GB 内存即可完成 W8A8 量化）。
- 🎯 **开箱即用**：集成主流大模型量化最佳实践，`msmodelslim quant` 一条命令完成量化。
- 🔧 **精度可控**：提供敏感层分析、自动调优与精度反馈闭环，量化精度可量化、可调优。
- 🧩 **生态友好**：导出的量化权重无缝接入 vLLM-Ascend 等主流推理框架。

## :loudspeaker: 最新消息

### 2025年12月

- msModelSlim 支持量化精度反馈自动调优，可根据精度需求自动搜索最优量化配置
- msModelSlim 支持自主量化多模态理解模型，支持多模态理解模型的量化接入
- msModelSlim 一键量化支持多卡量化，支持分布式逐层量化，提升大模型量化效率
- msModelSlim 支持 DeepSeek-V3.2 W8A8 量化，单卡64G显存、100G内存即可执行
- msModelSlim 支持 DeepSeek-V3.2-Exp W4A8 量化，单卡64G显存、100G内存即可执行
- msModelSlim 支持 Qwen3-VL-235B-A22B W8A8 量化

### 2025年11月

- msModelSlim 模型适配支持插件化和配置注册，支持依赖预检

### 2025年10月

- msModelSlim 支持 Qwen3-235B-A22B W4A8、Qwen3-30B-A3B W4A8 量化，vLLM Ascend 已支持量化模型推理部署

### 2025年9月

- msModelSlim 支持 DeepSeek-V3.2-Exp W8A8 量化，单卡64G显存，100G内存即可执行
- msModelSlim 现已解决 Qwen3-235B-A22B 在 W8A8 量化下频繁出现“游戏副本”等异常 token 的问题
- msModelSlim 支持 DeepSeek R1 W4A8 per-channel 量化【Prototype】
- msModelSlim 支持大模型量化敏感层分析

### 2025年8月

- msModelSlim 支持 Wan2.1 模型一键量化

- msModelSlim 支持大模型逐层量化，显著降低大模型量化内存占用

- msModelSlim 支持大模型 SSZ 权重量化算法，通过迭代搜索最优缩放因子和偏移量提升量化精度

> **注**：标注 **Prototype** 的特性尚未经过充分验证，可能存在不稳定或缺陷问题；标注 **beta** 的特性为非商用特性。

## msModelSlim 文档

欢迎使用 msModelSlim 文档。msModelSlim 是一款面向昇腾 AI 处理器的综合模型压缩与优化工具。
