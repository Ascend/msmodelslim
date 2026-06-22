# MindStudio ModelSlim 使用指南

<br>

## 1. 功能概述

**MindStudio ModelSlim（msModelSlim）** 是昇腾生态下的高性能模型压缩工具，支持稠密LLM、MoE及多模态模型的量化与压缩，开发者可通过Python API快速调优并导出适配MindIE、vLLM-Ascend等框架的模型，在昇腾AI处理器上高效部署。

## 2. 使用方法

### 2.1 一键量化

本功能基于 msModelSlim 在不同架构下的能力支持情况，提供相应的使用说明与操作指引。

详细使用方法请参见《[一键量化使用指南](./feature_guide/quick_quantization_v1/usage.md)》，您可在左侧导航栏中选择具体功能模块查看相关介绍。

### 2.2 自主量化

面向希望将自有模型接入 msModelSlim 的开发者，本节提供完整的模型集成指导。

建议您首先阅读《[架构说明](../development_guide/architecture.md)》以了解整体设计逻辑，随后参考以下文档完成模型接入：

- 《[LLM 大模型接入指南](../development_guide/integrating_models.md)》
- 《[多模态理解模型接入指南](../development_guide/integrating_multimodal_understanding_model.md)》
