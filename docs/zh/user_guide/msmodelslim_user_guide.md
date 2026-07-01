# MindStudio ModelSlim 使用指南

<br>

## 1. 功能概述

**MindStudio ModelSlim（msModelSlim）** 是昇腾生态下的高性能模型压缩工具，支持稠密LLM、MoE及多模态模型的量化与压缩，开发者可通过Python API快速调优并导出适配MindIE、vLLM-Ascend等框架的模型，在昇腾AI处理器上高效部署。

msModelSlim 提供以下核心功能模块，可根据实际需求选择使用：

| 功能模块 | 说明 | 适用场景 |
|---------|------|---------|
| [一键量化（V1）](#21-一键量化) | 集成主流大模型量化最佳实践，自动匹配最优配置，开箱即用 | 推荐首选，适用于支持一键量化的模型 |
| [传统量化（V0）](#22-传统量化) | 通过 Python 脚本方式执行量化，支持精细化参数调整 | 一键量化暂不支持的模型 |
| [敏感层分析](#23-敏感层分析) | 多维度评估各层量化敏感度，为量化配置调优提供数据支撑 | 量化精度调优、定位精度瓶颈 |
| [自动调优](#24-自动调优) | 根据精度目标自动迭代搜索量化配置，全流程自动化 | 精度不达标时无需人工反复调参 |
| [自主量化](#25-自主量化) | 提供标准接入框架，支持开发者将自有模型快速集成 | 自有模型接入 msModelSlim |
| [量化算法](#3-量化算法) | 详细说明 msModelSlim 支持的各类量化及离群值抑制算法 | 深入理解量化原理与算法选型 |
| [量化格式](#4-量化格式) | 说明量化权重的存储格式及各框架兼容性 | 多框架部署、权重转换 |

## 2. 使用方法

### 2.1 一键量化

本功能基于 msModelSlim 在不同架构下的能力支持情况，提供相应的使用说明与操作指引。

详细使用方法请参见《[一键量化使用指南](./feature_guide/quick_quantization_v1/usage.md)》，您可在左侧导航栏中选择具体功能模块查看相关介绍。

### 2.2 传统量化

传统量化（V0）通过 Python 脚本方式执行量化，适用于一键量化尚未支持的模型场景。

详细使用方法请参见《[传统量化使用指南](./feature_guide/traditional_quantization_v0/README.md)》。

### 2.3 敏感层分析

敏感层分析用于多维度评估各层对量化的敏感程度，帮助精准定位应回退或提升位宽的层，为量化配置调优提供数据支撑。

详细使用方法请参见《[敏感层分析使用指南](./feature_guide/sensitive_layer_analysis/usage.md)》。

### 2.4 自动调优

自动调优功能可根据精度目标自动迭代搜索最优量化配置，量化与评估全流程自动化，无需人工反复调参。

详细使用方法请参见《[自动调优使用指南](./feature_guide/auto_precision_tuning/usage.md)》。

### 2.5 自主量化

面向希望将自有模型接入 msModelSlim 的开发者，本节提供完整的模型集成指导。

建议您首先阅读《[架构说明](../development_guide/architecture.md)》以了解整体设计逻辑，随后参考以下文档完成模型接入：

- 《[LLM 大模型接入指南](../development_guide/integrating_models.md)》
- 《[多模态理解模型接入指南](../development_guide/integrating_multimodal_understanding_model.md)》

## 3. 量化算法

msModelSlim 支持多种量化算法及离群值抑制算法，详细说明请参见《[量化算法说明](./quantization_algorithms/README.md)》。

## 4. 量化格式

msModelSlim 支持多种量化权重存储格式，支持在不同推理框架间进行权重转换，详细说明请参见《[量化格式说明](./quantization_formats/README.md)》。
