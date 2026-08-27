# MindStudio ModelSlim User Guide

<!-- md-trans-meta sourceCommit=bf41aab80fdad8030fb6169433278e8e9eb99356 translatedAt=2026-08-20T09:24:44.770Z pushedAt=2026-08-20T09:25:18.052Z -->

<br>

## 1. Functional Overview

**MindStudio ModelSlim (msModelSlim)** is a high-performance model compression tool in the Ascend ecosystem. It supports quantization and compression of dense LLMs, MoE models, and multimodal models. Developers can quickly tune models through Python APIs and export models adapted to frameworks such as MindIE and vLLM-Ascend for efficient deployment on Ascend AI Processors.

msModelSlim provides the following core functional modules, which can be selected based on actual requirements:

| Functional Module | Description | Applicable Scenarios |
|---------|------|---------|
| [One-Click Quantization (V1)](#21-one-click-quantization) | Integrates best practices for mainstream large model quantization, automatically matches optimal configurations, and works out of the box. | Recommended as the first choice, applicable to models that support one-click quantization |
| [Traditional Quantization (V0)](#22-traditional-quantization) | Performs quantization through Python scripts and supports fine-grained parameter adjustment. | Models not yet supported by one-click quantization |
| [Sensitive Layer Analysis](#23-sensitive-layer-analysis) | Evaluates the quantization sensitivity of each layer from multiple dimensions, providing data support for quantization configuration tuning. | Quantization accuracy tuning and accuracy bottleneck locating |
| [Automatic Tuning](#24-automatic-tuning) | Automatically and iteratively searches for quantization configurations based on accuracy targets, with full-process automation. | When accuracy does not meet the target, eliminating the need for repeated manual parameter adjustment |
| [Autonomous Quantization](#25-autonomous-quantization) | Provides a standard integration framework, supporting developers in quickly integrating their own models. | Integrating custom models into msModelSlim |
| [Quantization Algorithm](#3-quantization-algorithm) | Describes in detail the various quantization and outlier suppression algorithms supported by msModelSlim. | In-depth understanding of quantization principles and algorithm selection |
| [Quantization Format](#4-quantization-format) | Describes the storage format of quantized weights and compatibility with various frameworks. | Multi-framework deployment and weight conversion |

## 2. Usage

### 2.1 One-Click Quantization

This feature provides corresponding usage instructions and operation guidance based on the capability support of msModelSlim under different architectures.

For detailed usage, refer to *[One-Click Quantization User Guide](./feature_guide/quick_quantization_v1/usage.md)*. You can select a specific functional module in the left navigation pane to view the related introduction.

### 2.2 Traditional Quantization

Traditional Quantization (V0) performs quantization through Python scripts and is applicable to model scenarios not yet supported by One-Click Quantization.

For detailed usage, see *[Traditional Quantization User Guide](./feature_guide/traditional_quantization_v0/README.md)*.

### 2.3 Sensitive Layer Analysis

Sensitive layer analysis is used to evaluate the sensitivity of each layer to quantization from multiple dimensions, helping accurately locate layers that should be rolled back or have their bit width increased, thereby providing data support for quantization configuration tuning.

For detailed usage, refer to *[Sensitive Layer Analysis User Guide](./feature_guide/sensitive_layer_analysis/usage.md)*.

### 2.4 Automatic Tuning

The Automatic Tuning feature can automatically and iteratively search for the optimal quantization configuration based on the accuracy target, automating the entire quantization and evaluation process without the need for repeated manual parameter adjustment.

For detailed usage, refer to *[Automatic Tuning User Guide](./feature_guide/auto_precision_tuning/usage.md)*.

### 2.5 Autonomous Quantization

This section provides complete model integration guidance for developers who want to integrate their own models into msModelSlim.

It is recommended that you first read *[Architecture Description](../development_guide/architecture.md)* to understand the overall design logic, and then refer to the following documents to complete model integration:

- *[LLM Integration Guide](../development_guide/integrating_models.md)*

- *[Multimodal Understanding Model Integration Guide](../development_guide/integrating_multimodal_understanding_model.md)*

- *[Multimodal Generation Model Integration Guide](../development_guide/integrating_multimodal_generation_model.md)*

## 3. Quantization Algorithm

msModelSlim supports multiple quantization algorithms and outlier suppression algorithms. For details, see *[Quantization Algorithm Description](./quantization_algorithms/README.md)*.

## 4. Quantization Format

msModelSlim supports multiple quantized weight storage formats and weight conversion between different inference frameworks. For details, see *[Quantization Format Description](./quantization_formats/README.md)*.
