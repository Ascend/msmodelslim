<!-- md-trans-meta sourceCommit=652caa58cbf1feddc29ea6bd484adddf6557db00 translatedAt=2026-08-21T06:10:49.731Z pushedAt=2026-08-21T06:21:55.322Z -->

<h1 align="center"> MindStudio ModelSlim</h1>
<div align="center">
  <br />
  <img src="docs/assets/modelslim_slogan.png" alt="MindStudio ModelSlim Slogan" width="300" />
  <p align="center">
    <em>Simple, fast, and lean—msModelSlim is all you need.</em>
  </p>
  <p><b><span style="font-size:24px;">Ascend Model Compression Tool</span></b></p>

  <!-- Use a divider instead of a background -->

 [![Quick Start](https://badgen.net/badge/Quick%20Start/QuickStart/blue)](./docs/en/quick_start/quantization_quick_start.md)
 [![AI Q&A (DeepWiki)](https://badgen.net/badge/AI%20Q%26A/DeepWiki/blue)](https://deepwiki.com/Keithwwa/msmodelslim)
 [![AI Q&A (ZRead)](https://badgen.net/badge/AI%20Q%26A/ZRead/blue)](https://zread.ai/mindstudio-docs/master)
 [![Ascend Community](https://badgen.net/badge/Ascend%20Community/Community/blue)](https://www.hiascend.com/en/developer/software/mindstudio)
 [![Report Issues](https://badgen.net/badge/Report%20Issues/Issues/blue)](https://gitcode.com/Ascend/msmodelslim/issues/new)

</div>

English | [简体中文](./README.md)

## ✨ Latest News

<span style="font-size:14px;">

🔹 **[2026.06.01]**

- Added quantization support for the `InternVL3_5-38B` (W8A8) and `InternVL3_5-241B-A28B` (W8A8) models

- Added quantization support for the `Kimi-K2.6` (W4A8) model

🔹 **[2026.04.01]**

- Added quantization support for the `DeepSeek-V4-Flash` (W8A8) model

- Added quantization support for the `Kimi-K2.5` (W4A8) model

🔹 **[2026.03.01]**

- Newly Support quantization for the `GLM-4.6V` (W8A8) model

</span>

<details>
<summary>🗂️ Historical Updates (click to expand)</summary>

**February 2026**

- msModelSlim supports W8A8 quantization for Qwen3-Omni-30B-A3B-Thinking and Qwen3-Omni-30B-A3B-Instruct

- msModelSlim supports W8A8 quantization for Qwen2.5-Omni-7B

- msModelSlim supports W8A8 quantization for Qwen3.5-397B-A17B

- msModelSlim supports W4A8 quantization for GLM-5

- msModelSlim optimizes One-Click Quantization scenario recommendation

**January 2026**

- msModelSlim supports Qwen3-VL-32B-Instruct W8A8 quantization

**December 2025**

- msModelSlim supports quantization accuracy feedback auto-tuning, which can automatically search for the optimal quantization configuration based on accuracy requirements

- msModelSlim supports Custom Quantization for multimodal understanding models, enabling quantization integration of multimodal understanding models

- msModelSlim One-Click Quantization supports multi-card quantization and distributed layer-by-layer quantization, improving the quantization efficiency of large models

- msModelSlim supports DeepSeek-V3.2 W8A8 quantization, which can be executed on a single card with 64 GB video memory and 100 GB memory

- msModelSlim supports DeepSeek-V3.2-Exp W4A8 quantization, which can be executed on a single card with 64 GB video memory and 100 GB memory

- msModelSlim supports Qwen3-VL-235B-A22B W8A8 quantization

**November 2025**

- msModelSlim model adaptation supports plugin-based and configuration registration, and supports dependency pre-check

**October 2025**

- msModelSlim supports Qwen3-235B-A22B W4A8 and Qwen3-30B-A3B W4A8 quantization, and vLLM-Ascend supports inference deployment of quantized models

### 🗓️ September 2025

- msModelSlim supports DeepSeek-V3.2-Exp W8A8 quantization, which can be executed with a single card with 64GB video memory and 100GB memory only

- msModelSlim has resolved the issue of frequent abnormal tokens such as "game copies" in Qwen3-235B-A22B under W8A8 quantization

- msModelSlim supports DeepSeek R1 W4A8 per-channel quantization [Prototype]

- msModelSlim supports sensitivity analysis for large model quantization

**August 2025**

- msModelSlim supports One-Click Quantization for the Wan2.1 model

- msModelSlim supports layer-wise quantization for large models, significantly reducing memory usage during large model quantization

- msModelSlim supports the SSZ weight quantization algorithm for large models, which improves quantization accuracy by iteratively searching for the optimal scaling factors and offsets

</details>

## ℹ️ Overview

**MindStudio ModelSlim (msModelSlim)** is a high-performance model compression tool in the Ascend ecosystem. It supports quantization and compression of dense LLMs, MoE models, and multimodal models. Developers can quickly tune models through the msModelSlim tool and export models adapted to frameworks such as MindIE and vLLM-Ascend for efficient deployment on Ascend AI Processors.

## ⚙️ Feature Introduction

| Feature Name | Description |
|---------|--------|
| **One-Click Quantization** | Integrates best practices for mainstream large model quantization, supports multiple quantization types such as W4A8, W8A8, and W8A16, automatically matches the optimal configuration, and works out of the box. |
| **Custom Quantization** | Provides a standard integration framework, allowing developers to quickly integrate their own LLMs and multimodal models into the one-click quantization workflow. |
| **Sensitivity Analysis** | Evaluates the quantization sensitivity of each layer from multiple dimensions, accurately locates layers that should be rolled back or have their bit width increased, and provides data support for quantization configuration tuning. |
| **Auto-Tuning** | Automatically iterates and searches quantization configurations based on accuracy targets, automating the entire quantization and evaluation process without repeated manual parameter adjustment. |
| **Weight Conversion** | Performs format and precision conversion on existing quantized weights offline without a calibration set (for example, FP8→BF16, BF16→MXFP8). |

> **Model Support Overview**: For details about the models and quantization types supported by each feature, see [Model Support Matrix](./docs/en/user_guide/model_support/foundation_model_support_matrix.md).

## 🚀 Quick Start

To help users quickly complete large model quantization, see [msModelSlim Quick Start](./docs/en/quick_start/quantization_quick_start.md).

## 📦 Installation Guide

Describes the environment dependencies and installation methods of the tool. See [msModelSlim Tool Installation Guide](./docs/en/install_guide/install_guide.md).

## 📘 Usage Guide

For detailed usage of the tool, see [msModelSlim User Guide](./docs/en/user_guide/msmodelslim_user_guide.md).

## 💡 Typical Cases

To help users understand and master the tool through typical problem scenarios, see [msModelSlim Typical Cases](./docs/en/best_practices/basic_cases.md).

## ❓ FAQs

For common issues and solutions, see [FAQs](./docs/en/support/faq.md).

## 🌌 Intelligent Search

To improve document retrieval efficiency, we provide multiple efficient search methods:<br>
🔹 [AI Q&A (DeepWiki)](https://deepwiki.com/Keithwwa/msmodelslim): natural language Q&A to quickly grasp the project architecture and module relationships.<br>
🔹 [Precise Search (ReadTheDocs)](https://www.hiascend.com/document/detail/en/mindstudio/latest/TITools/msModelSlim/docs/en/getting_started/quantization_quick_start.md): full-text keyword search to directly access interfaces, parameters, and error messages.<br>

## 🛠️ Contribution Guide

For details, see [Contribution Guide](./docs/en/contributing/contributing_guide.md).

## ⚖️ Related Notes

🔹 [Release Notes](https://gitcode.com/Ascend/msmodelslim/releases)<br>
🔹 [License Notice](docs/en/legal/license_notice.md)<br>
🔹 [Security Statement](docs/en/legal/SECURITY.md)<br>
🔹 [Disclaimer](docs/en/legal/disclaimer.md)<br>

## 🤝 Suggestions and Communication

You are welcome to contribute to the community. If you have any questions or suggestions, submit them to [Issues](https://gitcode.com/Ascend/msmodelslim/issues), and we will reply as soon as possible. Thank you for your support.

|                                                                         Instant Interaction (WeChat Group)                                                                          |                                                                               Official News (Official Account)                                                                                | In-depth Support (Assistant/Forum)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| <img src="https://raw.gitcode.com/Ascend/docs/files/master/common/Writing_Template/figures/qr_code_wechat_work.png" width="120"><br><sub>*Scan the QR code to join the technical exchange group*</sub> | <img src="https://raw.gitcode.com/Ascend/docs/files/master/common/Writing_Template/figures/qr_code_wechat_official_account.png" width="120"><br><sub>*Scan the QR code to follow the official account*</sub> | Scan the QR code to join the group and follow the official account for the fastest communication platform for MindStudio users and developers:<br> **Quick Questions:** Discuss technical issues with community members in real time<br>**Stay Updated:** Get version release and feature update notifications as soon as possible<br> **Experience Sharing:** Exchange best practices and hands-on insights with developers  <br> <br> **More Support Channels**:👉 Ascend Assistant: [![WeChat](https://img.shields.io/badge/WeChat-07C160?style=flat-square&logo=wechat&logoColor=white)](https://gitcode.com/Ascend/msit/blob/master/docs/zh/figures/readme/xiaozhushou.png) 👉 Ascend Forum: [![Website](https://img.shields.io/badge/Website-%231e37ff?style=flat-square&logo=RSS&logoColor=white)](https://www.hiascend.com/forum/) |

## 🙏 Acknowledgments

This tool is jointly contributed by the following departments of Huawei:<br>
🔹 Ascend Computing MindStudio Development Department<br>
🔹 Ascend Computing Ecosystem Enablement Department<br>
🔹 Ascend Computing Technology Development Department<br>
🔹 2012 Laboratories<br>

Thanks to every PR from the community. Contributions are welcome!
