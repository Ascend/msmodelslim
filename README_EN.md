<h1 align="center"> MindStudio ModelSlim</h1>
<div align="center">
  <br />
  <img src="docs/assets/modelslim_slogan.png" alt="MindStudio ModelSlim Slogan" width="300" />
  <p align="center">
    <em>Simple, fast, and lean—msModelSlim is all you need.</em>
  </p>
  <p><b><span style="font-size:24px;">Ascend Model Compression Tool</span></b></p>
  <!-- Use a divider instead of the background -->

 [![Quick Start](https://badgen.net/badge/Quick%20Start/QuickStart/blue)](./docs/en/quick_start/quantization_quick_start.md)
 [![AI FAQ (DeepWiki)](https://badgen.net/badge/AI%20FAQ/DeepWiki/blue)](https://deepwiki.com/Keithwwa/msmodelslim)
 [![AI FAQ (ZRead)](https://badgen.net/badge/AI%20FAQ/ZRead/blue)](https://zread.ai/mindstudio-docs/26.2.0)
 [![Ascend Community](https://badgen.net/badge/Ascend%20Community/Community/blue)](https://www.hiascend.com/eng/developer/software/mindstudio)
 [![Report Issues](https://badgen.net/badge/Report%20Issues/Issues/blue)](https://gitcode.com/Ascend/msmodelslim/issues/new)

</div>

 English | [简体中文](./README.md)

## ✨ Latest News

<span style="font-size:14px;">

🔹 **[2026.09.03]**

- New support for quantization of the `Step-3.7-Flash` (W8A8 MXFP8) multimodal model

🔹 **[2026.08.07]**

- Multimodal understanding models now support multi-card distributed layer-by-layer quantization, improving quantization efficiency for multimodal understanding models
- New support for quantization of the `Kimi-K3` (W4A8) model

🔹 **[2026.09.02]**

- Improved the MXFP8/MXFP4 mixed quantization best practices for Gemma4 Dense `gemma-4-31B-it`
- Improved the MXFP8/MXFP4 mixed quantization best practices for Gemma4 MoE `gemma-4-26B-A4B-it`

🔹 **[2026.07.15]**

- New support for quantization of the Gemma4 Dense `gemma-4-31B-it` (W8A8) and Gemma4 MoE `gemma-4-26B-A4B-it` (W8A8) models

</span>

<details>
<summary>🗂️ More historical updates (click to expand)</summary>

🔹 **[2026.07.07]**

- New support for quantization of the Tencent Hunyuan `Hy3` (W8A8) model

🔹 **[2026.06.01]**

- New Support for quantization of the `InternVL3_5-38B` (W8A8) and `InternVL3_5-241B-A28B` (W8A8) models
- Added quantization support for the `Kimi-K2.6` (W4A8) model

🔹 **[2026.04.01]**

- Added quantization support for the `DeepSeek-V4-Flash` (W8A8) model
- Added quantization support for the `Kimi-K2.5` (W4A8) model

📄 For more historical updates, see [Release Notes](https://gitcode.com/Ascend/msmodelslim/releases).

</details>

---

## ℹ️ Introduction

**MindStudio ModelSlim (msModelSlim)** is a high-performance model compression tool in the Ascend ecosystem. It supports quantization and compression of dense LLMs, MoE models, and multimodal models. Developers can quickly tune models through the command line and configuration and export models adapted to frameworks such as MindIE and vLLM-Ascend for efficient deployment on Ascend AI Processors.

**What it can bring to you:**

- 🚀 **Inference Acceleration**: Quantization significantly reduces memory usage, improving inference throughput and deployment cost efficiency (for example, the original weights of the Qwen3.6-27B model exceed 50 GB, and after W8A8 quantization they are reduced to 30+ GB, saving 40% of memory).
- 🎯 **Out of the box**: Integrates best practices for mainstream large model quantization. A single `msmodelslim quant` command completes quantization.
- 🔧 **Controllable precision**: Provides sensitive layer analysis, automatic tuning, and a closed-loop precision feedback mechanism, making quantization precision measurable and tunable.
- 🧩 **Ecosystem-friendly**: The exported quantized weights seamlessly integrate with mainstream inference frameworks such as vLLM-Ascend.

---

## 🚀 Quick Start

**A step-by-step guide to end-to-end model quantization**. See *[Quick Start](./docs/en/quick_start/quantization_quick_start.md)*.

---

## 📦 Installation Guide

msModelSlim has been published to PyPI and can be installed directly via pip.

```bash
# Install msModelSlim (the latest version has been updated to the PyPI source)
<!-- md-trans-meta sourceCommit=26957a5bae8af69f15670791c8a66e6cb2d0f974 translatedAt=2026-10-14T08:54:10.562Z pushedAt=2026-10-14T09:02:58.550Z -->

pip install msmodelslim

# Verify the installation: printing the command-line help information indicates successful installation
msmodelslim --help
```

> For more installation methods, see *[Installation Guide](./docs/en/install_guide/install_guide.md)*.

---

## 📘 User Guide

**Want to know how to use msModelSlim?** Choose an entry point as needed:

| Business Process | Corresponding Tool/Feature |
|---------|--------------|
| *[New Model Quantization Tuning Process](./docs/en/user_guide/process_new_model_quantization_tuning.md)* | [Weight Quantization](./docs/en/user_guide/usage_weight_quantization.md), [Sensitive Layer Analysis](./docs/en/user_guide/usage_sensitive_linear_analysis.md) |
| *[Precision Tuning Methods](./docs/en/user_guide/process_quantization_precision_tuning.md)* | [Sensitive Layer Analysis](./docs/en/user_guide/usage_sensitive_linear_analysis.md), [Auto Tuning](./docs/en/user_guide/usage_auto_precision_tuning.md), [Debug Mode](./docs/en/user_guide/usage_debug_mode.md) |
| *[Mainstream Model Quantization Deployment](./docs/en/user_guide/process_mainstream_model_deployment.md)* | *[One-Click Quantization](./docs/en/user_guide/usage_one_click_quantization.md)* |
| *[Locating Quantization Inference Accuracy Anomalies](./docs/en/user_guide/process_quantization_accuracy_anomaly_locating.md)* | [Debug Mode](./docs/en/user_guide/usage_debug_mode.md) |

> For more usage guides, see *[User Guide](./docs/en/user_guide/README.md)*.

---

## 📚 Knowledge Base

**Unfamiliar with quantization terms such as `W4A8`/`W8A8`?** See *[Inference Acceleration Knowledge Base](./docs/en/knowledge_base/README.md)*.

---

## 💡 Typical Cases

**Looking for typical scenarios for model quantization tuning?** See *[msModelSlim Typical Cases](./docs/en/best_practices/README.md)*.

---

## ❓ FAQs

**Encountering a problem?** See *[FAQs](./docs/en/support/faq.md)*.

---

## 🌌 Intelligent Search

To improve document retrieval efficiency, we provide multiple efficient search methods:<br>
🔹 [AI Agent (msagent)](https://gitcode.com/Ascend/msagent/blob/26.2.0/docs/en/agent_guide/quantizer.md): The msModelSlim quantization capability has been integrated into the msagent agent, enabling automatic model quantization tasks through natural language interaction.<br>
🔹 [AI Q&A (DeepWiki)](https://deepwiki.com/Keithwwa/msmodelslim): Natural language Q&A for quickly grasping the project architecture and module relationships.<br>

---

## 🛠️ Contribution Guide

**Want to contribute?** For details, see *[Contribution Guide](./docs/en/contributing/contributing_guide.md)*.

---

## ⚖️ Related Notes

🔹 *[Release Notes](https://gitcode.com/Ascend/msmodelslim/releases)*<br>
🔹 *[License Notice](docs/en/legal/license_notice.md)*<br>
🔹 *[Security Statement](docs/en/legal/SECURITY.md)*<br>
🔹 *[Disclaimer](docs/en/legal/disclaimer.md)*<br>

---

## 🤝 Suggestions and Communication

Everyone is welcome to contribute to the community. If you have any questions or suggestions, please submit them to [Issues](https://gitcode.com/Ascend/msmodelslim/issues), and we will respond as soon as possible. Thank you for your support.

|                                                                         Real-time Interaction (WeChat Group)                                                                          |                                                                               Official Updates (Official Account)                                                                                | In-depth Support (Assistant/Forum)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| <img src="https://raw.gitcode.com/Ascend/docs/files/master/common/Writing_Template/figures/qr_code_wechat_work.png" width="120"><br><sub>*Scan the QR code to join the technical exchange group*</sub> | <img src="https://raw.gitcode.com/Ascend/docs/files/master/common/Writing_Template/figures/qr_code_wechat_official_account.png" width="120"><br><sub>*Scan the QR code to follow the official account*</sub> | Scan the QR code to join the group and follow the official account for the fastest communication platform for MindStudio users and developers:<br> **Quick Questions:** Discuss technical issues with community members in real time<br>**Stay Updated:** Get version release and feature update notifications as soon as possible<br> **Experience Sharing:** Exchange best practices and hands-on insights with developers  <br> <br> **More Support Channels**:👉 Ascend Assistant: [![WeChat](https://img.shields.io/badge/WeChat-07C160?style=flat-square&logo=wechat&logoColor=white)](https://gitcode.com/Ascend/msit/blob/master/docs/zh/figures/readme/xiaozhushou.png) 👉 Ascend Forum: [![Website](https://img.shields.io/badge/Website-%231e37ff?style=flat-square&logo=RSS&logoColor=white)](https://www.hiascend.com/forum/) |

---

## 🙏 Acknowledgments

This tool is jointly contributed by the following departments of Huawei:<br>
🔹 Ascend Computing MindStudio Development Department<br>
🔹 Ascend Computing Ecosystem Enablement Department<br>
🔹 Ascend Computing Technology Development Department<br>
🔹 2012 Laboratories<br>

Thanks for every PR from the community. Contributions are welcome!
