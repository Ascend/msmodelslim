<h1 align="center"> MindStudio ModelSlim</h1>

<div align="center">
  <br />
  <br />
  <img src="docs/assets/modelslim_slogan.png" alt="ModelSlim Slogan" width="340" />
  <p align="center">
    <em>Simple, fast, and lean—msModelSlim is all you need.</em>
  </p>
  <strong>Ascend Model Compression Tool</strong>
  <!--Replace the background with a dividing line.-->

[![License](https://img.shields.io/badge/license-MulanPSL--2.0-blue)](https://gitcode.com/Ascend/msmodelslim/blob/master/LICENSE)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Keithwwa/msmodelslim)
<br><br>
[🌐 Project homepage](https://gitcode.com/Ascend/msmodelslim) |
[📖 Documentation](https://msmodelslim.readthedocs.io/zh-cn/latest/) |
[🔥 What's New](#whats-new) |
[🤔 Issue](https://gitcode.com/Ascend/msmodelslim/issues/new)
<br><br>
</div>

<a id="whats-new"></a>

## 🔥 What's New

### 🗓️ March 2026

- Added support for GLM-4.6V W8A8 quantization.

### 🗓️ February 2026

- Added support for Qwen3-Omni-30B-A3B-Thinking and Qwen3-Omni-30B-A3B-Instruct W8A8 quantization.
- Added support for Qwen2.5-Omni-7B W8A8 quantization.
- Added support for Qwen3.5-397B-A17B W8A8 quantization.
- Added support for GLM-5 W4A8 quantization.
- Optimized configuration recommendations for quick quantization scenarios.

### 🗓️ January 2026

- Added support for Qwen3-VL-32B-Instruct W8A8 quantization.

<details>
<summary>📋 Change History (Click to Expand)</summary>

### 🗓️ December 2025

- Added support for automatic tuning using quantization accuracy feedback, enabling automatic searches for optimal quantization configurations based on accuracy targets.
- Added support for quantization of custom multimodal understanding models, enabling the integration of quantization workflows for these models.
- Added support for multi-device execution during quick quantization, enabling distributed layer-wise quantization to increase foundation model quantization efficiency.
- Added support for DeepSeek-V3.2 W8A8 quantization, requiring only a single device with 64 GB GPU memory and 100 GB system memory.
- Added support for DeepSeek-V3.2-Exp W4A8 quantization, requiring only a single device with 64 GB GPU memory and 100 GB system memory.
- Added support for Qwen3-VL-235B-A22B W8A8 quantization.

### 🗓️ November 2025

- Added support for plugin-based model adaptation and configuration registration alongside dependency pre-checks.

### 🗓️ October 2025

- Added support for Qwen3-235B-A22B W4A8 and Qwen3-30B-A3B W4A8 quantization, with quantized model inference and deployment support on the vLLM Ascend framework.

### 🗓️ September 2025

- Added support for DeepSeek-V3.2-Exp W8A8 quantization, requiring only a single device with 64 GB GPU memory and 100 GB system memory.
- Resolved an issue where abnormal tokens (such as "game copy") frequently occurred during Qwen3-235B-A22B W8A8 quantization operations, as detailed in the Qwen3-MoE quantization best practices.
- Added support for DeepSeek R1 W4A8 per_channel quantization `[Prototype]`.
- Added support for sensitivity analysis of foundation model quantization layers.

### 🗓️ August 2025

- Added support for quick quantization of the Wan2.1 model.
- Added support for layer-wise quantization of foundation models, significantly reducing memory consumption during quantization workflows.
- Added support for SSZ weight quantization algorithm of foundation models, improving quantization accuracy by iteratively searching for optimal scaling factors and offsets.

</details>

> Note: Features labeled with `[Prototype]` are not fully verified, meaning they can be unstable or contain bugs. Features labeled with `[Beta]` represent non-commercial capabilities.

## 📖 Overview

The Ascend model compression tool MindStudio ModelSlim (msModelSlim) is a compression tool dedicated to hardware acceleration, leveraging compression technologies natively optimized for Ascend architectures. It integrates a suite of inference optimization technologies (such as quantization and compression) designed to accelerate dense foundation models, Mixture of Experts (MoE) models, multimodal understanding models, and multimodal generative models.

Ascend AI model developers can call Python APIs to adapt algorithms and models, optimize accuracy and performance, and export models in different formats. The models can run on Ascend AI Processors through inference frameworks such as MindIE and vLLM Ascend.

## 🗂️ Directory Structure

The following list describes the key project directories. For a comprehensive breakdown, see [Directory Structure](docs/en/dir_structure.md).

```text
├─config             # Configuration files
├─docs               # Documentation directory
├─example            # Examples directory
├─lab_calib          # Calibration dataset
├─lab_practice       # Best practices
├─msmodelslim
│  ├─app             # Application module
│  ├─cli             # Command-line interface
│  ├─core            # Other quantization modules and components
│  ├─infra           # Quantization infrastructure
│  ├─model           # Model adaptation layer
│  ├─ir              # Quantization mode
│  ├─processor       # Algorithm
│  └─utils           # General utility infrastructure
└─test               # Test directory
```

## 🧾 Release Notes

The release notes of msModelSlim include the software version mapping, software package download, and feature updates of each version. For details, see [Release Notes](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/en/appendix/release_notes.md).

## 🛠️ Environment Setup

For details about the installation procedure, see the [msModelSlim Installation Guide](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/en/getting_started/install_guide.md).

## 🚀 Quick Start

This section helps you quickly get started with the quick quantization of foundation models.

For details, see [Quick Start](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/en/getting_started/quantization_quick_start.md).

## ✨ Feature Description

### 🧩 Model Support Matrix

The model support matrix presents the adaptation status of different features and models in various scenarios in a table format.

For details, see [Model Support Matrix](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/en/model_support/foundation_model_support_matrix.md).

### 📘 Feature Guide

The feature guide provides feature introductions and usage instructions based on the features supported by msModelSlim across different architectures.

For details, see [Tool Documentation](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/en/feature_guide/quick_quantization_v1/usage.md). In the navigation tree on the left, select the feature you want to view.

### ⚙️ Custom Model Quantization

This section provides guidance for developers who need to connect their own models to msModelSlim and perform quick quantization.

For details about model connection, see [LLM Model Integration Guide](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/en/developer_guide/integrating_models.md) and [Multimodal Understanding Model Integration Guide](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/en/developer_guide/integrating_multimodal_understanding_model.md).

### 🧪 Cases Studies

The case collection provides text descriptions and code samples based on actual application scenarios, aiming to help users quickly get familiar with the usage of msModelSlim in specific scenarios, including accuracy tuning methods. msModelSlim will continuously improve the case collection.

<table>
  <thead>
    <tr>
      <th>Case Category</th>
      <th>Case Name</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>v1 framework quantization accuracy tuning</td>
      <td><a href="https://gitcode.com/Ascend/msmodelslim/blob/master/docs/en/case_studies/quantization_precision_tuning_guide.md">v1 Framework Quantization Accuracy Tuning Guide</a></td>
    </tr>
    <tr>
      <td>v1 framework Qwen3-32B accuracy tuning</td>
      <td><a href="https://gitcode.com/Ascend/msmodelslim/blob/master/docs/en/case_studies/qwen3_32b_w8a8_precision_tuning_case.md">v1 Framework Qwen3-32B w8a8 Accuracy Tuning Case</a></td>
    </tr>
    <tr>
      <td><strong>Weight Conversion</strong></td>
      <td>Guide for Using msModelSlim Quantized Weights with AutoAWQ and AutoGPTQ</td>
      <td><a href="https://gitcode.com/Ascend/msmodelslim/blob/master/docs/en/case_studies/msmodelslim_quantized_weight_to_autoawq_autogptq.md">Quantized weight format conversion guide</a></td>
    </tr>
    <tr>
      <td><strong>Inference and Deployment</strong></td>
      <td>Quantized Weight Usage Cases in Acceleration Library and MindIE-Torch Scenarios</td>
      <td><a href="https://gitcode.com/Ascend/msmodelslim/blob/master/docs/en/case_studies/quantization_weight_use_cases_in_acceleration_and_mindie_torch.md">Usage methods of quantized weights in inference acceleration libraries</a></td>
    </tr>
  </tbody>
</table>

## ❓ FAQ

For details about the frequently asked questions, see [FAQ](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/zh/appendix/faq.md).

## 🤝 Contribution Guide

For details, see [Contribution Guide](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/zh/contributing/contributing_guide.md).

## 📌 Related Statements

<details>
<summary>🛡️ Security Statement</summary>

Describes the security hardening information, public network address information, and communication matrix of msModelSlim. For details, see [msModelSlim Security Statement](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/zh/appendix/security_statement.md).

</details>

<details>
<summary>⚠️ Disclaimer</summary>

### 👤 To msModelSlim Users

- This tool is intended solely for debugging and development. You are responsible for any risks and should carefully review the following information:
  - msModelSlim depends on third-party open-source software such as Transformers and PyTorch, which is provided and maintained by their respective communities. Resolution of issues in these dependencies relies on community contributions and feedback. Please notice that the msModelSlim repository does not guarantee fixes for issues in third-party software, nor does it guarantee testing or correction of all vulnerabilities or errors in such software.
  - When you use msModelSlim, it reads model weights from local storage based on provided command-line parameters or configuration files. Using untrusted model weights may cause unknown security risks. You are advised to use methods such as SHA256 verification to ensure model weights are trusted before passing them to the tool.
  - To ensure security and implement the principle of least privilege, you are advised to use msModelSlim as a standard user rather than a high-privilege user (such as root).
    - Adhere to the principle of least privilege. For example, prevent other users from writing data by disabling permissions such as `666` and `777`.
    - Ensure the `umask` value of the execution user is greater than or equal to `0027` to prevent excessive permissions on generated quantized model directories and files.
      - To check the `umask` value, run the `umask` command.
      - To change the `umask` value, run the `umask new_value` command.
    - Ensure that original model data and quantized model data are stored in the current user directory without symbolic links to avoid potential security issues.
  - Data processing and deletion: Users are responsible for managing and deleting any data generated while using this tool. You are advised to promptly delete any related data after use to prevent information leaks.
  - Data confidentiality and transmission: Users understand and agree not to share or transmit any data generated by this tool. Neither the tool nor its developers are responsible for any information leaks, data breaches, or other negative consequences.
  - User input security: Users are responsible for the security of any commands they enter and for any risks or losses resulting from improper input. The tool and its developers are not liable for issues caused by incorrect command usage.
- Disclaimer scope: This disclaimer applies to all individuals and entities using this tool. By using the tool, you acknowledge and accept this statement and assume all risks and responsibilities arising from its use. If you do not agree, please stop using the tool immediately.
- Before using this tool, please read and understand the preceding disclaimer. If you have any questions, contact the developer.

### 📦 To Data Owners

If you do not want your dataset to be mentioned in the models of msModelSlim, or if you wish to update its description, please submit an issue on Gitcode. msModelSlim will delete or update your dataset description according to your request. Thank you for your understanding and contribution to msModelSlim.

</details>

<details>
<summary>📜Contribution Statement</summary>

1. Error report submission: If you discover a vulnerability in msModelSlim that is not a security issue, first search the **Issues** in the msModelSlim repository to avoid submitting duplicates. If the vulnerability is not listed, create a issue. If you discover a security-related issue, do not disclose it publicly. Please refer to the security handling guidelines for details. All error reports must include complete information about the issue.
2. Security issue handling: For guidance on handling security issues in this project, please contact the core team via email for instructions.
3. Resolving existing issues: Browse open Issues to identify issues that need attention, and attempt to fix them.
4. Proposing new features: Use the **Feature** label when creating an issue for a new feature. We will review and confirm proposals periodically.
5. How to contribute:
  a. Fork the repository of the project.
  b. Clone it to your local machine.
  c. Create a development branch.
  d. Conduct local testing. All unit tests, including any new test cases, must pass before submission.
  e. Submit your code.
  f. Create a pull request (PR).
  g. Code review: Modify the code according to review comments and resubmit your changes. This process may involve multiple rounds of iterations.
  h. After your PR is approved by the required number of reviewers, the committer will conduct the final review.
  i. After your PR is approved and all tests pass, the CI system will merge it into the project's main branch.

</details>

<details>
<summary>📄 LICENSE</summary>

For the license of msModelSlim, see [LICENSE](LICENSE).

Documents in the `docs` directory of msModelSlim are licensed under CC-BY 4.0. For details, see [LICENSE](docs/en/LICENSE).

</details>

## 💬 Suggestions and Feedback

You are welcome to contribute to the community. If you have any questions or suggestions, please submit [Issues](https://gitcode.com/Ascend/msmodelslim/issues). We will reply as soon as possible. Thank you for your support.

## 🙏 Acknowledgements

msModelSlim is jointly developed by the following Huawei departments and Ascend ecosystem partners:

Huawei:

- Computing Product Line
- 2012 Labs

Thank you to everyone in the community for your PRs. We warmly welcome your contributions.

## 👥 About the MindStudio Team

The Huawei MindStudio full-pipeline development toolchain team is dedicated to providing an end-to-end solution for building Ascend AI applications, accelerating the processes of training, inference, and operator development. You can learn more about the Huawei MindStudio team through the following channels:
<div style="display: flex; align-items: center; gap: 10px;">
    <span>MindStudio WeChat official account:</span>
    <img width="100" src="./docs/assets/officialAccount.jpg"/>
    <span style="margin-left: 20px;">Ascend open-source assistant:</span>
    <a href="https://gitcode.com/Ascend/msmodelslim/blob/master/docs/assets/assistant.png">
        <img src="https://camo.githubusercontent.com/22bbaa8aaa1bd0d664b5374d133c565213636ae50831af284ef901724e420f8f/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5765436861742d3037433136303f7374796c653d666f722d7468652d6261646765266c6f676f3d776563686174266c6f676f436f6c6f723d7768697465" data-canonical-src="./docs/assets/assistant.png" style="max-width: 100%;">
    </a>
    <span style="margin-left: 20px;">Ascend forum: </span>
    <a href="https://www.hiascend.com/forum/" rel="nofollow">
        <img src="https://camo.githubusercontent.com/dd0b7ef70793ab93ce46688c049386e0755a18faab780e519df5d7f61153655e/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f576562736974652d2532333165333766663f7374796c653d666f722d7468652d6261646765266c6f676f3d6279746564616e6365266c6f676f436f6c6f723d7768697465" data-canonical-src="https://img.shields.io/badge/Website-%231e37ff?style=for-the-badge&amp;logo=bytedance&amp;logoColor=white" style="max-width: 100%;">
    </a>
</div>
Send "communication group" to the official account to obtain the QR code of the technical communication group.
