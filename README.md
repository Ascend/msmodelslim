<h1 align="center"> MindStudio ModelSlim</h1>
<div align="center">
  <br />
  <img src="docs/assets/modelslim_slogan.png" alt="MindStudio ModelSlim Slogan" width="300" />
  <p align="center">
    <em>Simple, fast, and lean—msModelSlim is all you need.</em>
  </p>
  <p><b><span style="font-size:24px;">昇腾模型压缩工具</span></b></p>
  <!-- 用分隔线替代背景 -->

 [![快速入门](https://badgen.net/badge/快速入门/QuickStart/blue)](./docs/zh/quick_start/quantization_quick_start.md)
 [![AI问答(DeepWiki)](https://badgen.net/badge/AI问答/DeepWiki/blue)](https://deepwiki.com/Keithwwa/msmodelslim)
 [![AI问答(ZRead)](https://badgen.net/badge/AI问答/ZRead/blue)](https://zread.ai/mindstudio-docs/master)
 [![昇腾社区](https://badgen.net/badge/昇腾社区/Community/blue)](https://www.hiascend.com/cn/developer/software/mindstudio)
 [![报告问题](https://badgen.net/badge/报告问题/Issues/blue)](https://gitcode.com/Ascend/msmodelslim/issues/new)

</div>

## ✨ 最新消息

<span style="font-size:14px;">

🔹 **[2026.09.03]**

- 新增对 `Step-3.7-Flash`（W8A8 MXFP8）多模态模型的量化支持

🔹 **[2026.08.07]**

- 多模态理解模型支持多卡分布式逐层量化，提升多模态理解模型量化效率
- 新增对 `Kimi-K3`（W4A8）模型的量化支持

🔹 **[2026.09.02]**

- 完善 Gemma4 Dense `gemma-4-31B-it` 的 MXFP8/MXFP4 混合量化最佳实践
- 完善 Gemma4 MoE `gemma-4-26B-A4B-it` 的 MXFP8/MXFP4 混合量化最佳实践

🔹 **[2026.07.15]**

- 新增对 Gemma4 Dense `gemma-4-31B-it`（W8A8）、Gemma4 MoE `gemma-4-26B-A4B-it`（W8A8）模型的量化支持

</span>

<details>
<summary>🗂️ 更多历史更新（点击展开）</summary>

🔹 **[2026.07.07]**

- 新增对腾讯混元 `Hy3`（W8A8）模型的量化支持

🔹 **[2026.06.01]**

- 新增对 `InternVL3_5-38B`（W8A8）、`InternVL3_5-241B-A28B`（W8A8）模型的量化支持
- 新增对 `Kimi-K2.6`（W4A8）模型的量化支持

🔹 **[2026.04.01]**

- 新增对 `DeepSeek-V4-Flash`（W8A8）模型的量化支持
- 新增对 `Kimi-K2.5`（W4A8）模型的量化支持

📄 更多历史更新记录，请参见《[最新消息](./docs/index.md)》。

</details>

---

## ℹ️ 简介

**MindStudio ModelSlim（msModelSlim）** 是昇腾生态下的高性能模型压缩工具，支持稠密LLM、MoE及多模态模型的量化与压缩。开发者可通过命令行与配置快速调优并导出适配 MindIE、vLLM-Ascend 等框架的模型，在昇腾AI处理器上高效部署。

**它能为你带来什么：**

- 🚀 **推理加速**：量化后显著降低显存占用，提升推理吞吐与部署成本效率（例如 Qwen3.6-27B 模型原始权重50+GB，量化后30+GB，显存节约40%）。
- 🎯 **开箱即用**：集成主流大模型量化最佳实践，`msmodelslim quant` 一条命令完成量化。
- 🔧 **精度可控**：提供敏感层分析、自动调优与精度反馈闭环，量化精度可量化、可调优。
- 🧩 **生态友好**：导出的量化权重无缝接入 vLLM-Ascend 等主流推理框架。

---

## 🚀 快速开始

**手把手教你完成端到端模型量化**，请参见《[快速入门](./docs/zh/quick_start/quantization_quick_start.md)》：

---

## 📦 安装指南

msModelSlim已发布到PyPI，可通过pip直接安装。

```bash
# 安装 msModelSlim（最新版本已更新到PyPI源）
pip install msmodelslim

# 验证安装：打印命令行帮助信息即表示安装成功
msmodelslim --help
```

> 更多安装方式，请参见《[安装指南](./docs/zh/install_guide/install_guide.md)》。

---

## 📘 使用指南

**想了解msModelSlim如何使用？** 可按需选择入口：

| 业务流程 | 对应工具/功能 |
|---------|--------------|
| 《[新模型量化调优流程](./docs/zh/user_guide/process_new_model_quantization_tuning.md)》 | [权重量化](./docs/zh/user_guide/usage_weight_quantization.md)、[敏感层分析](./docs/zh/user_guide/usage_sensitive_linear_analysis.md) |
| 《[精度调优方法](./docs/zh/user_guide/process_quantization_precision_tuning.md)》 | [敏感层分析](./docs/zh/user_guide/usage_sensitive_linear_analysis.md)、[自动调优](./docs/zh/user_guide/usage_auto_precision_tuning.md)、[调试模式](./docs/zh/user_guide/usage_debug_mode.md) |
| 《[主流模型量化部署](./docs/zh/user_guide/process_mainstream_model_deployment.md)》 | [一键量化](./docs/zh/user_guide/usage_quick_quantization.md) |
| 《[量化推理精度异常定位](./docs/zh/user_guide/process_quantization_accuracy_anomaly_locating.md)》 | [调试模式](./docs/zh/user_guide/usage_debug_mode.md) |

> 更多使用指南，请参见《[使用指南](./docs/zh/user_guide/README.md)》。

---

## 📚 知识库

**不懂 `W4A8`/`W8A8` 等量化术语？** 请参见《[推理加速知识库](./docs/zh/knowledge_base/README.md)》。

---

## 💡 典型案例

**需要模型量化调优的典型场景？** 请参见《[msModelSlim 典型案例](./docs/zh/best_practices/README.md)》。

---

## ❓ FAQ

**遇到了问题？** 请参见《[FAQ](./docs/zh/support/faq.md)》。

---

## 🌌 智能检索

为提升文档查阅效率，我们提供多种高效检索方式：<br>
🔹 [AI 智能体（msagent）](https://gitcode.com/Ascend/msagent/blob/master/docs/zh/agent_guide/Quantizer.md)：msModelSlim 量化能力已集成至 msagent 智能体，通过自然语言交互即可自动完成模型量化任务。<br>
🔹 [AI 问答（DeepWiki）](https://deepwiki.com/Keithwwa/msmodelslim)：自然语言问答，快速把握项目架构与模块关系。<br>

---

## 🛠️ 贡献指南

**想要贡献力量？** 具体请参见《[贡献指南](./docs/zh/contributing/contributing_guide.md)》。

---

## ⚖️ 相关说明

🔹 《[版本说明](https://gitcode.com/Ascend/msmodelslim/releases)》<br>
🔹 《[许可证声明](docs/zh/legal/license_notice.md)》<br>
🔹 《[安全声明](docs/zh/legal/SECURITY.md)》<br>
🔹 《[免责声明](docs/zh/legal/disclaimer.md)》<br>

---

## 🤝 建议与交流

欢迎大家为社区做贡献。如果有任何疑问或建议，请提交 [Issues](https://gitcode.com/Ascend/msmodelslim/issues)，我们会尽快回复。感谢您的支持。

|                                                                         即时互动（微信群）                                                                          |                                                                               官方资讯（公众号）                                                                                | 深度支持（助手/论坛）                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| <img src="https://raw.gitcode.com/Ascend/docs/files/master/common/Writing_Template/figures/qr_code_wechat_work.png" width="120"><br><sub>*扫码加入技术交流群*</sub> | <img src="https://raw.gitcode.com/Ascend/docs/files/master/common/Writing_Template/figures/qr_code_wechat_official_account.png" width="120"><br><sub>*扫码关注官方公众号*</sub> | 扫码入群并关注公众号，直达 MindStudio 用户与开发者最快捷的交流平台：<br> **快速提问：** 与社区小伙伴即时探讨技术问题<br>**掌握动态：** 第一时间获取版本发布与功能更新通知<br> **经验共享：** 与广大开发者交流最佳实践与实战心得  <br> <br> **更多支持渠道**：👉 昇腾助手：[![WeChat](https://img.shields.io/badge/WeChat-07C160?style=flat-square&logo=wechat&logoColor=white)](https://gitcode.com/Ascend/msit/blob/master/docs/zh/figures/readme/xiaozhushou.png) 👉 昇腾论坛：[![Website](https://img.shields.io/badge/Website-%231e37ff?style=flat-square&logo=RSS&logoColor=white)](https://www.hiascend.com/forum/) |

---

## 🙏 致谢

本工具由华为公司的下列部门联合贡献：<br>
🔹 昇腾计算MindStudio开发部<br>
🔹 昇腾计算生态使能部<br>
🔹 昇腾计算技术开发部<br>
🔹 2012实验室<br>

感谢来自社区的每一个 PR，欢迎贡献！
