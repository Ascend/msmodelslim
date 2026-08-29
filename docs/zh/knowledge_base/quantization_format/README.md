# 量化格式

> **词条类别**：量化数据格式
>
> **英文名称**：Quantization Format
>
> **应用领域**：大语言模型量化压缩、多模态理解与生成模型量化压缩、推理框架权重加载
>
> **msModelSlim 实现**：`msmodelslim/format/`

## 1. 概述

[量化格式](./README.md)是量化工具与推理框架之间的**落盘与加载协议**：它规定量化权重的文件结构、张量命名与元数据组织方式，使目标推理引擎能正确反量化并完成推理。它解决算法产出的量化参数如何被下游部署消费的问题。

## 2. 词条介绍

### 2.1 格式地图

地图用于**选型与导航**：先按目标推理栈与模型类型选定格式，再进入对应词条核对模式 / 交付件，或进入使用指南完成确认模式支持、配置与执行。特定场景以各格式词条与使用指南为准，本表不展开。

| 格式 | 典型适用场景 | 目标推理框架 | 模式概要与推荐 | 分布式导出 | 词条 | 使用指南 |
| --- | --- | --- | --- | --- | --- | --- |
| AscendV1 | 昇腾侧 LLM / 多模态理解等通用部署 | vLLM Ascend、SGLang、MindIE | 覆盖 W8A8 / W8A16 / W4A4 / MXFP / KV Cache / FA 等 20+；**910 系推荐优先 W8A8（静/动）**，显存更紧时再选 W4；**950 系推荐优先 MXFP（W8A8_MXFP8 / W4A4_MXFP4 等）** | 支持 | 《[AscendV1](ascendv1/term_ascendv1.md)》 | 《[AscendV1 使用指南](ascendv1/ascendv1_usage.md)》 |
| compressed-tensors | 量化权重要在 vLLM 等 HF 生态框架中加载，或需要与 compressed-tensors 规范的 `quantization_config` 对齐 | vLLM 等 HF 生态 | 当前仅 **W8A8 Static / W8A8 Dynamic**；激活 scale 已离线校准选 Static，需免校准或适应动态输入选 Dynamic；不支持 KV Cache | 不支持 | 《[compressed-tensors](compressed_tensors/term_compressed_tensors.md)》 | 《[compressed-tensors 使用指南](compressed_tensors/compressed_tensors_usage.md)》 |
| MindIE-SD | 扩散 / DiT 等多模态**生成**模型交付 MindIE | MindIE（多模态生成） | 面向生成流水线的枚举子集（含 W8A8、MXFP、FAQuant 等）；**推荐按模型最佳实践 `quant_type` 选择**，勿套用 LLM 默认 AscendV1 路径 | 支持 | 《[MindIE-SD](mindie_sd/term_mindie_sd.md)》 | 《[MindIE-SD 使用指南](mindie_sd/mindie_sd_usage.md)》 |

#### 三种格式如何区分

| 维度 | AscendV1 | compressed-tensors | MindIE-SD |
| --- | --- | --- | --- |
| 定位 | 昇腾推理默认量化落盘协议 | HF 生态兼容的量化格式 | 多模态生成专用 MindIE 落盘协议 |
| 模型面 | 大语言模型和多模态理解模型 | 线性层量化、HF 风格权重 | 扩散 / DiT 等生成模型（如 Wan2.2） |
| 元数据形态 | 独立 `quant_model_description.json` + `quant_model_weights*.safetensors` | 注入 `config.json` 的 `quantization_config` + `model*.safetensors` | `quant_model_description*.json` + `quant_model_weight*.safetensors`（权重文件名为单数 weight） |
| 保存器 `type` | `ascendv1_saver` | `compressed_tensors` | `mindie_format_saver` |
| 核心优势 | 描述 JSON 粒度最细：逐层标注量化类型，支持混合量化（同模型不同层可用不同模式）；模式枚举最全，覆盖 20+ 量化组合 | 与 HF `from_pretrained` / vLLM 自动检测路径天然兼容，无需额外安装 `compressed-tensors` 包，零门槛接入 HF 生态 | 与 MindIE 多模态生成流水线深度对齐，支持 `fa_quant_type` 等生成场景专属元数据，`dump_config` 可捕获校准数据 |

三者**不可互换加载**：换推理栈或换模型品类通常需重新导出。模式枚举、字段级交付件与限制见上表词条；配置与执行步骤见对应使用指南。

接入新格式请遵循《[量化格式接入指南](iformat_integration_guide.md)》及《[msModelSlim 资料规范](../../contributing/development_guide/docs_standards/README.md)》。

## 3. 关联流程

| 流程 | 说明 |
| --- | --- |
| 《[量化格式接入指南](iformat_integration_guide.md)》 | 基于 IFormat 接入新落盘格式 |
| 《[AscendV1 使用指南](ascendv1/ascendv1_usage.md)》 | AscendV1 确认模式支持、配置与执行 |
| 《[compressed-tensors 使用指南](compressed_tensors/compressed_tensors_usage.md)》 | compressed-tensors 确认模式支持、配置与执行 |
| 《[MindIE-SD 使用指南](mindie_sd/mindie_sd_usage.md)》 | MindIE-SD 确认模式支持、配置与执行 |
| 《[一键量化使用指南](../../user_guide/usage_quick_quantization.md)》 | `spec.save` 与命令行总览 |

## 4. 关联词条

- [AscendV1](ascendv1/term_ascendv1.md)：下位概念，昇腾侧默认量化落盘格式。
- [compressed-tensors](compressed_tensors/term_compressed_tensors.md)：下位概念，HuggingFace / vLLM 生态兼容的量化落盘格式。
- [MindIE-SD](mindie_sd/term_mindie_sd.md)：下位概念，多模态生成场景的 MindIE 落盘格式。
- [量化算法](../quantization_algorithms/README.md)：其他，量化计算与校准侧算法族，产出由本格式落盘。
- [量化模式](../quantization_mode/README.md)：配套术语，格式枚举与交付件字段对应各量化模式。
- [大模型支持矩阵](../model/README.md)：其他，模型 × 模式 × 推理栈的选型与验证口径。
