# 权重量化使用指南

## 1. 适用范围

本指南面向需要将**尚未收录于支持矩阵、或虽已收录但目标量化模式未验证**的模型接入 msModelSlim，并产出可部署量化权重的开发者与算法工程师。覆盖以下模型类别：

- **大语言模型（LLM）**：稠密 / MoE 文本模型等
- **多模态理解模型**：视觉-语言理解（VLM）等
- **多模态生成模型**：文生视频 / 图生视频等

本指南描述三类模型共用的业务路径：下载浮点模型 → 完成模型适配 → 编写配置 → 执行量化 → 校验交付件。

**适用场景**：

- 目标模型**未出现在支持矩阵**，或虽已收录但**目标量化模式（`quant_type`）未验证**，需要完成适配与配置后，通过 `msmodelslim quant` 产出可部署量化权重。

> [!NOTE] 不适用场景
>
> - 支持矩阵中已标记「一键量化」且目标 `quant_type` 已验证的模型：请直接按《[一键量化完整指南](usage_quick_quantization.md)》执行，无需走本指南的适配开发步骤。

## 2. 流程关系与前置条件

**上级流程**：《[新模型量化调优流程](process_new_model_quantization_tuning.md)》。

**前置条件**：

- 已核对《[大模型支持矩阵](../knowledge_base/model/README.md)》，确认目标模型未收录，或目标量化模式（`quant_type`）尚未验证。
- 已确认目标模型所属类别（LLM / 多模态理解 / 多模态生成）。
- 已按模型发布页与对应接入指南准备运行依赖：不同模型常要求特定 `transformers` 等版本；多模态生成还须具备原推理仓及桥接所需依赖。具体版本以模型卡片 / 接入指南为准，本文不枚举。
- 已完成 msModelSlim **源码安装**（新模型适配需改代码与注册 entry point，详见《[安装指南](../install_guide/install_guide.md)》）。量化设备方面：常规路径需可用的昇腾 NPU；`msmodelslim quant` 也支持 `--device cpu`，可用于小规模调试，大参数或复杂流水线仍以 NPU 为准。
- 已确定目标推理框架，并选定 msModelSlim 支持的导出格式（见《[量化格式支持矩阵](../knowledge_base/quantization_format/README.md)》，如 AscendV1、MindIE-SD）。

**后续操作**：

- 交付件校验通过后，回到《[新模型量化调优流程](process_new_model_quantization_tuning.md)》继续算子开发与组图、精度/性能测评等步骤。

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型目录 | 用户本地路径；一般自 ModelScope / Hugging Face 获取 | 含模型配置、权重分片及类别所需附属文件（如 tokenizer、config 等） | 文件齐全；若官方提供校验值或版本号，与本地一致 |
| 输入 | 量化配置 YAML | 本指南步骤中编写的配置文件路径 | 符合对应配置协议（`modelslim_v1` / `multimodal_vlm_modelslim_v1` / `multimodal_sd_modelslim_v1`） | 新模型接入阶段以显式指定配置文件路径可加载为准 |
| 交付件 | 量化权重目录 | 用户指定的量化输出目录 | 含所选导出格式约定的描述文件与权重分片（如 AscendV1 的 `quant_model_description.json`） | 文件齐全；符合所选导出格式约定 |

## 4. 流程总览

本流程负责权重量化并交付权重；推理 / 生成由目标引擎承担（如 MindIE、vLLM Ascend、MindIE-SD）。先完成环境与资源预检，再下载浮点模型；**尚未接入**时完成模型适配，**已接入**可跳过适配；随后编写量化配置、执行量化，最后校验量化权重交付件是否完整可用。各阶段顺序如下：

```mermaid
flowchart LR
  precheck[执行前预检] --> download[下载浮点模型]
  download --> adapt[完成模型适配]
  download -->|已接入可跳过| scheme[编写量化配置]
  adapt --> scheme
  scheme --> run[执行量化命令]
  run --> check[校验交付件]
```

## 5. 操作步骤

### 命令行预览

```bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \          # 浮点权重目录
  --save_path ${SAVE_PATH} \            # 量化权重输出目录
  --device npu \                        # 量化设备，如 npu、npu:0,1,2,3
  --model_type ${MODEL_TYPE} \          # 已注册或支持矩阵中的模型名，大小写敏感
  --config_path ${CONFIG_PATH} \        # 本指南步骤 3 编写的量化配置 YAML
  --trust_remote_code False             # 仅可信模型必要时设为 True
```

### 执行前预检

进入下载与量化前，按下列 checklist 逐项核对，任一项未通过时，先处理再继续后续步骤，执行一键量化前建议再跑一遍本预检。

| 序号 | 检查项 | 推荐命令 | 通过标准 |
| --- | --- | --- | --- |
| 1 | 原始权重未被意外更改 | `ls -lt <浮点模型目录> \| head` 或 `stat <关键权重文件>`，记录并比对最后修改时间 | 下载后与量化前的修改时间一致，无异常变化 |
| 2 | 硬盘空间足够 | `df -h` | 浮点目录与量化输出目录所在分区空间可覆盖模型体积、量化产物及余量 |
| 3 | NPU 未被占用 | `npu-smi info` | 目标卡状态正常，无非预期任务长期占卡 |

### 步骤 1：下载浮点模型

**目标**：获得完整、可追溯的浮点模型本地目录。

**执行前检查**：已完成[执行前预检](#执行前预检)中的硬盘空间项。

**操作**：

1. 从 [ModelScope](https://www.modelscope.cn/)、[Hugging Face](https://huggingface.co/) 或团队内部模型存放位置获取完整权重到本地目录；具体下载方式以对应社区或仓库文档为准。
2. 核对目录含配置、权重分片及类别所需附属文件。若官方页面提供文件校验值（如 MD5/SHA256）或明确的版本号/提交号，与本地下载结果比对一致即可。下载完成后记录最后修改时间，供量化前预检比对。

**输出**：浮点模型目录 `${MODEL_PATH}`。

**通过条件**：文件齐全；有官方校验值或版本号时与本地一致。

### 步骤 2：完成模型适配

**目标**：确保存在可被 `msmodelslim quant --model_type <模型名>` 命中的模型适配器。

**操作**：

- **尚未接入的模型**：须先完成适配器开发与注册，再进入步骤 3。
- **支持矩阵中已接入的模型**：可跳过本步骤，直接进入步骤 3 编写配置；确认所用 `--model_type` 名称即可。

尚未接入时按下列操作执行：

1. 按模型类别阅读对应接入指南，并按其要求完成适配器实现与注册（细节以接入指南为准）：

   | 模型类别 | 接入指南 |
   | --- | --- |
   | 大语言模型 | 《[LLM 大模型接入指南](../knowledge_base/model/integrating_models.md)》 |
   | 多模态理解 | 《[多模态理解模型接入指南](../knowledge_base/model/integrating_multimodal_understanding_model.md)》 |
   | 多模态生成 | 《[多模态生成模型接入指南](../knowledge_base/model/integrating_multimodal_generation_model.md)》 |

2. 完成适配器开发与注册后，在仓库根目录重新执行 `bash install.sh`，使注册生效。

**输出**：已注册且可被 CLI 命中的模型适配器（对应 `--model_type` 名称）；已接入并跳过本步骤时，输出即为既有 `--model_type`。

**通过条件**：使用该 `--model_type` 启动量化时，日志显示已命中预期模型适配器，且不再报未知模型/未命中适配器。

### 步骤 3：编写量化配置

**目标**：按上级流程「量化方案设计」已确定的方案，写出量化配置 YAML。首次推荐方案见《[新模型量化调优流程](process_new_model_quantization_tuning.md)》步骤 1。

**操作**：

1. 确认本轮量化模式、算法与量化范围（来自上级流程的方案结论）。
2. **编写量化配置**并保存为本地文件。下列为 **LLM / 多模态理解** 可用的首版骨架（`w8a8` 动态 + 仅 `linear_quant`）；按模型类别修改 `apiversion`、`dataset`、`include`/`exclude`。**多模态生成勿直接套用本骨架**：`save` 常使用 `mindie_format_saver`，并补充 `multimodal_sd_config`（如 `dump_config`、`inference_config`），字段见下方协议链接。

   ```yaml
   apiversion: modelslim_v1
   # 多模态理解改为 multimodal_vlm_modelslim_v1
   # 多模态生成改为 multimodal_sd_modelslim_v1
   spec:
     process:
       - type: "linear_quant"
         qconfig:
           act:
             scope: "per_token"
             dtype: "int8"
             symmetric: true
             method: "minmax"
           weight:
             scope: "per_channel"
             dtype: "int8"
             symmetric: true
             method: "minmax"
         # 多模态理解初版建议改为仅匹配语言部分
         include: ["*"]
         exclude: []
     dataset: "mix_calib.jsonl"
     # 多模态理解常用 "calibImages"；多模态生成按场景配置短名称或路径
     save:
       - type: "ascendv1_saver"
         part_file_size: 4
     # 多模态生成另需按协议补充 multimodal_sd_config 等字段
   ```

   各类别完整配置字段以对应协议为准：

   | 模型类别 | 配置协议 | 权威说明 |
   | --- | --- | --- |
   | 大语言模型 | `modelslim_v1` | [一键量化完整指南 - modelslim_v1 配置详解](usage_quick_quantization.md#52-modelslim_v1-配置详解) |
   | 多模态理解 | `multimodal_vlm_modelslim_v1` | [一键量化完整指南 - multimodal_vlm_modelslim_v1 配置详解](usage_quick_quantization.md#54-multimodal_vlm_modelslim_v1-配置详解) |
   | 多模态生成 | `multimodal_sd_modelslim_v1` | [一键量化完整指南 - multimodal_sd_modelslim_v1 配置详解](usage_quick_quantization.md#53-multimodal_sd_modelslim_v1-配置详解) |

**输出**：量化配置 YAML 文件。

**通过条件**：配置字段符合所选模型类别的完整配置协议要求。

### 步骤 4：执行一键量化命令

**目标**：通过 `msmodelslim quant` 生成量化权重目录。

**执行前检查**：

- 再次完成[执行前预检](#执行前预检)。
- 确认当前环境依赖仍满足量化加载需要：`transformers` 等版本与模型发布页一致；多模态生成还需原推理仓及相关依赖可正常 import / 调用。
- 在源码目录外执行命令；输出目录可写，且不与浮点模型目录混用。

**操作**：

新模型接入阶段使用 `--config_path` 显式指定步骤 3 的配置，使用自定义配置时，量化结果由配置与适配自行保证，msModelSlim 不对未经验证的自定义配置效果负责。

```bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --device npu \
  --model_type ${MODEL_TYPE} \
  --config_path ${CONFIG_PATH} \
  --trust_remote_code False
```

参数说明：

- `--model_path`：浮点权重目录。
- `--save_path`：量化产物输出目录。
- `--device`：量化设备，如 `npu`、`npu:0,1,2,3`。
- `--model_type`：步骤 2 注册的模型名，或支持矩阵中已有名称；大小写敏感。
- `--config_path`：步骤 3 编写的 YAML。
- `--trust_remote_code`：仅当模型必须执行仓库内自定义代码且来源可信时设为 `True`。

**输出**：量化权重目录。

**通过条件**：命令正常结束，且输出目录已写出量化权重相关文件。

**审计记录**：实际执行的命令行、所用配置文件路径、量化日志路径（若启用调试模式则含 `debug_info` 目录）。

### 步骤 5：校验交付件

**目标**：确认量化产物完整、可定位，并符合所选导出格式约定。

**操作**：

1. 按所选导出格式核对产物结构与关键文件。常见格式说明：

   | 导出格式 | 产物说明 |
   | --- | --- |
   | AscendV1 | 《[AscendV1 格式说明](../knowledge_base/quantization_format/ascendv1/ascendv1.md)》 |
   | MindIE-SD 等 | 《[量化格式支持矩阵](../knowledge_base/quantization_format/README.md)》及对应 saver 说明 |

2. 对比浮点模型目录与量化权重目录的体积（如 `du -sh ${MODEL_PATH} ${SAVE_PATH}`），确认量化产物相对浮点权重有合理缩小；具体幅度随位宽与导出格式而异，不作固定压缩比要求。

**输出**：通过校验的量化权重目录。

**通过条件**：产物符合所选导出格式约定；量化目录体积相对浮点目录合理缩小。

**审计记录**：量化权重目录路径；浮点与量化目录体积对比结果；运行环境信息（如 NPU / 驱动 / CANN）；msModelSlim 与相关依赖版本。

## 6. 全局验收条件

- 浮点模型目录完整可追溯；适配器可被 `--model_type` 命中；量化配置符合所选协议。
- 量化权重目录齐全，符合所选导出格式约定；相对浮点目录体积合理缩小。

## 7. 全局异常处置

本表仅覆盖本指南步骤内（预检 → 校验交付件）的常见阻塞；精度/性能不达标及多次迭代仍无法达标，见《[量化精度调优指南](process_quantization_precision_tuning.md)》及后续部署测评。

| 现象 | 处理方向 |
| --- | --- |
| 预检未通过 | 按[执行前预检](#执行前预检)处理：恢复或重下权重、扩容或换盘、释放或改选空闲 NPU 后再继续 |
| 模型加载 / `trust_remote_code` / 依赖版本失败 | 核对模型发布页与前置条件中的依赖；必要时仅对可信模型开启远程代码 |
| `--model_type` 未命中 | 检查 `config.ini` 注册与 `install.sh` 是否重跑 |
| 配置解析失败或协议字段缺失 | 回到步骤 3，对照对应配置协议补齐/修正 YAML 后重跑步骤 4 |
| 量化中接口、子图、校准 dump 或专家编排错误 | 回到步骤 2，按模型类别对照对应接入指南与同系列适配器补齐实现 |
| 产物不全或描述文件不符合预期 | 核对 saver / 导出格式与《[量化格式支持矩阵](../knowledge_base/quantization_format/README.md)》；确认描述文件后重跑步骤 4，再执行步骤 5 |
| 量化过程 OOM / 显存不足 | 改用多卡量化、缩小校准规模，或换更大显存设备后重跑 |

## 8. 案例列表

| 案例 | 简述 | 链接 |
| --- | --- | --- |
| DeepSeek-V4-Pro W4A8 新接入 | LLM 新模型适配 + W4A8 配置 + 一键量化 + 部署评测闭环 | 《[DeepSeek-V4-Pro W4A8 新模型量化案例](../best_practices/deepseek_v4_pro_w4a8_new_llm_quantization_case.md)》 |

## 9. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| 模型适配 | 为特定模型实现并注册适配器，使 CLI 可通过 `--model_type` 命中加载与量化能力 | 《[LLM 大模型接入指南](../knowledge_base/model/integrating_models.md)》 |
| 量化算法 | 离群值抑制、线性量化、敏感层分析等算法说明 | 《[量化算法总览](../knowledge_base/quantization_algorithms/README.md)》 |
| 量化模式 | 如 w8a8、w4a8 等比特组合策略的命名与约定 | 《[量化模式命名规范](../knowledge_base/model/README.md#量化模式命名规范)》 |
| 量化格式 | 量化权重导出格式及其与推理框架的对应关系 | 《[量化格式支持矩阵](../knowledge_base/quantization_format/README.md)》 |

## 10. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| `msmodelslim quant` | 权重量化统一命令行入口 | 本指南 [命令行预览](#命令行预览) |
| 量化配置协议 | `modelslim_v1` / `multimodal_vlm_modelslim_v1` / `multimodal_sd_modelslim_v1` 等 YAML 约定 | [一键量化完整指南 - 量化配置协议详解](usage_quick_quantization.md#5-量化配置协议详解) |

## 11. 安全说明

- `trust_remote_code` 默认保持 `False`；仅当浮点仓库必须执行自定义代码且来源可信、可审计时开启。
- 浮点权重、校准数据（含图像/视频路径所指向内容）与量化产物应按业务权限管控；勿将含业务数据的校准集或日志提交到公开渠道。
- 调试模式可能落盘中间张量与统计信息，使用后按需清理 `save_path/debug_info`。
