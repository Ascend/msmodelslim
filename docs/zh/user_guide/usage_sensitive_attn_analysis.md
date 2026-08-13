# Attention 敏感层分析使用指南

## 1. 适用范围

本指南面向需要通过 `msmodelslim analyze attn`，按 **Attention 模块**粒度识别量化敏感层，并据此调整量化 YAML（配合 `fa3_quant` 等做回退）的开发者与算法工程师。

**适用场景**：

- FA 量化方案设计：配合 Flash Attention 3 等注意力激活量化，识别需回退的 attention 模块。
- 精度不达标时，结合 Attention 敏感度排序迭代 FA 相关配置。

**不适用场景**：

- 多模态理解 / 多模态生成模型：当前敏感层分析仅支持大语言模型。
- 需要按单层线性层回退或提位宽：请使用《[线性层敏感层分析使用指南](usage_sensitive_linear_analysis.md)》。
- 需要按 Decoder 块或整块 Attention / MLP / MoE 回退：请使用《[层级敏感层分析使用指南](usage_sensitive_layer_wise_analysis.md)》。

## 2. 流程关系与前置条件

**上级流程**：《[新模型量化调优流程](process_new_model_quantization_tuning.md)》。

**前置条件**：

- 已安装 msModelSlim（详见《[安装指南](../install_guide/install_guide.md)》）。
- 目标模型为 LLM，且已确定可用的 `--model_type`（与支持矩阵 / 适配器注册名一致，大小写敏感；通常已在上级流程或权重量化流程中完成适配）。
- 已具备可用的昇腾 NPU（或仅做小规模调试时使用 `--device cpu`）。
- 若所选指标要求额外分析接口，目标适配器已具备或计划补齐对应能力（见步骤 3）。

**后续操作**：

- 将分析结果中的模块名写入量化配置（如 `fa3_quant` 相关回退配置），再执行 `msmodelslim quant`。

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型目录 | 用户本地路径；一般自 ModelScope / Hugging Face 获取 | 含模型配置、权重分片及 tokenizer 等 | 路径有效，可被目标 `model_type` 加载 |
| 输入 | 校准集 | 用户本地路径，或工具 [`lab_calib`](../../../lab_calib/) 示例短名称 | `.json` / `.jsonl` | 可被 `msmodelslim analyze` 解析且格式符合要求 |
| 交付件 | 敏感层分析结果 | 标准输出（排序列表 + 可粘贴配置片段） | Score 从高到低；含 topK attention 模块名列表 | 能识别预期模块名范围，并可粘贴进量化配置 |

## 4. 流程总览

确认 `attn` 场景与推荐指标后，准备浮点权重与校准集，按需完成模型适配（含分析接口），再执行分析命令并解读结果，用于改写量化配置。

```mermaid
flowchart LR
  scene[确认指标] --> weight[获取浮点权重]
  weight --> adapt[完成模型适配]
  weight -->|已接入且接口齐全可跳过| run[执行分析命令]
  adapt --> run
  run --> result[解读分析结果]
```

## 5. 操作步骤

### 命令行预览

```bash
msmodelslim analyze attn \
  --model_type ${MODEL_TYPE} \          # 已注册或支持矩阵中的模型名，大小写敏感
  --model_path ${MODEL_PATH} \          # 浮点权重目录
  --metrics ${METRICS} \                # 分析指标
  --calib_dataset ${CALIB_DATASET} \    # 校准集路径或工具内置短名称
  --topk ${TOPK} \                      # TopK 数量，默认 15
  --device npu \                        # 分析设备：npu / cpu
  --trust_remote_code False             # 默认 False；仅可信模型必要时设为 True
```

### 步骤 1：确认推荐指标

**目标**：选定命令行预览中的 `${METRICS}`。

**操作**：

将 `${METRICS}` 设为下表之一：

| 可选指标 | 适用说明 | 算法说明 | 推荐 |
| --- | --- | --- | --- |
| `mse` | 浮点与量化权重下 attention 输出 MSE，用于 FA 量化回退 | 《[Attention MSE](../knowledge_base/quantization_algorithms/attention_mse/term_attention_mse.md)》 | **首选**（当前唯一可选） |

**输出**：已选定的 `${METRICS}`。

**通过条件**：`${METRICS}` 属于上表可选值。

### 步骤 2：获取浮点权重与校准数据

**目标**：准备可加载的浮点模型目录与合法校准集。

**操作**：

1. 从 [ModelScope](https://www.modelscope.cn/)、[Hugging Face](https://huggingface.co/) 或团队内部模型存放位置获取完整权重到本地目录；具体下载方式以对应社区或仓库文档为准。
2. 核对目录含配置、权重分片及 tokenizer 等附属文件。若官方页面提供文件校验值（如 MD5/SHA256）或明确的版本号/提交号，与本地下载结果比对一致即可。
3. 准备校准集：敏感层分析所用校准集通常与后续量化保持一致。须为 `.json` / `.jsonl`：JSON 为字符串列表（每项一条校准文本），JSONL 为每行一个 JSON 对象。可使用自有文件，或工具提供的 [`lab_calib`](../../../lab_calib/) 示例。相对路径解析顺序：优先在命令启动目录查找；未找到再在 `lab_calib` 示例目录按同名匹配；均未找到则报错。

**输出**：浮点模型目录与校准集路径（或工具内置校准集短名称）。

**通过条件**：模型目录可加载；校准集路径可被命令解析且格式符合要求。

### 步骤 3：完成模型适配

**目标**：确保存在可被 `msmodelslim analyze --model_type <模型名>` 命中的模型适配器；若所选指标要求额外分析接口，一并按算法文档补齐。

**操作**：

- **尚未接入的模型**：须先完成适配器开发与注册，再进入步骤 4。通用适配要求见《[LLM 大模型接入指南](../knowledge_base/model/integrating_models.md)》。
- **支持矩阵中已接入的模型**：可跳过通用适配，确认所用 `--model_type` 名称即可。
- **算法侧额外接口**：若所选指标要求额外分析接口，一并按算法文档补齐（入口见《[量化算法总览 - 敏感层分析算法](../knowledge_base/quantization_algorithms/README.md#敏感层分析算法)》）。

完成或修改适配器后，在仓库根目录重新执行 `bash install.sh`，使注册生效。

**输出**：已注册且可被 CLI 命中的模型适配器（对应 `--model_type`）；若所选指标需额外接口，适配器已具备对应能力。

**通过条件**：使用该 `model_type` 启动分析时能正确命中适配器；不再因缺少所选指标要求的分析接口而失败。

### 步骤 4：执行分析命令

**目标**：运行 `msmodelslim analyze attn`，得到 Attention 敏感度排序与 YAML 片段。

**执行前检查**：

- 已完成步骤 1～3 的指标、权重与模型适配确认。
- `trust_remote_code` 默认 `False`；仅当模型必须执行仓库内自定义代码且来源可信时设为 `True`。

**操作**：

按[命令行预览](#命令行预览)将变量替换为实际值后执行。

**输出**：标准输出中的敏感层排序与 YAML 格式片段。

**通过条件**：命令正常结束；输出含 Score 排序及 `=== YAML Format for quantization ===` 片段。

**审计记录**：实际命令行、`model_type` / `${METRICS}`、校准集路径、标准输出结果摘要（或保存的日志路径）。

### 步骤 5：解读分析结果并用于配置

**目标**：读懂标准输出中的排序与 YAML 片段，并落实到量化配置。

**操作**：

1. 阅读敏感层排序列表：Score **越高**表示该模块对量化越敏感，优先考虑回退。`attn` 输出按注意力模块粒度排序，示意如下：

   ```text
   === Layer Analysis Results (mse method) ===
   Patterns analyzed: ['*']
   Total layers analyzed: 36
   Layer Sensitivity Scores (higher score = more sensitive to quantization):
   --------------------------------------------------------------------------------
     1. model.layers.33.self_attn                          | Score:   2.0504e+00
     2. model.layers.24.self_attn                          | Score:   4.3414e-01
     3. model.layers.31.self_attn                          | Score:   4.2244e-01
     ...
   --------------------------------------------------------------------------------
   Top 36 most sensitive layers selected for disable_names

   === YAML Format for quantization ===

   top 36:
     - 'model.layers.33.self_attn'
     - 'model.layers.24.self_attn'
     - 'model.layers.31.self_attn'
     ...

   === End of YAML Format ===
   ```

2. 检查 YAML 片段中的 attention 模块名是否覆盖预期范围。
3. 将 YAML 片段中的模块名复制到量化配置（通常写入 `fa3_quant` 或对应 processor 的回退 / 排除配置）。
4. 用更新后的 YAML 执行量化，再按业务口径测评；未达标则扩大回退或更换策略后重跑本流程。

**输出**：已根据敏感层结果更新的量化配置（及后续量化所用 YAML 路径）。

**通过条件**：YAML 中回退模块名与分析结果一致且可被量化配置加载。

## 6. 全局验收条件

- 已选用 `attn` scope 与推荐分析指标。
- 若所选指标需额外分析接口，适配器已满足要求（或使用已支持列表中的 `model_type`）。
- 分析命令成功产出排序与可粘贴 YAML 片段，模块名经人工核对。

## 7. 全局异常处置

| 现象 | 处理方向 |
| --- | --- |
| 校准集格式错误或无法读取 | 核对 `.json`/`.jsonl` 格式、路径与权限；路径解析与示例集见步骤 2 |
| `model_type` 未命中或告警走默认模型 | 尚未完成模型适配与 `--model_type` 注册；先按步骤 3 补齐并重新 `bash install.sh` |
| 分析报缺少接口 / 不支持 | 回到步骤 3，按算法文档补齐所选指标要求的分析接口，或改用已支持该指标的 `model_type` |
| 显存不足或运行失败 | 缩小校准集，或换更大显存设备 |
| 回退后仍无法部署或精度异常 | 核对回退模块名是否正确；核对引擎对回退层数的限制 |

## 8. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| 模型适配 | 新模型接入与注册 | 《[LLM 大模型接入指南](../knowledge_base/model/integrating_models.md)》 |
| 敏感层分析算法 | 用于 `msmodelslim analyze` 的各类敏感度指标（如 Attention MSE 等） | 《[量化算法总览 - 敏感层分析算法](../knowledge_base/quantization_algorithms/README.md#敏感层分析算法)》 |

## 9. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| `msmodelslim analyze attn` | Attention 敏感层分析命令行入口 | 本指南 [命令行预览](#命令行预览) |

## 10. 安全说明

- `trust_remote_code` 默认保持 `False`；仅当浮点仓库必须执行自定义代码且来源可信、可审计时开启。
- 浮点权重与校准数据应按业务权限管控；勿将含业务数据的校准集或分析日志提交到公开渠道。
