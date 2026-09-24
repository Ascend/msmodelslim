# Attention Head 筛选分析使用指南

## 1. 适用范围

本指南面向需要通过 `msmodelslim analyze attn_head` 命令行，按 **Attention Head** 粒度识别关键注意力头（induction heads 与 echo heads），并据此生成 `head.pt` 文件交付给后续 MindIE 推理框架（用于长序列 KV Cache 压缩）的开发者与算法工程师。

**适用场景**：

- **LLM 模型**：当前 `attn_head` 分析仅支持大语言模型。
- RA Compress 压缩方案设计：识别具有 prefix matching（归纳头）和 copying matching（回声头）能力的 KV Cache 头，形成 MindIE 长序列 KV Cache 压缩需要保留的关键头清单。
- 精度不达标时，调整 `induction_head_ratio` / `echo_head_ratio` 重新筛选 head 清单，或调整 MindIE 的 KV Cache 压缩策略。

> **产物用途**：本指南产出的 `head.pt` 是交付给**MindIE 推理框架**使用的关键头清单：在 MindIE 中把该文件路径配置到长序列 KV Cache 压缩相关参数中，即可完成长序列压缩，压缩后的模型可用于后续推理部署；具体已适配的量化模型列表，请参见《[MindIE 是什么](https://www.hiascend.com/document/detail/zh/mindie/10RC3/whatismindie/mindie_what_0001.html)》的“MindIE 支持模型列表”章节。该产物不写入量化配置（不参与量化 YAML 的 `exclude` / `include` 等字段）。

**不适用场景**：

- **多模态理解模型（VLM）**：不支持。
- **多模态生成模型**（文生图或文生视频等）：不支持。
- 需要按单层线性层回退或提位宽：请参见《[线性层敏感层分析使用指南](usage_sensitive_linear_analysis.md)》。
- 需要按 Decoder 块或整块 Attention、MLP、MoE 回退：请参见《[层级敏感层分析使用指南](usage_sensitive_layer_analysis.md)》。
- 配合 FA 量化识别需回退的 Attention 模块：请参见《[Attention 敏感层分析使用指南](usage_sensitive_attn_analysis.md)》。

## 2. 流程关系与前置条件

**上级流程**：《[新模型量化调优流程](process_new_model_quantization_tuning.md)》。

**前置条件**：

- 已安装 msModelSlim（详见《[安装指南](../install_guide/install_guide.md)》）。
- 目标模型为 LLM，且已确定可用的 `--model_type`（与支持矩阵或适配器注册名一致，大小写敏感；通常已在上级流程或权重量化流程中完成适配）。
- 已具备可用的昇腾 NPU（单卡 `--device npu`，或多卡 `--device npu --device_id 0 1 ...`；仅做小规模调试时使用 `--device cpu`）。
- 目标适配器已实现 `RaCompressAnalysisInterface`（`get_proj_names` 提供 Q、K、QKV 投影层名称片段，`get_tokenizer` 返回模型自身的 tokenizer，二者均为必需方法），或使用默认模式（`q_proj`、`k_proj`、`qkv_proj`）与兜底首 token。

**后续操作**：

- 将生成的 `head.pt` 文件提供给 MindIE 推理框架使用（在 MindIE 中把该文件路径配置到长序列 KV Cache 压缩相关参数中），用户可据此完成长序列压缩，压缩后的模型可用于后续推理部署；具体已适配的量化模型列表，请参见《[MindIE 是什么](https://www.hiascend.com/document/detail/zh/mindie/10RC3/whatismindie/mindie_what_0001.html)》的“MindIE 支持模型列表”章节。该产物不写入量化配置。

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型目录 | 用户本地路径；一般自 ModelScope / Hugging Face 获取 | 含模型配置、权重分片及 tokenizer 等 | 路径有效，可被目标 `model_type` 加载 |
| 输入 | 校准输入（算法内建） | 由算法在运行时自行构造的受控输入，无需用户提供 | 首 token + 2500 个随机 token × 4 段重复，总长 `1 + 2500 × 4` | 无需校验；不受 `--calibration_dataset` 影响 |
| 交付件 | head 筛选结果 | 标准输出（induction 与 echo head 列表）及可选 `head.pt` 文件 | `head.pt` 为 Python dict 序列化的 `.pt` 文件，交付给 MindIE 推理框架用于长序列 KV Cache 压缩 | 能识别预期 layer 与 head 范围，`head.pt` 可被 MindIE 的长序列 KV Cache 压缩能力加载 |

## 4. 流程总览

确认 `attn_head` 场景与 `ra_compress` 指标后，准备浮点权重，按需完成模型适配（含 RA Compress 接口），再执行分析命令并解读结果，生成交付给 MindIE 推理框架使用的 `head.pt` 关键头清单。校准输入（首 token + 随机重复段）由算法在运行时自动构造，无需另行准备校准集。

```mermaid
flowchart LR
  scene[确认指标] --> prepare[获取浮点权重]
  prepare --> adapt[完成模型适配]
  prepare -->|已接入且接口齐全可跳过| run[执行分析命令]
  adapt --> run
  run --> result[解读分析结果]
```

## 5. 操作步骤

### 命令行预览

```bash
msmodelslim analyze attn_head \
  --model_type ${MODEL_TYPE} \          # 已注册或支持矩阵中的模型名，大小写敏感
  --model_path ${MODEL_PATH} \          # 浮点权重目录
  --metrics ${METRICS} \               # 分析指标，固定为 ra_compress
  --device npu \                        # 分析设备：npu、cpu；多卡另加 --device_id 0 1 2 3
  --trust_remote_code false \           # 默认 false；仅可信模型必要时设为 true
  --save_path ${SAVE_PATH}              # 可选；指定则保存 head.pt，不指定仅打印到控制台
```

### 步骤 1：确认推荐指标

**目标**：选定命令行预览中的 `${METRICS}`。

**操作**：

将 `${METRICS}` 设为下表之一：

| 可选指标 | 适用说明 | 算法说明 | 推荐 |
| --- | --- | --- | --- |
| `ra_compress` | 识别具有 prefix matching 和 copying matching 能力的注意力头，用于 KV Cache 压缩 | RA Compress 算法：对 Q@K^T 注意力分数在特定段偏移位置取均值，按比例选 top heads（`induction_head_ratio` 默认 0.14、`echo_head_ratio` 默认 0.01，可在 `unary_analysis` 配置的 `metric_params` 中调整） | 当前唯一可选指标，直接选用 |

**输出**：已选定的 `${METRICS}`。

**通过条件**：`${METRICS}` 属于上表可选值。

### 步骤 2：获取浮点权重

**目标**：准备可加载的浮点模型目录。

**操作**：

1. 从 [ModelScope](https://www.modelscope.cn/)、[Hugging Face](https://huggingface.co/) 或团队内部模型存放位置获取完整权重到本地目录；具体下载方式以对应社区或仓库文档为准。
2. 核对目录含配置、权重分片及 tokenizer 等附属文件。若官方页面提供文件校验值（如 MD5/SHA256）或明确的版本号/提交号，与本地下载结果比对一致即可。

> `ra_compress` 不需要用户提供校准集：校准输入是算法内建的受控输入，由算法在运行时自动构造（首 token + 2500 个随机 token × 4 段重复，总长 `1 + 2500 × 4`）。命令行的 `--calibration_dataset` 对该场景不参与计算；模型目录中的 tokenizer 会被用于解析首 token，若缺失则回落到兜底 token id 并告警。

**输出**：可加载的浮点模型目录。

**通过条件**：模型目录可加载，且含 tokenizer 等附属文件。

### 步骤 3：完成模型适配

**目标**：确保存在可被 `msmodelslim analyze --model_type <模型名>` 命中的模型适配器；适配器已实现 `RaCompressAnalysisInterface` 接口的必需方法或使用默认投影层名称模式。

**操作**：

- **尚未接入的模型**：须先完成适配器开发与注册，再进入步骤 4。通用适配要求见《[LLM 大模型接入指南](../knowledge_base/ptq/llm/integration_guide_large_language_model_quantization.md)》。
- **支持矩阵中已接入的模型**：可跳过通用适配，确认所用 `--model_type` 名称即可。
- **算法侧额外接口**：`RaCompressAnalysisInterface` 以类型判定接入——适配器实现该接口才会被指标侧识别。其中 `get_proj_names()` 返回 Q、K、QKV 三路投影层名称片段（如 `q_proj`、`k_proj`、`qkv_proj`）；空串表示该投影不存在，不会被当作"匹配全部 Linear"。`get_tokenizer()` 返回模型实际使用的 tokenizer（通常是 `AutoTokenizer.from_pretrained(model_path)`）。二者均为必需方法，缺一个适配器无法实例化；未提供 tokenizer（返回 `None` 或抛异常）时，指标侧按内置兜底路径取校准输入首 token 并告警。若适配器未实现该接口，则使用默认模式（`q_proj`、`k_proj`、`qkv_proj`）。接口定义见 [ra_compress/interface.py](../../../msmodelslim/processor/analysis/unary_operator/metrics/ra_compress/interface.py)。

完成或修改适配器后，在仓库根目录重新执行 `bash install.sh`，使注册生效。

**输出**：已注册且可被 CLI 命中的模型适配器（对应 `--model_type`）；适配器已具备 `RaCompressAnalysisInterface` 能力（实现 `get_proj_names` 与 `get_tokenizer`）或使用默认模式可正确匹配投影层。

**通过条件**：使用该 `model_type` 启动分析时能正确命中适配器；能正确识别 Q、K、QKV 投影层并采集输出。

### 步骤 4：执行分析命令

**目标**：运行 `msmodelslim analyze attn_head`，得到 induction heads 与 echo heads 筛选结果，并可选保存 `head.pt`。
**执行前检查**：

- 已完成步骤 1～3 的指标、权重与模型适配确认。
- `trust_remote_code` 默认 `false`；仅当模型必须执行仓库内自定义代码且来源可信时设为 `true`。
- 无需指定 `--calibration_dataset`：`ra_compress` 的校准输入由算法在运行时自行构造（见步骤 2），该参数不参与计算。

**操作**：

按[命令行预览](#命令行预览)将变量替换为实际值后执行。示例：

```bash
msmodelslim analyze attn_head \
  --model_type Qwen2.5-7B-Instruct \
  --model_path /data/models/Qwen/Qwen2.5-7B-Instruct/ \
  --metrics ra_compress \
  --device npu \
  --trust_remote_code true \
  --save_path ./head_result
```

**输出**：标准输出中的 induction 与 echo head 筛选结果。若指定 `--save_path`，则同时保存 `head.pt` 文件。

**通过条件**：命令正常结束；输出含 induction heads 和 echo heads 列表；若指定 `--save_path`，`head.pt` 文件成功生成。

**审计记录**：实际命令行、`${MODEL_TYPE}`、`${METRICS}`、标准输出结果摘要（或保存的日志路径）、`head.pt` 文件路径（如指定）。

### 步骤 5：解读分析结果并交付 MindIE 推理框架

**目标**：读懂标准输出中的 induction 与 echo head 筛选结果，并将 `head.pt` 交付给 MindIE 推理框架用于长序列 KV Cache 压缩：用户可据此完成长序列压缩，压缩后的模型可用于后续推理部署。

**操作**：

1. 阅读分析结果输出。`attn_head` 输出按层列出被选中的 KV head 索引，示意如下：

   ```text
   ================================================================================
   === RA Compress Analysis Results ===
   Method: ra_compress
   --------------------------------------------------------------------------------
   === Induction Heads (prefix matching) ===
   Selected 5 layers with induction heads:
     Layer   0: KV heads [0, 2]
     Layer   5: KV heads [1, 3]
     Layer  12: KV heads [0]
     Layer  18: KV heads [2, 3]
     Layer  23: KV heads [1]
   --------------------------------------------------------------------------------
   === Echo Heads (copying matching) ===
   Selected 2 layers with echo heads:
     Layer   0: KV heads [3]
     Layer  12: KV heads [2]
   --------------------------------------------------------------------------------
   RA compress heads saved to: ./head_result/head.pt
   ================================================================================
   ```

2. 检查筛选结果是否覆盖预期范围：
   - **Induction heads（prefix matching）**：具有归纳头能力的 KV head，在段间 prefix 位置注意力分数较高。这些 head 擅长捕获跨段重复模式，是 KV Cache 压缩中需要保留的关键头。
   - **Echo heads（copying matching）**：具有回声头能力的 KV head，在段间 copying 位置注意力分数较高。这些 head 倾向于复制前一段对应位置的信息。
3. 若指定了 `--save_path`，`head.pt` 文件为 Python dict 序列化的 `.pt` 文件，结构如下：

   ```python
   {
       "prefix_matching": {0: [0, 2], 5: [1, 3], 12: [0], 18: [2, 3], 23: [1]},
       "copying": {0: [3], 12: [2]},
   }
   ```

   - `prefix_matching` 的 key 为 layer 索引（int），value 为该层被选中的 KV head 索引列表。
   - `copying` 的 key 为 layer 索引（int），value 为该层被选中的 KV head 索引列表。

4. 将 `head.pt` 文件交付给 MindIE 推理框架，用于长序列 KV Cache 压缩配置：在 MindIE 中把该文件路径配置到长序列 KV Cache 压缩相关参数中。用户可据此完成长序列压缩，压缩后的模型可用于后续推理部署；具体已适配的量化模型列表，请参见《[MindIE 是什么](https://www.hiascend.com/document/detail/zh/mindie/10RC3/whatismindie/mindie_what_0001.html)》的“MindIE 支持模型列表”章节。该产物不写入量化配置（不参与量化 YAML 的 `exclude` / `include` 等字段）。

**输出**：`head.pt` 文件路径（如指定 `--save_path`）及控制台输出的 head 列表。

**通过条件**：筛选结果中的 layer 与 head 索引在模型实际结构范围内；`head.pt` 可被 MindIE 的 KV Cache 压缩能力正确加载。

## 6. 验收条件

- 已选用 `attn_head` scope 与 `ra_compress` 指标。
- 适配器已满足 `RaCompressAnalysisInterface` 接口要求（实现 `get_proj_names` 与 `get_tokenizer`）或使用默认模式可正确匹配投影层。
- 分析命令成功产出 induction 与 echo head 列表；若指定 `--save_path`，`head.pt` 文件成功生成。
- head 索引经人工核对在模型实际结构范围内。

## 7. 异常处置

| 现象 | 处理方向 |
| --- | --- |
| 校准输入 tokenizer 缺失或首 token 解析失败 | 确认模型目录内 tokenizer 完整；`get_tokenizer()` 返回 `None` 或抛异常时指标侧会回落到兜底 token 并告警，正常返回模型自身的 tokenizer 即可消除告警 |
| RoPE 取值不可用（维度不匹配 / 序列过短） | 该取值由指标侧内置兜底完成，会依次回落到前向 kwargs（`position_embeddings`）、注意力模块与 `rope_theta` 推导；持续告警时检查模型 `config.json` 的 `rope_theta` 与 head_dim |
| `model_type` 未命中或告警走默认模型 | 按步骤 3 完成模型适配与 `--model_type` 注册，重新执行 `bash install.sh` 使注册生效后，再执行分析命令 |
| 分析报缺少接口或不支持 | 按步骤 3 与接口文档补齐 `RaCompressAnalysisInterface`（`get_proj_names` 与 `get_tokenizer`），或改用已支持该指标的 `model_type` 后，再执行分析命令 |
| 未识别到任何 induction 或 echo heads | 检查模型投影层命名是否与 `get_proj_names()` 返回的名称片段一致；确认其返回值非空（空串表示该投影不存在，不会匹配任何层），修正后重新执行分析命令 |
| 显存不足或运行失败 | 换更大显存设备，或减少 `--device_id` 指定的卡数后，再执行分析命令 |
| `head.pt` 加载失败 | 确认保存路径权限，并确认使用 `torch.load()` 加载 |

## 8. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| 模型适配 | 新模型接入与注册 | 《[LLM 大模型接入指南](../knowledge_base/ptq/llm/integration_guide_large_language_model_quantization.md)》 |
| RA Compress | 基于注意力头筛选的 KV Cache 压缩算法 | 本指南 [步骤 1](#步骤-1确认推荐指标) |
| Induction Head | 具有 prefix matching 能力的注意力头，擅长捕获跨段重复模式 | 本指南 [步骤 5](#步骤-5解读分析结果并交付-mindie-推理框架) |
| Echo Head | 具有 copying matching 能力的注意力头，倾向于复制前一段对应位置信息 | 本指南 [步骤 5](#步骤-5解读分析结果并交付-mindie-推理框架) |
| `RaCompressAnalysisInterface` | 模型适配器可选实现的接口：`get_proj_names` 与 `get_tokenizer` 均需实现 | [接口定义](../../../msmodelslim/processor/analysis/unary_operator/metrics/ra_compress/interface.py) |

## 9. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| `msmodelslim analyze attn_head` | Attention Head 筛选分析命令行入口 | 《[msmodelslim analyze 命令行 API](../api_reference/cli/msmodelslim_analyze.md)》；本指南 [命令行预览](#命令行预览) |
| `RaCompressAnalysisInterface` | 模型适配器 RA Compress 接口（`get_proj_names` / `get_tokenizer`） | [接口定义](../../../msmodelslim/processor/analysis/unary_operator/metrics/ra_compress/interface.py) |

## 10. 安全说明

- `trust_remote_code` 默认保持 `false`；仅当浮点模型必须执行自定义代码且来源可信、可审计时开启。
- 浮点权重与校准数据应按业务权限管控；勿将含业务数据的校准集或分析日志提交到公开渠道。
- `head.pt` 文件包含模型结构信息（层索引与头索引），应按业务权限管控。
