# Attention Head 筛选分析使用指南

## 1. 适用范围

本指南面向需要通过 `msmodelslim analyze attn_head`，按 **Attention Head** 粒度识别关键注意力头（induction heads / echo heads），并据此生成 `head.pt` 文件用于 RA Compress 压缩的开发者与算法工程师。

**适用场景**：

- **LLM 模型**：当前 `attn_head` 分析仅支持大语言模型。
- RA Compress 压缩方案设计：识别具有 prefix matching（归纳头）和 copying matching（回声头）能力的 KV cache 头，用于后续 KV cache 压缩配置。
- 精度不达标时，结合 head 筛选结果迭代 RA Compress 相关配置。

**不适用场景**：

- **多模态理解模型（VLM）**：不支持。
- **多模态生成模型**（文生图 / 文生视频等）：不支持。
- 需要按单层线性层回退或提位宽：请参见《[线性层敏感层分析使用指南](usage_sensitive_linear_analysis.md)》。
- 需要按 Decoder 块或整块 Attention / MLP / MoE 回退：请参见《[层级敏感层分析使用指南](usage_sensitive_layer_analysis.md)》。
- 配合 FA 量化识别需回退的 attention 模块：请参见《[Attention 敏感层分析使用指南](usage_sensitive_attn_analysis.md)》。

## 2. 流程关系与前置条件

**上级流程**：《[新模型量化调优流程](process_new_model_quantization_tuning.md)》。

**前置条件**：

- 已安装 msModelSlim（详见《[安装指南](../install_guide/install_guide.md)》）。
- 目标模型为 LLM，且已确定可用的 `--model_type`（与支持矩阵 / 适配器注册名一致，大小写敏感；通常已在上级流程或权重量化流程中完成适配）。
- 已具备可用的昇腾 NPU（或仅做小规模调试时使用 `--device cpu`）。
- 目标适配器已实现 `RaCompressAnalysisInterface` 接口（提供 Q / K / QKV 投影层名称模式），或使用默认模式（`q_proj` / `k_proj` / `qkv_proj`）。

**后续操作**：

- 将生成的 `head.pt` 文件提供给 RA Compress 量化流程，或在量化配置中引用 head 筛选结果。

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型目录 | 用户本地路径；一般自 ModelScope / Hugging Face 获取 | 含模型配置、权重分片及 tokenizer 等 | 路径有效，可被目标 `model_type` 加载 |
| 输入 | 校准集 | 工具内置 [`calib_dummy.jsonl`](../../../lab_calib/) 短名称或用户本地路径 | `.json` / `.jsonl`；tokenize 后总长度须 ≥ 10000（2500 × 4 段重复） | 可被 `msmodelslim analyze` 解析且 token 长度满足要求 |
| 交付件 | head 筛选结果 | 标准输出（induction / echo head 列表）+ 可选 `head.pt` 文件 | `head.pt` 为 Python dict 序列化的 `.pt` 文件 | 能识别预期 layer / head 范围，`head.pt` 可被后续量化流程加载 |

## 4. 流程总览

确认 `attn_head` 场景与 `ra_compress` 指标后，准备浮点权重与专用校准集，按需完成模型适配（含 RA Compress 接口），再执行分析命令并解读结果，用于生成 `head.pt` 或调整压缩配置。

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
msmodelslim analyze attn_head \
  --model_type ${MODEL_TYPE} \          # 已注册或支持矩阵中的模型名，大小写敏感
  --model_path ${MODEL_PATH} \          # 浮点权重目录
  --metrics ra_compress \               # 分析指标，固定为 ra_compress
  --calib_dataset calib_dummy.jsonl \   # 须显式指定为 calib_dummy.jsonl（合成重复段校准集，tokenize 后总长度 ≥ 10000）
  --device npu \                        # 分析设备：npu / cpu
  --trust_remote_code False \           # 默认 False；仅可信模型必要时设为 True
  --save_path ${SAVE_PATH}              # 可选；指定则保存 head.pt，不指定仅打印到控制台
```

### 步骤 1：确认推荐指标

**目标**：选定命令行预览中的 `${METRICS}`。

**操作**：

将 `${METRICS}` 设为下表之一：

| 可选指标 | 适用说明 | 算法说明 | 推荐 |
| --- | --- | --- | --- |
| `ra_compress` | 识别具有 prefix matching 和 copying matching 能力的注意力头，用于 KV cache 压缩 | RA Compress 算法：对 Q@K^T 注意力分数在特定段偏移位置取均值，按 ratio 选 top heads | **首选**（当前唯一可选） |

**输出**：已选定的 `${METRICS}`。

**通过条件**：`${METRICS}` 属于上表可选值。

### 步骤 2：获取浮点权重与校准数据

**目标**：准备可加载的浮点模型目录与满足 token 长度要求的校准集。

**操作**：

1. 从 [ModelScope](https://www.modelscope.cn/)、[Hugging Face](https://huggingface.co/) 或团队内部模型存放位置获取完整权重到本地目录；具体下载方式以对应社区或仓库文档为准。
2. 核对目录含配置、权重分片及 tokenizer 等附属文件。若官方页面提供文件校验值（如 MD5/SHA256）或明确的版本号/提交号，与本地下载结果比对一致即可。
3. 准备校准集：`ra_compress` 算法要求校准数据 tokenize 后总长度 ≥ **10000 tokens**（即 `DUMMY_INPUT_LENGTH=2500` × `REPET_TIMES=4` 段重复）。工具内置 [`calib_dummy.jsonl`](../../../lab_calib/) 已满足此要求，`attn_head` 子命令须在命令行显式指定 `--calib_dataset calib_dummy.jsonl`（该子命令不再为此场景设置默认值）。也可使用自有校准集，但须确保 tokenize 后的 token 总数 ≥ 10000，否则分析将跳过分数计算并返回空结果。

**输出**：浮点模型目录与校准集路径（或工具内置校准集短名称）。

**通过条件**：模型目录可加载；校准集 tokenize 后 token 总数 ≥ 10000。

### 步骤 3：完成模型适配

**目标**：确保存在可被 `msmodelslim analyze --model_type <模型名>` 命中的模型适配器；适配器已实现 `RaCompressAnalysisInterface` 接口或使用默认投影层名称模式。

**操作**：

- **尚未接入的模型**：须先完成适配器开发与注册，再进入步骤 4。通用适配要求见《[LLM 大模型接入指南](../knowledge_base/model/integrating_models.md)》。
- **支持矩阵中已接入的模型**：可跳过通用适配，确认所用 `--model_type` 名称即可。
- **算法侧额外接口**：`ra_compress` 要求适配器实现 `RaCompressAnalysisInterface` 接口，提供 Q / K / QKV 投影层名称模式。若适配器未实现该接口，将使用默认模式（`q_proj` / `k_proj` / `qkv_proj`）。若模型投影层命名与默认模式不同，须按接口文档补齐。接口定义见 [ra_compress/interface.py](../../../msmodelslim/processor/analysis/unary_operator/metrics/ra_compress/interface.py)。

完成或修改适配器后，在仓库根目录重新执行 `bash install.sh`，使注册生效。

**输出**：已注册且可被 CLI 命中的模型适配器（对应 `--model_type`）；适配器已具备 `RaCompressAnalysisInterface` 能力或使用默认模式可正确匹配投影层。

**通过条件**：使用该 `model_type` 启动分析时能正确命中适配器；能正确识别 Q / K / QKV 投影层并采集输出。

### 步骤 4：执行分析命令

**目标**：运行 `msmodelslim analyze attn_head`，得到 induction heads / echo heads 筛选结果，并可选保存 `head.pt`。

**执行前检查**：

- 已完成步骤 1～3 的指标、权重与模型适配确认。
- `trust_remote_code` 默认 `False`；仅当模型必须执行仓库内自定义代码且来源可信时设为 `True`。
- `--calib_dataset` 须显式指定为 `calib_dummy.jsonl`（`attn_head` 子命令不再为此场景设置默认值）；如使用自定义校准集，请确认 token 长度满足要求（见步骤 2）。

**操作**：

按[命令行预览](#命令行预览)将变量替换为实际值后执行。示例：

```bash
msmodelslim analyze attn_head \
  --model_type Qwen2.5-7B-Instruct \
  --model_path /data/models/Qwen/Qwen2.5-7B-Instruct/ \
  --metrics ra_compress \
  --calib_dataset calib_dummy.jsonl \
  --device npu \
  --trust_remote_code True \
  --save_path ./head_result
```

**输出**：标准输出中的 induction / echo head 筛选结果。若指定 `--save_path`，则同时保存 `head.pt` 文件。

**通过条件**：命令正常结束；输出含 induction heads 和 echo heads 列表；若指定 `--save_path`，`head.pt` 文件成功生成。

**审计记录**：实际命令行、`model_type` / `${METRICS}`、校准集路径、标准输出结果摘要（或保存的日志路径）、`head.pt` 文件路径（如指定）。

### 步骤 5：解读分析结果并用于配置

**目标**：读懂标准输出中的 induction / echo head 筛选结果，并将 `head.pt` 用于后续 RA Compress 量化流程。

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
   - **Induction heads（prefix matching）**：具有归纳头能力的 KV head，在段间 prefix 位置注意力分数较高。这些 head 擅长捕获跨段重复模式，是 KV cache 压缩中需要保留的关键头。
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

4. 将 `head.pt` 文件路径提供给后续 RA Compress 量化流程，或在量化配置中引用 head 筛选结果。

**输出**：`head.pt` 文件路径（如指定 `--save_path`）或控制台输出的 head 列表。

**通过条件**：筛选结果中的 layer / head 索引在模型实际结构范围内；`head.pt` 可被后续量化流程正确加载。

## 6. 全局验收条件

- 已选用 `attn_head` scope 与 `ra_compress` 指标。
- 适配器已满足 `RaCompressAnalysisInterface` 接口要求或使用默认模式可正确匹配投影层。
- 校准集 tokenize 后 token 总数 ≥ 10000。
- 分析命令成功产出 induction / echo head 列表；若指定 `--save_path`，`head.pt` 文件成功生成。
- head 索引经人工核对在模型实际结构范围内。

## 7. 全局异常处置

| 现象 | 处理方向 |
| --- | --- |
| 校准集 token 长度不足（< 10000） | 使用工具内置 `calib_dummy.jsonl`（默认值），或确保自定义校准集 tokenize 后总长度 ≥ 10000 |
| 校准集格式错误或无法读取 | 核对 `.json`/`.jsonl` 格式、路径与权限；路径解析与示例集见步骤 2 |
| `model_type` 未命中或告警走默认模型 | 尚未完成模型适配与 `--model_type` 注册；先按步骤 3 补齐并重新 `bash install.sh` |
| 分析报缺少接口 / 不支持 | 回到步骤 3，按接口文档补齐 `RaCompressAnalysisInterface`，或改用已支持该指标的 `model_type` |
| 未识别到任何 induction / echo heads | 检查校准集 token 长度是否满足要求；检查模型投影层命名是否与适配器配置一致 |
| 显存不足或运行失败 | 缩小校准集，或换更大显存设备 |
| `head.pt` 加载失败 | 确认保存路径权限；确认使用 `torch.load()` 加载 |

## 8. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| 模型适配 | 新模型接入与注册 | 《[LLM 大模型接入指南](../knowledge_base/model/integrating_models.md)》 |
| RA Compress | 基于注意力头筛选的 KV cache 压缩算法 | 本指南 [步骤 1](#步骤-1确认推荐指标) |
| Induction Head | 具有 prefix matching 能力的注意力头，擅长捕获跨段重复模式 | 本指南 [步骤 5](#步骤-5解读分析结果并用于配置) |
| Echo Head | 具有 copying matching 能力的注意力头，倾向于复制前一段对应位置信息 | 本指南 [步骤 5](#步骤-5解读分析结果并用于配置) |
| `RaCompressAnalysisInterface` | 模型适配器需实现的接口，提供 Q / K / QKV 投影层名称模式 | [接口定义](../../../msmodelslim/processor/analysis/unary_operator/metrics/ra_compress/interface.py) |

## 9. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| `msmodelslim analyze attn_head` | Attention Head 筛选分析命令行入口 | 本指南 [命令行预览](#命令行预览) |
| `RaCompressAnalysisInterface` | 模型适配器 RA Compress 接口 | [接口定义](../../../msmodelslim/processor/analysis/unary_operator/metrics/ra_compress/interface.py) |

## 10. 安全说明

- `trust_remote_code` 默认保持 `False`；仅当浮点仓库必须执行自定义代码且来源可信、可审计时开启。
- 浮点权重与校准数据应按业务权限管控；勿将含业务数据的校准集或分析日志提交到公开渠道。
- `head.pt` 文件包含模型结构信息（层索引与头索引），应按业务权限管控。
