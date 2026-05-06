---
toc_depth: 3
---
# 量化敏感层分析工具使用指南

## 简介

`analyze` 是 msModelSlim 工具中的量化敏感层分析功能接口，用于分析模型中各层的量化敏感度，帮助用户识别量化敏感层，从而进行针对性的优化。

下图为端到端量化中各环节流程示意图，其中**敏感层分析**（`msmodelslim analyze`）属于**方案设计**阶段：在撰写或迭代量化 YAML 时，基于校准数据得到层/结构敏感度排序，用于决定哪些层做回退，即哪些层少量化或不量化。分析完成后，日志中可出现便于粘贴的 YAML 片段，格式说明见[输出说明](#输出说明)。

```text
                            YAML            权重
    浮点权重 ─────▶ 方案设计 ─────▶ 模型量化 ─────▶ 测评 ────▶ 量化权重
                        ▲                                   │
                        └──────── 精度与性能反馈 ────────────┘
```

**根据敏感层分析结果修改 YAML 的常见做法**：

- **回退到浮点**：对排序靠前的层，在量化 YAML 中通过 **`exclude`**（或收窄 **`include`**）使这些层**不参与**当前量化，推理时仍按 **浮点路径**执行；若涉及 QKV 等成组模块，建议同一组采用一致策略，避免精度断层。
- **低比特下提位宽**：在 **W4/A4 等低比特**整体方案中，对分析结果中**敏感度偏高**的层不必整网抬升比特，可在 `spec.process` 里**仅对这些层单独增配或覆盖**，把该段的 **4bit 提到 8bit**，在其余层仍保持低比特的前提下局部换精度。
- **配合测评迭代**：测评精度未达标时，结合**与目标的精度差距**与敏感层排序，选定优先回退或提位宽的层。可先**多回退一批层**（或采用更保守的 YAML 策略），优先让精度快速达标；达标后再逐步**减少回退层**、在精度与压缩之间做增量收敛。

粘贴控制台给出的 YAML 片段时，请核对 **Processor 类型、`include`/`exclude` 通配** 与现有 `spec.process` 是否一致，必要时手工合并进已有配置列表。

## 使用前准备

安装 msModelSlim 工具，详情请参见《[msModelSlim工具安装指南](../../getting_started/install_guide.md)》。

## 功能介绍

### 昇腾AI处理器支持情况

目前仅支持Atlas A3 训练系列产品/Atlas A3 推理系列产品和Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件，且内存需要大于1.5倍模型大小。

### 功能说明

- **多维度分析**能够从数据分布、稳健性、峰态特征、注意力以及层级输出差异等多个维度，精准评估层敏感度。按分析层级划分如下：
  - **linear层级**衡量算法：`std`、`quantile`、`kurtosis`
  - **attention层级**衡量算法：`mse`
  - **layer层级**衡量算法：`mse_layer_wise`、`mse_model_wise`
- **灵活配置**: 支持自定义校准数据集（JSON/JSONL格式）、层名匹配以及丰富的参数选项，满足不同场景的量化需求。
- **智能输出**: 支持打印Top K敏感层列表，实际打印数量可能会大于或等于目标数量，如QKV一起打印。

### 注意事项

- transformer版本依赖于模型，与量化功能无关。
- 实际回退的层数受推理引擎实现的限制，因此可能与topk参数设置存在一些差异。
- topk默认值为15，作为回退经验值仅供参考，如果打印层涉及qkv，会将qkv一起输出。
- 由于安全规范，trust_remote_code默认为False。
- 敏感层分析目前仅支持大语言模型。

### 命令格式

```bash
msmodelslim analyze [scope] [参数选项]
```

其中 `scope` 用于指定分析范围（linear / attention / layer 级别等），**不指定时默认使用 `linear`**。

可选 scope：

- `linear`: 线性层敏感度分析，输出的结果为线性层排序结果。
- `attn`: attention 结构敏感度分析，输出的结果为注意力层排序结果。
- `layer`: 整个 Decoder 层级敏感度分析，输出的结果为整层排序结果。

> [!note] 兼容说明
>
> - **旧命令仍可用**：仍支持旧写法 `msmodelslim analyze --model_type ... --model_path ...`（不显式写 scope），但在旧写法下 `--metrics` **仅支持** `"std"`, `"quantile"`, `"kurtosis"`, `"attention_mse"` 四种类型，其中`"attention_mse"`对应于新写法 `msmodelslim analyze attn --metrics mse ...`。
> - **推荐使用新写法**：显式指定 scope（`linear / attn / layer`），可获得更清晰的 help 和更稳定的参数语义。

### 参数说明

#### 通用参数（所有 scope 共享）

| 参数 | 类型 | 默认值 | 描述 | 示例值 |
|------|------|--------|------|--------|
| `--model_type` | `str` | - | 模型类型，用于指定要分析的模型架构，见[参数说明](#参数说明)（含参数选择注意事项） | `Qwen2.5-7B-Instruct` |
| `--model_path` | `str` | - | 原始模型的路径，建议使用绝对路径 | `/path/Qwen2.5-7B-Instruct` |
| `--device` | `str` | `npu` | 指定运行分析的目标设备，可选值：`npu`, `cpu`。 | `npu` |
| `--calib_dataset` | `str` | `"mix_calib.jsonl"` | 校准数据集文件路径，支持JSON/JSONL格式，以.json或.jsonl结尾。支持绝对路径和相对路径。 | `/path/data.jsonl` |
| `--topk` | `int` | `15` | 输出Top K敏感的层数量，为大于0的整数。推荐范围为10~20。 | `15` |
| `--trust_remote_code` | `bool` | `False` | 是否信任远程代码，需要用户自行保障安全性。可选值：`True`, `False`。 | `False` |
| `-h, --help` | - | - | 命令行参数帮助信息 | - |

#### `linear` 参数（线性层分析）

命令格式：

```bash
msmodelslim analyze linear [通用参数] [linear参数]
```

| 参数 | 类型 | 默认值 | 描述 | 示例值 |
|------|------|--------|------|--------|
| `--pattern` | `List[str]` | `["*"]` | 待分析的层名称列表，支持通配符匹配。支持设置多个pattern，使用空格分隔。 | `"*down_proj"` `"*up_proj"` |
| `--metrics` | `str` | `"kurtosis"` | 分析使用的度量算法，可选值：`"std"`, `"quantile"`, `"kurtosis"` | `"kurtosis"` |

#### `attn` 参数（attention 结构分析）

命令格式：

```bash
msmodelslim analyze attn [通用参数] [attn参数]
```

| 参数 | 类型 | 默认值 | 描述 | 示例值 |
|------|------|--------|------|--------|
| `--metrics` | `str` | `"mse"` | 分析使用的度量算法，可选值：`"mse"` | `"mse"` |

#### `layer` 参数（decoder层级输出）

命令格式：

```bash
msmodelslim analyze layer [通用参数] [layer参数]
```

| 参数 | 类型 | 默认值 | 描述 | 示例值 |
|------|------|--------|------|--------|
| `--quant_modules` | `List[str]` | `["*"]` | 目标模块列表，支持通配符匹配，用于指定参与对比的模块范围。 | `"*self_attn*"` `"*mlp*"` |
| `--metrics` | `str` | `"mse_layer_wise"` | 分析使用的度量算法，可选值：`"mse_model_wise"`, `"mse_layer_wise"` | `"mse_layer_wise"` |

**参数选择注意事项**：

- **model_type 支持说明**：多数敏感层分析指标下可选的 `model_type` 与 ModelslimV1 量化一致；**`attn --metrics mse`**（旧称 `attention_mse`）不在此列：须在模型适配器中实现 **AttentionMSEAnalysisInterface**，仅部分 `model_type` 已接入；**`mse_model_wise`** 另受逐层链式前向与**模型最终输出**对齐、校准规模与显存等约束，部分架构可能在某层之后无法继续分析。
- **模型路径和数据集要求**：**`model_path`** 须为真实存在的绝对或相对路径，且包含有效权重与配置。**`calib_dataset`** 须为 `.json` / `.jsonl`；JSON 为字符串列表（可参考 `lab_calib/anti_prompt.json`），JSONL 为每行一个 JSON 对象（可参考 `lab_calib` 下如 `boolq.jsonl`）；相对路径在 `lab_calib` 目录下解析。
- **算法推荐**：`linear` 可首选 **`kurtosis`**，再按需对比 `std` / `quantile`；`layer` **优先 `mse_layer_wise`**；仅当关注 Attention 量化且 `model_type` 已适配时，使用 **`attn --metrics mse`**。

### 分析算法说明

敏感度度量按**分析范围（scope）**与输出粒度分为三类；各类下 `--metrics` 的详细说明见下列分篇（文档路径：`docs/zh/quantization_algorithms/sensitive_layer_analysis/`）。

#### linear（线性层）

对模型中**单个线性层**（及实现支持的卷积层等）做敏感度分析，结果为**线性层粒度**排序。可选 `--metrics`：`std`、`quantile`、`kurtosis`；**均无需**模型适配器额外实现分析接口。

- 《[Std：敏感层分析算法说明](../../quantization_algorithms/sensitive_layer_analysis/std.md)》
- 《[Quantile：敏感层分析算法说明](../../quantization_algorithms/sensitive_layer_analysis/quantile.md)》
- 《[Kurtosis：敏感层分析算法说明](../../quantization_algorithms/sensitive_layer_analysis/kurtosis.md)》

#### attn（attention 结构）

对模型中**attention 结构**做敏感度分析，结果为**attention 模块粒度**排序。可选 `--metrics`：`mse`（旧写法中为 `attention_mse`）。需对应 `model_type` 的模型适配器实现 **AttentionMSEAnalysisInterface**。

- 《[Attention MSE（mse）：敏感层分析算法说明](../../quantization_algorithms/sensitive_layer_analysis/attention_mse.md)》

#### layer（decoder 层级输出）

对**Decoder block**做敏感度分析，结果为**层级粒度**排序，用于整层回退或整块回退（如整块 attention / MLP）。可选 `--metrics`：`mse_layer_wise`、`mse_model_wise`；**均无需**模型适配器额外实现分析接口；`mse_model_wise` 在部分模型上受链式前向对齐约束。

- 《[层级 MSE（mse_layer_wise）：敏感层分析算法说明](../../quantization_algorithms/sensitive_layer_analysis/mse_layer_wise.md)》
- 《[模型级 MSE（mse_model_wise）：敏感层分析算法说明](../../quantization_algorithms/sensitive_layer_analysis/mse_model_wise.md)》

### 使用示例

linear 分析示例：

```bash
msmodelslim analyze linear \
    --model_type Qwen2.5-7B-Instruct \
    --model_path ${model_path} \
    --metrics kurtosis \
    --calib_dataset ${calib_dataset} \
    --pattern "*.down_proj*" "*.o_proj*" \
    --topk 15 \
    --device npu
```

attn 分析示例：

```bash
msmodelslim analyze attn \
    --model_type DeepSeek-V3 \
    --model_path ${model_path} \
    --metrics mse \
    --calib_dataset ${calib_dataset} \
    --topk 15 \
    --device npu
```

layer 分析示例：

```bash
msmodelslim analyze layer \
    --model_type Qwen3-32B \
    --model_path ${model_path} \
    --metrics mse_layer_wise \
    --quant_modules "*mlp*" \
    --calib_dataset ${calib_dataset} \
    --topk 15 \
    --device npu
```

### 输出说明

支持纯净的yaml格式打印，方便用户直接粘贴到yaml文件中。

#### 控制台输出

以上述 `linear` 示例（`kurtosis` + `--pattern`）为例，控制台输出如下：

```text
...
msmodelslim.core.analysis_service.pipeline_analysis.service - INFO - ==========ANALYSIS: Starting Layer Analysis==========
msmodelslim.core.analysis_service.pipeline_analysis.service - INFO - Analysis metrics: kurtosis
msmodelslim.core.analysis_service.pipeline_analysis.service - INFO - Layer patterns: ['*.down_proj*', '*.o_proj*']
msmodelslim.core.runner.layer_wise_runner - INFO - Start to handle dataset
msmodelslim.core.runner.layer_wise_runner - INFO - Handle dataset success
msmodelslim.core.runner.layer_wise_runner - INFO - Start to init model
msmodelslim.core.runner.layer_wise_runner - INFO - Init model success
msmodelslim.core.runner.layer_wise_runner - INFO - KV cache requirement: False
msmodelslim.core.runner.layer_wise_runner - INFO - Scheduler 3 unit: [LoadProcessor(device=npu:0, non_blocking=False), UnaryAnalysisProcessor, LoadProcessor(device=meta, non_blocking=False)]
msmodelslim.core.runner.layer_wise_runner - INFO - Run processor LoadProcessor(device=npu:0, non_blocking=False) for "model.layers.0"
msmodelslim.core.runner.layer_wise_runner - INFO - Run processor UnaryAnalysisProcessor for "model.layers.0"
msmodelslim.core.runner.layer_wise_runner - INFO - Run processor LoadProcessor(device=meta, non_blocking=False) for "model.layers.0"
...
msmodelslim.app.analysis.application - INFO - === Layer Analysis Results (kurtosis method) ===
msmodelslim.app.analysis.application - INFO - Patterns analyzed: ['*.down_proj*', '*.o_proj*']
msmodelslim.app.analysis.application - INFO - Total layers analyzed: 128
msmodelslim.app.analysis.application - INFO - Layer Sensitivity Scores (higher score = more sensitive to quantization):
...
msmodelslim.app.analysis.application - INFO - === YAML Format for quantization ===
...
msmodelslim.app.analysis.application - INFO - === End of YAML Format ===
msmodelslim.app.analysis.application - INFO - ===========ANALYSIS COMPLETE===========
```

## FAQ

### 问题现象：校准数据集文件格式错误或无法读取

**解决方案**：

1. 确认文件格式为支持的JSON或JSONL格式。
2. 确保每条记录包含必要的字段。
3. 验证文件路径是否正确。
4. 确认校准集文件存在可读权限。

### 问题现象：输入不支持的model_type会发生什么？

**解决方案**：
当输入的model_type不在支持列表中时：

- 系统会打印warning日志，提示使用默认模型。
- 自动使用默认模型进行处理。
- 可能无法获得最佳的分析效果。
- **建议**：优先使用与所选`--metrics`及[参数说明](#参数说明)中「参数选择注意事项」约定一致的标准`model_type`，以获得最佳兼容性与分析效果。
