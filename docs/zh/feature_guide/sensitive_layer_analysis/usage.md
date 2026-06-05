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
- **低比特下提位宽**：在 **W4/A4 等低比特**整体方案中，对分析结果中**敏感度偏高**的层不必整网抬升比特，可在 `spec.process` 里**仅对这些层单独增配或覆盖**，把该段的 **4 bit 提到 8 bit**，在其余层仍保持低比特的前提下局部换精度。
- **配合测评迭代**：测评精度未达标时，结合精度差距与敏感层排序，选定优先回退或提位宽的层。可先**多回退一批层**（或采用更保守的 YAML 策略），优先让精度快速达标；达标后再逐步**减少回退层**、在精度与压缩之间做增量收敛。

敏感层分析完成后，控制台会输出符合量化 YAML 配置格式的敏感层排序结果，可直接复制并粘贴至量化配置的 include 或 exclude 中（通常用于 type: linear_quant 的 Processor），请检查其中的层名通配符是否覆盖预期范围，再以新的配置进行量化。

## 使用前准备

安装 msModelSlim 工具，详情请参见《[msModelSlim 工具安装指南](../../getting_started/install_guide.md)》。

## 功能介绍

### 功能说明

- **多维度分析**能够从数据分布、稳健性、峰态特征、注意力以及层级输出差异等多个维度，精准评估层敏感度。按分析层级划分如下：
  - **linear层级**衡量算法：`std`、`quantile`、`kurtosis`
  - **attention层级**衡量算法：`mse`
  - **layer层级**衡量算法：`mse_layer_wise`、`mse_model_wise`
- **灵活配置**：支持自定义校准数据集（JSON/JSONL 格式）、层名匹配以及丰富的参数选项，满足不同场景的量化需求。
- **智能输出**：支持打印 Top K 敏感层列表，实际打印数量可能会大于或等于目标数量，如 QKV 一起打印。

### 注意事项

- transformers 版本依赖于模型，与量化功能无关。
- 实际回退的层数受推理引擎实现的限制，因此可能与 topk 参数设置存在一些差异。
- topk 默认值为 15，作为回退经验值仅供参考，如果打印层涉及 QKV，会将 QKV 一起输出。
- 由于安全规范，trust_remote_code 默认为 False。
- 敏感层分析目前仅支持大语言模型。

### 命令格式

```bash
msmodelslim analyze [scope] [参数选项]
```

其中 `scope` 用于指定分析范围（linear/attention/layer 级别等）。

可选 scope：

- `linear`：线性层敏感度分析，输出的结果为线性层排序结果。
- `attn`：attention 结构敏感度分析，输出的结果为注意力层排序结果。
- `layer`：整个 Decoder 层级敏感度分析，输出的结果为整层排序结果。

> [!note] 兼容说明
>
> - **旧命令仍可用但后续将日落**：仍支持旧写法 `msmodelslim analyze --model_type ... --model_path ...`（不显式写 scope），但旧写法下 `--metrics` 仅支持 `"std"`, `"quantile"`, `"kurtosis"`, `"attention_mse"` 四种类型，其中 `"attention_mse"` 对应新写法 `msmodelslim analyze attn --metrics mse`。新写法下 attn scope 的 `--metrics` 可选值为 `"mse"`，不再保留 `"attention_mse"` 这一名称。
> - **推荐使用新写法**：显式指定 scope（`linear/attn/layer`），可获得更清晰的 help 和更稳定的参数语义。

### 参数说明

#### 通用参数（所有 scope 共享）

| 参数 | 类型 | 默认值 | 描述 | 示例值 |
|------|------|--------|------|--------|
| `--model_type` | `str` | - | 模型类型，用于指定要分析的模型架构 | `Qwen2.5-7B-Instruct` |
| `--model_path` | `str` | - | 原始模型的路径，建议使用绝对路径 | `/path/Qwen2.5-7B-Instruct` |
| `--device` | `str` | `npu` | 指定运行分析的目标设备，可选值：`npu`, `cpu`。 | `npu` |
| `--calib_dataset` | `str` | `"mix_calib.jsonl"` | 校准数据集文件路径，支持JSON/JSONL格式，以.json或.jsonl结尾。支持绝对路径和相对路径。 | `/path/data.jsonl` |
| `--topk` | `int` | `15` | 输出TopK敏感的层数量，为大于0的整数。推荐范围为10~20。 | `15` |
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

#### 参数选择注意事项

**model_type 支持说明**

- 多数敏感层分析指标（`linear`/`layer` 下的 metrics）下可选的 `model_type` 与 ModelslimV1 量化一致。
- `attn --metrics mse` 不在上述范围，需对应 `model_type` 的模型适配器实现 **AttentionMSEAnalysisInterface** 方可使用。该算法的模型支持列表见[Attention MSE 算法适用要求](../../quantization_algorithms/sensitive_layer_analysis/attention_mse.md#适用要求)。

**模型路径和数据集要求**

- **`model_path`**：须为真实存在的绝对或相对路径，且包含有效权重与配置。
- **`calib_dataset`**：须为 `.json`/`.jsonl` 格式。其中 JSON 为字符串列表，每项表示一条校准文本，而 JSONL 为每行一个 JSON 对象。可直接使用工具提供的 [`lab_calib`](../../../../lab_calib/) 示例校准集目录下的校准集。若数据集使用相对路径，解析规则如下：优先在命令启动路径下查找，找到则直接使用；未找到时，在示例校准集目录下匹配同名数据集；均未找到时抛出异常。

**层级选择**

- `linear`：对单个线性层做细粒度敏感度分析，支持每层的精确回退。不建议用于 MoE 模型的专家部分，因专家层线性层数量庞大且并非所有线性层都参与激活，分析结果参考价值有限。
- `attn`：对注意力模块做敏感度分析，目前多用于需配合 Flash Attention 3 激活量化的场景，帮助识别需要回退的注意力层。
- `layer`：对 Decoder 块做整体敏感度分析，输出块粒度排序结果，用于需要整层或整块（Attention/MoE/MLP）回退的场景。

各层级下不同算法（metrics）的选择，参见下方[分析算法说明](#分析算法说明)。

### 分析算法说明

敏感度度量按分析范围（scope）与输出粒度分为三类，各类下 `--metrics` 的详细说明见下列分篇。

#### linear（线性层）

对模型中**单个线性层**（及实现支持的卷积层等）做敏感度分析，结果为**线性层粒度**排序。可选 `--metrics`：`std`、`quantile`、`kurtosis`；均无需模型适配器额外实现分析接口。各算法详细说明请参见：

- 《[Std：敏感层分析算法说明](../../quantization_algorithms/sensitive_layer_analysis/std.md)》
- 《[Quantile：敏感层分析算法说明](../../quantization_algorithms/sensitive_layer_analysis/quantile.md)》
- 《[Kurtosis：敏感层分析算法说明](../../quantization_algorithms/sensitive_layer_analysis/kurtosis.md)》

> **推荐**：`linear` 可首选 `kurtosis`，该指标对激活尖峰敏感，能有效识别分布极端值对量化的影响；若数据含较多离群点可配合 `quantile`，若关注范围与离散度比值可配合 `std`。

#### attn（attention 结构）

对模型中**attention 结构**做敏感度分析，结果为**attention 模块粒度**排序。可选 `--metrics`：`mse`。需对应 `model_type` 的模型适配器实现 **AttentionMSEAnalysisInterface**。各算法详细说明请参见：

- 《[Attention MSE（mse）：敏感层分析算法说明](../../quantization_algorithms/sensitive_layer_analysis/attention_mse.md)》

#### layer（decoder 层级输出）

对**Decoder block**做敏感度分析，结果为**层级粒度**排序，用于整层回退或整块回退（如整块 attention/MLP）。可选 `--metrics`：`mse_layer_wise`、`mse_model_wise`；均无需模型适配器额外实现分析接口。各算法详细说明请参见：

- 《[层级 MSE（mse_layer_wise）：敏感层分析算法说明](../../quantization_algorithms/sensitive_layer_analysis/mse_layer_wise.md)》
- 《[模型级 MSE（mse_model_wise）：敏感层分析算法说明](../../quantization_algorithms/sensitive_layer_analysis/mse_model_wise.md)》

> **推荐**：`layer` 优先使用 `mse_layer_wise`。`mse_model_wise` 依赖链式前向将上一层输出传入下一层来模拟真实推理路径，部分架构上可能因张量形状无法对齐而跳过后续层，且校准规模与中间缓存也会增加显存压力。

### 使用示例

以下示例中 `${model_path}` 表示原始模型路径，`${calib_dataset}` 表示校准数据集文件路径，请根据实际情况替换为实际值。

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

运行分析后，控制台会输出两部分关键信息：**敏感层排序列表**（Score 从高到低，分数越高表示该层对量化越敏感）和 **可直接粘贴的 YAML 配置片段**。YAML 片段中已按敏感度排序生成回退层列表，用户可直接复制到量化配置文件中使用。

根据 scope 的不同，输出格式分为以下三类。

> [!note] 注意
> - 进行linear层级分析时，对于 QKV（`q_proj`、`k_proj`、`v_proj`）等成组模块，同一组具有相同的分数，输出时会一起列出。
> - 某些成组模块必须同时排除，如 MLP 中的 `up_proj` 与 `gate_proj`，若只回退其中一个可能会导致模型无法部署。

#### linear 输出

输出结果按层粒度排序，每行包含具体的层名称和对应的敏感度分数：

```text
=== Layer Analysis Results (std method) ===
Patterns analyzed: ['*']
Total layers analyzed: 252
Layer Sensitivity Scores (higher score = more sensitive to quantization):
--------------------------------------------------------------------------------
  1. model.layers.6.mlp.down_proj                       | Score:   2.1326e+02
  2. model.layers.14.mlp.down_proj                      | Score:   1.6952e+02
  3. model.layers.3.mlp.gate_proj                       | Score:   1.5288e+02
  ...
--------------------------------------------------------------------------------
Top 80 most sensitive layers selected for disable_names

=== YAML Format for quantization ===

top 80:
  - 'model.layers.6.mlp.down_proj'
  - 'model.layers.14.mlp.down_proj'
  - 'model.layers.3.mlp.gate_proj'
  ...

=== End of YAML Format ===
```

#### attn 输出

输出结果按注意力模块粒度排序，每行包含注意力模块名称和对应的敏感度分数：

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

#### layer 输出

输出结果按 Decoder 块粒度排序。YAML 中回退层名默认包含通配符（如 `model.layers.2.*`），表示回退该 Decoder 层内的所有子模块。若只需要回退该层内的某个具体结构（如仅 MLP 或仅 Attention），可在 YAML 中将通配符替换为具体模块名，或在量化配置中配合 `include` 指定需要量化的子模块范围。

```text
=== Layer Analysis Results (mse_layer_wise method) ===
Patterns analyzed: ['*']
Total layers analyzed: 36
Layer Sensitivity Scores (higher score = more sensitive to quantization):
--------------------------------------------------------------------------------
  1. model.layers.2                                     | Score:   2.4396e+03
  2. model.layers.35                                    | Score:   1.3626e+02
  3. model.layers.34                                    | Score:   1.2008e+02
  ...
--------------------------------------------------------------------------------
Top 36 most sensitive layers selected for disable_names

=== YAML Format for quantization ===

top 36:
  - 'model.layers.2.*'
  - 'model.layers.35.*'
  - 'model.layers.34.*'
  ...

=== End of YAML Format ===
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
- **建议**：优先使用与所选 `--metrics` 及[参数选择注意事项](#参数选择注意事项)约定一致的标准 `model_type`，以获得最佳兼容性与分析效果。
