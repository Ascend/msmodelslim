# RA Compress 注意力头分析配置使用指南

## 1. 适用范围

本指南面向需要使用 [RA Compress 注意力头分析算法](./term_ra_compress.md) 的用户。RA Compress（`ra_compress`）作为 `msmodelslim analyze attn_head` 的 metrics 指标，用于 KV 注意力头粒度的筛选，输出归纳头（induction heads）与回声头（echo heads）的入选索引，供后续 KV cache 压缩流程使用。

适用场景：

- 需要为长序列推理生成 KV cache 压缩配置（`head.pt`）的场景；
- 需要了解模型中哪些 KV head 承担跨段信息传递（归纳 / 复制头）的场景。

不适用场景：

- **多模态理解模型（VLM）** 与 **多模态生成模型**（文生图 / 文生视频等）：`ra_compress` / `analyze attn_head` 当前仅支持大语言模型（LLM）；
- 目标 `model_type` 的适配器未实现 `RaCompressAnalysisInterface` 且 Q/K 投影层命名与默认模式不匹配。

模型是否在官方预验证列表中请参考[《大模型支持矩阵》](../../model/README.md)；如需快速了解 CLI 基础用法可参阅[《一键量化完整指南》](../../../user_guide/usage_one_click_quantization.md)。

## 2. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型权重目录 | 模型下载或本地路径 | HuggingFace 格式，含 `config.json` 及 `*.safetensors` 分片 | 可被目标 Transformers 版本正常加载 |
| 输入 | 模型适配器 | 用户适配代码，通过 `--model_type` 调用 | 实现 `PipelineInterface` 及对应的 `RaCompressAnalysisInterface` 专用接口 | 能被调度器（Runner）与处理器（Processor）正常驱动 |
| 输入 | 校准数据集 | 工具内置 `calib_dummy.jsonl`（本算法须显式指定） | JSONL 格式文本 Prompt，推荐 50 条；经 tokenizer 后严格对齐到 2500×4 段边界 | 命令行已通过 `--calibration_dataset calib_dummy.jsonl` 指定且可被加载 |
| 输入 | 量化配置文件 | 本地 YAML 文件 | 符合 `modelslim_v1` 协议规范 | 通过模式校验（Schema Validation） |
| 交付件 | `head.pt` | `--save_path` 指定路径 | dict 序列化 `.pt` 文件，含 `prefix_matching` 与 `copying` 字段 | 文件存在且可被 `torch.load` 读取，可被后续 KV cache 压缩流程消费 |

## 3. 流程总览

RA Compress 的整体使用流程如下：

```mermaid
flowchart LR
    A[适配模型<br/>流水线/算法接口] --> B[确认投影层命名<br/>与校准数据]
    B --> C[执行<br/>analyze attn_head 命令]
    C --> D[Q@K^T 段间偏移<br/>注意力聚合]
    D --> E[按 ratio<br/>选 induction/echo heads]
    E --> F[输出 head.pt<br/>并衔接压缩流程]
```

各阶段的关键细节如下：

- **适配模型流水线/算法接口**：适配器需实现 `PipelineInterface` 基础流水线接口；当 Q/K 投影层命名与默认模式不匹配时，需实现 `RaCompressAnalysisInterface` 提供投影层命名模式。
- **确认投影层命名与校准数据**：默认匹配 `q_proj`、`k_proj`、`qkv_proj`；校准数据须显式指定内置合成校准集 `calib_dummy.jsonl`，使用其他校准集不保证筛选效果。
- **执行 analyze attn_head 命令**：以 `--metrics ra_compress` 启动 KV 注意力头粒度分析。
- **Q@K^T 段间偏移注意力聚合**：在段边界处计算偏移注意力，度量各 KV head 的跨段信息传递能力。
- **按 ratio 选 induction/echo heads**：归纳头（prefix matching）与回声头（copying matching）按比例形成入选索引。
- **输出 head.pt 并衔接压缩流程**：`head.pt` 送入 RA Compress KV cache 压缩量化流程，生成压缩后的模型。

## 4. 操作步骤

### 步骤 1：适配模型流水线与算法接口

**目标**：量化工具不能直接操作任意结构的模型，它要求每个模型先套一层“适配器”，把模型的各种操作翻译成框架能统一调用的标准方法。本步骤即确认并完成这层适配器与 RA Compress 专用接口。**模型适配必须先完成，才能执行 RA Compress 注意力头分析。**

**操作**：模型适配代码需实现以下接口，相关接口均由 `msmodelslim.model.interface_hub` 汇总导出：

1. **`PipelineInterface`**（`ModelSlimPipelineInterfaceV1`）：基础流水线接口，负责模型加载（`init_model`）、数据预处理（`handle_dataset`）、逐层遍历（`generate_model_visit`）、逐层前向（`generate_model_forward`）等。
2. **`RaCompressAnalysisInterface`**（位于 `msmodelslim.processor.analysis.unary_operator.metrics.ra_compress.interface`）：RA Compress 分析专用接口（按需实现）。

```python
from abc import ABC, abstractmethod
from typing import Dict

class RaCompressAnalysisInterface(ABC):
    @abstractmethod
    def get_ra_compress_proj_patterns(self) -> Dict[str, str]:
        """返回 Q/K/QKV 投影层名称模式字典。"""
        ...
```

- **`get_ra_compress_proj_patterns`**：返回 `{"q": "...", "k": "...", "qkv": "..."}` 字典，描述 Q 投影层、K 投影层与 QKV 融合投影层的名称模式；未使用的模式可以留空字符串（如无 QKV 融合时 `qkv=""`）。
- 模型 Q/K 投影层命名为 `q_proj` / `k_proj` / `qkv_proj` 时可省略该接口，工具按默认模式匹配。

对于基于 HuggingFace Transformers 实现的标准开源 LLM，建议通过组合继承快速构建适配器：

```python
from msmodelslim.model.interface_hub import (
    IModel,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1 as PipelineInterface,
    # ---- RA Compress 算法适配接口（本步骤重点）----
    RACompressAnalysisInterface,  # 算法适配：get_ra_compress_proj_patterns 返回模型 Q/K/QKV 投影层名称模式
)

class MyModelAdapter(TransformersModel, ModelInfoInterface, PipelineInterface,
                     RACompressAnalysisInterface):
    def get_ra_compress_proj_patterns(self):
        # 返回模型实际的 Q/K/QKV 投影层名称模式
        return {"q": "q_proj", "k": "k_proj", "qkv": "qkv_proj"}
```

之后分析命令中通过 `--model_type MyModelAdapter` 引用该适配器。

如需开发新模型适配代码，请参考[《LLM 模型接入量化流程指南》](../../ptq/llm/integration_guide_large_language_model_quantization.md)。

**输出**：已在框架中注册可用的模型适配器（通过 `--model_type` 调用）。

### 步骤 2：建立推荐配置

**操作**：

下面参数用于建立**第一版可比较基线**。

- `--metrics`：`ra_compress`；
- `--calibration_dataset`：**必须**显式指定为内置合成校准集 `calib_dummy.jsonl`（`attn_head` 子命令不再为此场景设置默认值），使用其他校准集不保证筛选效果；
- top heads 选取比例（ratio）：使用实现内部默认值（induction head `0.14`、echo head `0.01`），CLI 不提供 `--ratio` 参数；
- `--save_path`：指定 `head.pt` 与结果文件保存目录。

**输出**：一组可复现的注意力头分析基线参数，用于生成第一版 induction/echo heads 排序。

### 步骤 3：选择并调整参数

**操作**：

每轮只调一个变量，并保持同一校准数据与模型版本。

| 配置项 | 含义（原理） | 推荐配置 | 选择与调整建议 |
| --- | --- | --- | --- |
| `attn_head` | KV 注意力头粒度分析子命令。 | 固定。 | 不调。 |
| `--metrics` | 指定分析算法，取值为 `ra_compress` 时使用本算法。 | 固定 `ra_compress`。 | 不调。 |
| `--calibration_dataset` | 用于段边界对齐的校准数据；本算法须使用内置合成校准集。 | **必须** `calib_dummy.jsonl`。 | 不建议更换；使用其他校准集不保证筛选效果。 |
| `--save_path` | `head.pt` 与结果文件保存目录。 | 按需指定。 | 保证目录存在且有写权限。 |
| `--trust_remote_code` | 是否信任模型远程代码。 | 模型含自定义代码时设 `true`。 | 按模型要求设置。 |

**输出**：一份参数含义和调整方向明确的分析配置。

### 步骤 4：编写量化配置并执行命令

**目标**：整合上述步骤生成完整命令，并通过 CLI 启动注意力头分析流程。

#### 完整示例：RA Compress 注意力头分析

##### 执行命令（单卡分析）

```bash
msmodelslim analyze attn_head \
  --model_path <浮点模型目录> \
  --model_type <模型适配器名称> \
  --metrics ra_compress \
  --calibration_dataset calib_dummy.jsonl \
  --trust_remote_code true \
  --save_path ./head_result \
  --device npu \
  --device_id 0
```

参数说明：

| 参数 | 说明 |
| --- | --- |
| `attn_head` | KV 注意力头粒度分析 |
| `--metrics` | 指定分析算法，取值为 `ra_compress` 时使用本算法 |
| `--calibration_dataset` | **必须**显式指定为 `calib_dummy.jsonl`；使用其他校准集不保证筛选效果 |
| `--save_path` | `head.pt` 与结果文件保存目录 |

**输出**：命令行打印入选的 induction heads / echo heads 列表；`--save_path` 目录下生成 `head.pt`（含 `prefix_matching` 与 `copying` 字段），供后续 RA Compress KV cache 压缩量化流程使用。

## 5. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| RA Compress 注意力头分析算法 | 说明该算法的定义、核心原理、关键性质、适用场景与限制。 | [《RA Compress 量化术语百科词条》](./term_ra_compress.md) |
| Induction Head | 归纳头，在段边界处体现 prefix matching 行为的关键 KV head。 | [《RA Compress 量化术语百科词条》](./term_ra_compress.md) |
| Echo Head | 回声头，在段边界处体现 copying matching 行为的关键 KV head。 | [《RA Compress 量化术语百科词条》](./term_ra_compress.md) |

## 6. 相关文档

| 接口或文档 | 简述 | 链接 |
| --- | --- | --- |
| `PipelineInterface` | 模型流水线适配接口（数据预处理、模型加载、模块遍历）。 | [《LLM 量化使用指南·步骤 1》](../../ptq/llm/usage_large_language_model_quantization.md) |
| `RaCompressAnalysisInterface` | RA Compress 分析模型适配接口，由 `msmodelslim.model.interface_hub` 汇总导出。 | [接口汇总模块](../../../../../msmodelslim/model/interface_hub.py) |
| unary_analysis 配置说明 | 字段类型、默认值、合法取值与完整配置约束。 | [《unary_analysis 配置说明》](../../../api_reference/config/processor/unary_analysis.md) |
| modelslim_v1 配置说明 | 需要继续探索 runner、prior、save、dataset 等任务级高级配置时查阅。 | [《modelslim_v1 配置说明》](../../../api_reference/config/task/modelslim_v1.md) |
| 权重量化使用指南 | 用户指南：量化命令参数与完整使用说明。 | [《权重量化使用指南》](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/zh/user_guide/usage_weight_quantization.md) |
