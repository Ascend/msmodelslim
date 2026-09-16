# Attention MSE 分析配置流程指南

## 1. 适用范围

本指南面向需要使用[Attention MSE 敏感层分析算法](./term_attention_mse.md)的用户。Attention MSE（`mse`）作为 `msmodelslim analyze attn` 的 metrics 指标，从 Attention 模块输出视角评估各 Attention 模块量化敏感度，辅助 Attention 结构的回退或保护决策。

适用场景：

- Attention 结构参与权重量化前，需要定位输出漂移更大的 Attention 模块，形成回退或重点保护候选集合；
- 需要以 Attention 子系统实际输出误差复核权重类指标（如 `std`）的敏感度结论。

模型是否在官方预验证列表中请参考[《大模型支持矩阵》](../../model/README.md)；如需快速了解 CLI 基础用法可参阅[《一键量化完整指南》](../../../user_guide/usage_one_click_quantization.md)。

## 2. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型权重目录 | 模型下载或本地路径 | HuggingFace 格式，含 `config.json` 及 `*.safetensors` 分片 | 可被目标 Transformers 版本正常加载 |
| 输入 | 模型适配器 | 用户适配代码，通过 `--model_type` 调用 | 实现 `PipelineInterface` 及 `AttentionAnalysisInterface` 专用接口 | 能被调度器（Runner）与处理器（Processor）正常驱动 |
| 输入 | 校准数据集 | 工具内置 `lab_calib/` 或用户自定义路径 | JSONL 或 JSON 格式文本 Prompt，推荐 50 条 | 可被适配器 `handle_dataset` 成功编码为前向张量 |
| 交付件 | Attention 敏感度分析结果 | `--save_path` 指定路径（YAML），未指定时仅打印到控制台 | 各 Attention 模块量化扰动对应的 MSE score 与 Top-K 排序 | 结果稳定可复现，可作为 Attention 回退或保护候选输入 |

## 3. 流程总览

Attention MSE 的整体使用流程如下：

```mermaid
flowchart LR
    A[适配模型<br/>流水线/算法接口] --> B[固定校准数据<br/>与 Attention 范围]
    B --> C[逐模块执行<br/>浮点/量化双路前向]
    C --> D[采集 Attention 输出<br/>并计算 MSE]
    D --> E[按模块形成<br/>敏感度排序]
    E --> F[验证精度<br/>并收敛候选]
```

各阶段的关键细节如下：

- **适配模型流水线/算法接口**：适配器需实现 `PipelineInterface` 基础流水线接口；实现 `AttentionAnalysisInterface` 提供 Attention 模块匹配与输出提取逻辑，是后续所有分析步骤的前提。
- **固定校准数据与 Attention 范围**：`attn` 范围默认覆盖模型中全部 Attention 模块；校准数据（推荐 50 条）应覆盖真实输入分布，分析期间不更换数据与量化基线，否则排序无法归因。
- **逐模块执行浮点/量化双路前向**：Processor 对每个 Attention 模块分别执行浮点路径与量化路径前向，其余条件保持一致，形成受控干预。
- **采集 Attention 输出并计算 MSE**：通过 hook 采集两路 Attention 输出，逐样本计算均方误差后取平均，Q/K/V 误差经点积与 softmax 非线性的放大或抑制被完整纳入评价。
- **按模块形成敏感度排序**：按 `mse` score 排序取 Top-K 候选，作为回退或重点保护的候选集合，而不是硬阈值。
- **验证精度并收敛候选**：候选须经实际量化精度或模型任务指标验证后才固化到最终策略。

## 4. 操作步骤

### 步骤 1：适配模型流水线与算法接口

**目标**：量化工具不能直接操作任意结构的模型，它要求每个模型先套一层“适配器”，把模型的各种操作翻译成框架能统一调用的标准方法。本步骤即确认并完成这层适配器与 Attention MSE 专用接口。**模型适配必须先完成，才能执行 Attention MSE 敏感度分析。**

**操作**：模型适配代码需实现以下接口：

1. **`PipelineInterface`**（`ModelSlimPipelineInterfaceV1`）：基础流水线接口，负责模型加载（`init_model`）、数据预处理（`handle_dataset`）、逐层遍历（`generate_model_visit`）、逐层前向（`generate_model_forward`）等。
2. **`AttentionAnalysisInterface`**（由 `msmodelslim.model.interface_hub` 汇总导出，实现位于 `msmodelslim.processor.analysis.binary_operator.metrics.attention_mse.interface`）：Attention MSE 分析专用接口。

```python
from abc import ABC, abstractmethod
from typing import Callable, Union

import torch

class AttentionAnalysisInterface(ABC):
    @abstractmethod
    def get_attention_module_cls(self) -> str:
        """返回用于匹配注意力层的模块类（即需要挂 hook 的 attention 子模块）的字符串表示。"""
        ...

    @abstractmethod
    def get_attention_output_extractor(self) -> Callable[[Union[tuple, torch.Tensor]], torch.Tensor]:
        """返回一个提取函数，用于从 attention 模块的 forward 输出中取出用于敏感度分析的张量部分。"""
        ...
```

- **`get_attention_module_cls`**：返回 Attention 模块的类名字符串。Processor 按 `module.__class__.__name__` 精确匹配并挂载输出 hook，例如 `"DeepseekV3Attention"`、`"MLA"`。
- **`get_attention_output_extractor`**：从 Attention 模块 `forward` 返回值（Tensor、tuple 等）中提取用于敏感度分析的主 tensor。例如 DeepSeek-V3 的 `forward` 返回 `(attn_output, ...)` 的 tuple，返回 `lambda x: x[0]`；GLM 系列直接返回输出张量时，返回 `lambda x: x`。

对于基于 HuggingFace Transformers 实现的标准开源 LLM，建议通过组合继承快速构建适配器：

```python
from msmodelslim.model.interface_hub import (
    IModel,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1 as PipelineInterface,
    # ---- Attention MSE 算法适配接口（本步骤重点）----
    AttentionAnalysisInterface,  # 算法适配：定位需挂 hook 的 attention 子模块并提取注意力输出
)

class MyModelAdapter(TransformersModel, ModelInfoInterface, PipelineInterface,
                     AttentionAnalysisInterface):
    def get_attention_module_cls(self) -> str:
        return "MyModelAttention"  # 需要挂 hook 的 attention 子模块类名

    def get_attention_output_extractor(self):
        return lambda x: x[0]  # 从 forward 输出中提取注意力输出张量
```

之后分析命令中通过 `--model_type MyModelAdapter` 引用该适配器。仓库内可参考的适配实现包括 `msmodelslim/model/deepseek_v3/model_adapter.py` 与 `msmodelslim/model/glm_5/model_adapter.py`。

如需开发新模型适配代码，请参考[《LLM 模型接入量化流程指南》](../../ptq/llm/integration_guide_large_language_model_quantization.md)。

**输出**：已在框架中注册可用的模型适配器（通过 `--model_type` 调用）。

### 步骤 2：建立推荐配置

**操作**：

下面参数用于建立**第一版可比较基线**。

- 分析范围：`attn`；
- `--metrics`：`mse`；
- `--topk`：先用 `15`；
- 校准数据：推荐 50 条，与后续正式量化相同或同分布。

**输出**：一组可复现的 Attention 敏感度分析基线参数，用于生成第一版候选排序。

### 步骤 3：选择并调整参数

**操作**：

每轮只调一个变量，并保持同一校准集和量化基线。

| 配置项 | 含义（原理） | 推荐配置 | 选择与调整建议 |
| --- | --- | --- | --- |
| 分析范围 `attn` | 固定为 `attn` 子命令，分析对象默认覆盖模型中全部 Attention 模块，不支持通过 `--patterns`/`--quant_modules` 收窄。 | 固定 `attn`。 | 需要注意力头粒度（induction/echo head）分析时使用 `attn_head` 的 `ra_compress` 指标，与本指标的排序含义不同，不应混用。 |
| `--metrics` | 指定本指南对应的分析指标 `mse`，决定 score 的定义和排序含义。 | 固定 `mse`。 | 高分说明该 Attention 模块的量化扰动更容易体现在自身输出端；与其他指标的数值尺度/排序不同，不应混用阈值。 |
| `--calib_dataset`（`--calibration_dataset`） | 用于双路前向采集 Attention 输出的校准数据，JSON/JSONL 格式文本样本文件，推荐 50 条，默认 `mix_calib.jsonl`。 | 与后续正式量化相同或同分布的校准集。 | 排序在不同数据上明显变化时，先检查校准集是否覆盖真实长度与主题。 |
| `--topk` | 控制最终展示/导出的高分候选数量，默认 `15`；不改变 score 计算。 | 从 `15` 开始。 | 需要更多回退候选时增大，需要快速人工审查时减小；不要把 `topk` 当作敏感度阈值。 |
| `--device_id` | 分析设备索引；传入多个索引（如 `0 1 2 3`）时自动启用分布式分析 Runner。 | 单卡不传；模型较大时多卡并行。 | 本指标支持分布式执行；多卡与单卡的排序应一致，差异明显时优先排查校准数据切分。 |

> 历史兼容：旧用法 `msmodelslim analyze --metrics attention_mse` 已废弃，CLI 会自动转换为 `attn --metrics mse` 并打印告警；新脚本应直接使用 `analyze attn` 写法。

**输出**：一份完成单变量调整的分析参数方案，关键字段均有明确的选择依据和调整方向。

### 步骤 4：执行分析命令

**目标**：整合上述步骤，通过 CLI 启动 Attention MSE 敏感度分析流程。

#### 完整示例：Attention MSE 敏感度分析

##### 执行命令（单卡分析）

```bash
msmodelslim analyze attn \
  --model_path <浮点模型目录> \
  --model_type <模型适配器名称> \
  --metrics mse \
  --topk 15 \
  --calib_dataset ./mix_calib.jsonl \
  --device npu
```

多卡分析时追加 `--device_id 0 1 2 3`；需要留存结果文件时追加 `--save_path <结果目录>`（YAML 格式），否则结果仅打印到控制台。

**输出**：命令行输出各 Attention 模块 `mse` score 与 Top-K 排序，保存于指定 `--save_path` 文件，作为后续 Attention 回退或重点保护的候选输入。

## 5. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| Attention MSE 敏感层分析算法 | 说明该算法的定义、核心原理、关键性质、适用场景与限制。 | [《Attention MSE 敏感层分析算法 量化术语百科词条》](./term_attention_mse.md) |

## 6. 相关文档

| 接口或文档 | 简述 | 链接 |
| --- | --- | --- |
| `PipelineInterface` | 模型流水线适配接口（数据预处理、模型加载、模块遍历）。 | [《LLM 量化使用指南·步骤 1》](../../ptq/llm/usage_large_language_model_quantization.md) |
| `AttentionAnalysisInterface` | Attention MSE 分析模型适配接口，由 `msmodelslim.model.interface_hub` 汇总导出。 | [接口汇总模块](../../../../../msmodelslim/model/interface_hub.py) |
| 敏感层分析使用指南 | 完整 CLI、输入输出和进阶流程。 | [《敏感层分析使用指南》](../../../user_guide/usage_sensitive_layer_analysis.md) |
| 权重量化使用指南 | 用户指南：量化命令参数与完整使用说明。 | [《权重量化使用指南》](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/zh/user_guide/usage_weight_quantization.md) |
