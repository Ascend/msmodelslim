# 模型级 MSE 分析配置使用指南

## 1. 适用范围

本指南面向需要使用[模型级 MSE 敏感层分析算法](./term_mse_model_wise.md) 的用户。模型级 MSE（`mse_model_wise`）作为 `msmodelslim analyze layer` 的 metrics 指标，从模型最终输出视角评估各层量化敏感度，辅助整层或整块回退决策。

适用场景：

- W8/W4 等低比特量化前，需要从端到端输出误差视角定位敏感层，形成回退或混合精度候选集合；
- layer-wise 等局部指标排序不稳定，需要以模型最终输出 MSE 复核敏感层结论。

模型是否在官方预验证列表中请参考[《大模型支持矩阵》](../../model/README.md)；如需快速了解 CLI 基础用法可参阅[《一键量化完整指南》](../../../user_guide/usage_one_click_quantization.md)。

## 2. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型权重目录 | 模型下载或本地路径 | HuggingFace 格式，含 `config.json` 及 `*.safetensors` 分片 | 可被目标 Transformers 版本正常加载 |
| 输入 | 模型适配器 | 用户适配代码，通过 `--model_type` 调用 | 实现 `PipelineInterface` 及对应的 `MSEModelWiseAnalysisInterface` 专用接口 | 能被调度器（Runner）与处理器（Processor）正常驱动 |
| 输入 | 校准数据集 | 工具内置 `lab_calib/` 或用户自定义路径 | JSONL 或 JSON 格式文本 Prompt，推荐 50 条 | 可被适配器 `handle_dataset` 成功编码为前向张量 |
| 交付件 | 敏感层分析结果 | `--save_path` 指定路径（YAML），未指定时仅打印到控制台 | 各层量化扰动对应的模型级 MSE score 与 Top-K 排序 | 结果稳定可复现，可作为回退或混合精度候选输入 |

## 3. 流程总览

模型级 MSE 的整体使用流程如下：

```mermaid
flowchart LR
    A[适配模型<br/>流水线/算法接口] --> B[固定分析范围<br/>与校准数据]
    B --> C[逐层注入量化扰动]
    C --> D[比较模型<br/>最终输出 MSE]
    D --> E[按层形成回退候选]
    E --> F[验证精度<br/>并收敛候选]
```

各阶段的关键细节如下：

- **适配模型流水线/算法接口**：适配器需实现 `PipelineInterface` 基础流水线接口；可选实现 `MSEModelWiseAnalysisInterface` 提供块 I/O 的 hidden states 提取逻辑，是后续所有分析步骤的前提。
- **固定分析范围与校准数据**：分析范围与后续实际量化范围保持一致，校准数据（推荐 50 条）应覆盖真实输入分布；分析期间不更换数据与量化基线，否则排序无法归因。
- **逐层注入量化扰动**：Processor 对候选 Decoder 块逐层制造量化路径，复刻正式量化计划中的 `quant_modules` 范围。
- **比较模型最终输出 MSE**：从模型最终输出端与浮点基线计算 MSE，误差传播到输出后的影响被完整纳入评价。
- **按层形成回退候选**：按 `mse_model_wise` score 排序取 Top-K 候选，作为回退或混合精度的候选集合，而不是硬阈值。
- **验证精度并收敛候选**：候选须经实际量化精度或模型任务指标验证后才固化到最终策略。

## 4. 操作步骤

### 步骤 1：适配模型流水线与算法接口

**目标**：量化工具不能直接操作任意结构的模型，它要求每个模型先套一层“适配器”，把模型的各种操作翻译成框架能统一调用的标准方法。本步骤即确认并完成这层适配器与模型级 MSE 专用接口。**模型适配必须先完成，才能执行模型级 MSE 敏感层分析。**

**操作**：模型适配代码需实现以下接口，相关接口均由 `msmodelslim.model.interface_hub` 汇总导出：

1. **`PipelineInterface`**（`ModelSlimPipelineInterfaceV1`）：基础流水线接口，负责模型加载（`init_model`）、数据预处理（`handle_dataset`）、逐层遍历（`generate_model_visit`）、逐层前向（`generate_model_forward`）等。
2. **`MSEModelWiseAnalysisInterface`**（位于 `msmodelslim.processor.analysis.binary_operator_model_wise.metrics.mse_model_wise.interface`）：模型级 MSE 分析专用接口（可选）。

```python
from abc import ABC, abstractmethod
from typing import Any

import torch

class MSEModelWiseAnalysisInterface(ABC):
    @abstractmethod
    def extract_hidden_states(self, value: Any) -> torch.Tensor:
        """从 block 相关数据中提取用于 MSE 比较 / 层间链式传播的主 tensor。"""
        ...
```

- **`extract_hidden_states`**：从 block `forward` 返回值（Tensor、tuple、ModelOutput、dict 等）中提取主 tensor；层间链式传播时，Processor 将其返回值作为下一层 `forward` 的首个 positional arg。
- 未实现该接口时，Processor 使用 `DefaultMSEModelWiseBlockData` 处理常见 LLM block I/O；VLM 等默认逻辑不足的结构需在适配器中实现。

对于基于 HuggingFace Transformers 实现的标准开源 LLM，建议通过组合继承快速构建适配器：

```python
from msmodelslim.model.interface_hub import (
    IModel,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1 as PipelineInterface,
    # ---- MSE Model Wise 算法适配接口（本步骤重点）----
    MSEModelWiseAnalysisInterface,  # 算法适配：提取 block 输入/输出 hidden states，供块级 MSE 比较
)

class MyModelAdapter(TransformersModel, ModelInfoInterface, PipelineInterface,
                     MSEModelWiseAnalysisInterface):
    def extract_hidden_states(self, value):
        # 从 block 输出中提取用于 MSE 比较的主 tensor
        ...
```

之后分析命令中通过 `--model_type MyModelAdapter` 引用该适配器。

如需开发新模型适配代码，请参考[《LLM 模型接入量化流程指南》](../../ptq/llm/integration_guide_large_language_model_quantization.md)。

**输出**：已在框架中注册可用的模型适配器（通过 `--model_type` 调用）。

### 步骤 2：建立推荐配置

**操作**：

下面参数用于建立**第一版可比较基线**。

- 分析范围：`layer`；
- `--metrics`：`mse_model_wise`；
- `--top_k`：先用 `15`；
- `--quant_modules`：与实际量化计划保持一致，不确定时从 `"*"` 开始。

**输出**：一组可复现的敏感性分析基线参数，用于生成第一版候选排序。

### 步骤 3：选择并调整参数

**操作**：

每轮只调一个变量，并保持同一校准集和量化基线。

| 配置项 | 含义（原理） | 推荐配置 | 选择与调整建议 |
| --- | --- | --- | --- |
| 分析范围 `layer` | 逐层制造候选量化路径，再从模型最终输出端与浮点基线计算 MSE。 | 固定 `layer`。 | 更接近端到端影响，但执行/存储成本通常高于只比较 block 输出。 |
| `--metrics` | 指定本指南对应的分析指标 `mse_model_wise`，决定 score 的定义和排序含义。 | 固定 `mse_model_wise`。 | 高分说明该层量化扰动更容易传播到模型最终输出；与其他指标的数值尺度/排序不同，不应混用阈值。 |
| `--quant_modules` | 每个候选 Decoder block 内实际被量化的模块范围，须尽量复刻正式量化计划。 | 不确定时 `["*"]`；正式决策时对齐实际量化模块。 | 只有确认某类模块不会量化时才收窄；不同范围下的排名不要直接比较。 |
| `--calibration_dataset` | 用于前向收集输出的校准数据，JSON/JSONL 格式（LLM 文本样本文件，推荐 50 条）。 | 与后续正式量化相同或同分布的校准集。 | 排序在不同数据上明显变化时，先检查校准集是否覆盖真实长度与主题。 |
| `--top_k` | 控制最终展示/导出的高分候选数量，默认 `15`；不改变 score 计算。 | 从 `15` 开始。 | 需要更多回退候选时增大，需要快速人工审查时减小；不要把 `top_k` 当作敏感度阈值。 |

**输出**：一份完成单变量调整的分析参数方案，关键字段均有明确的选择依据和调整方向。

### 步骤 4：执行分析命令

**目标**：整合上述步骤，通过 CLI 启动模型级 MSE 敏感层分析流程。

#### 完整示例：模型级 MSE 敏感层分析

##### 执行命令（单卡分析）

```bash
msmodelslim analyze layer \
  --model_path <浮点模型目录> \
  --model_type <模型适配器名称> \
  --metrics mse_model_wise \
  --top_k 15 \
  --quant_modules "*" \
  --calibration_dataset ./mix_calib.jsonl \
  --device npu \
  --device_id 0
```

多卡分析时追加 `--device_id 0 1 2 3`；需要留存结果文件时追加 `--save_path <结果目录>`（YAML 格式），否则结果仅打印到控制台。

**输出**：命令行输出各层 `mse_model_wise` score 与 Top-K 排序，作为后续回退或混合精度的候选输入。

## 5. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| 模型级 MSE 敏感层分析算法 | 说明该算法的定义、核心原理、关键性质、适用场景与限制。 | [《模型级 MSE 敏感层分析算法 量化术语百科词条》](./term_mse_model_wise.md) |

## 6. 相关文档

| 接口或文档 | 简述 | 链接 |
| --- | --- | --- |
| `PipelineInterface` | 模型流水线适配接口（数据预处理、模型加载、模块遍历）。 | [《LLM 量化使用指南·步骤 1》](../../ptq/llm/usage_large_language_model_quantization.md) |
| `MSEModelWiseAnalysisInterface` | 模型级 MSE 分析模型适配接口，由 `msmodelslim.model.interface_hub` 汇总导出。 | [接口汇总模块](../../../../../msmodelslim/model/interface_hub.py) |
| binary_operator_model_wise 配置说明 | 字段类型、默认值、合法取值与完整配置约束。 | [《binary_operator_model_wise 配置说明》](../../../api_reference/config/processor/binary_operator_model_wise.md) |
| modelslim_v1 配置说明 | 需要继续探索 runner、prior、save、dataset 等任务级高级配置时查阅。 | [《modelslim_v1 配置说明》](../../../api_reference/config/quant/modelslim_v1.md) |
| 权重量化使用指南 | 用户指南：量化命令参数与完整使用说明。 | [《权重量化使用指南》](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/docs/zh/user_guide/usage_weight_quantization.md) |
