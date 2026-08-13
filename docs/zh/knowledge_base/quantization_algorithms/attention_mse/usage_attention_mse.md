# Attention MSE 使用指南

## 1. 适用范围

本流程适用于在 msModelSlim 中使用 Attention MSE（`mse`）敏感层分析算法。Attention MSE 作为 `msmodelslim analyze attn` 的 metrics 指标，用于注意力模块粒度的敏感度排序。

适用角色：算法工程师、模型部署工程师

适用场景：

- 需要对 Attention 结构做权重量化或评估其敏感度的场景。
- 需要注意力模块粒度敏感度排序以辅助回退决策的场景。

不适用场景：

- 目标 `model_type` 的适配器未实现 `AttentionMSEAnalysisInterface`（分析会报错）。

## 2. 流程关系与前置条件

**上级流程**：模型适配与验证通过后，定稿量化 YAML 前的敏感层分析阶段。

**前置条件**：

- 已安装兼容版本的 msModelSlim 工具（详见《[msModelSlim 工具安装指南](../../../install_guide/install_guide.md)》）。
- 已确认目标 `model_type` 的适配器实现了 `AttentionMSEAnalysisInterface`（当前支持 DeepSeek 系列）。
- 已准备好校准数据集。

**后续操作**：根据分析结果进行 attention 层回退与 YAML 调参，进入量化流程。

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型权重目录 | 模型下载或本地路径 | HuggingFace 格式，含 `config.json` 及 `*.safetensors` | 可被目标 Transformers 版本加载 |
| 输入 | 校准数据集 | 工具默认或用户指定 | JSONL 格式，每条含文本 prompt | 可被工具加载并完成前向推理 |
| 交付件 | 敏感层分析结果 | 命令行输出 | attention 模块粒度的 score 排序结果 | 结果可读且包含目标层 |

## 4. 流程总览

```mermaid
flowchart LR
    A[准备模型与校准集] --> B[执行 analyze 命令]
    B --> C[浮点/量化双路前向]
    C --> D[输出 attn 敏感度排序]
```

## 5. 操作步骤

### 步骤 1：确认适配器实现分析接口

**目标**：确认目标 `model_type` 的适配器支持 `attn` 范围分析。

**操作**：

1. 确认适配器实现了 `AttentionMSEAnalysisInterface`。
2. 确认 `get_attention_module_cls()` 返回待挂 hook 的 attention 模块类名字符串。
3. 确认 `get_attention_output_extractor()` 能从 `forward` 返回值中取出用于计算 MSE 的张量。

各方法的具体约定，请参阅《[Attention MSE 词条](./term_attention_mse.md)》。

**输出**：适配器接口确认。

### 步骤 2：执行敏感层分析命令

**目标**：使用 `mse` 指标完成 attention 模块敏感度分析。

**操作**：

```bash
msmodelslim analyze attn \
    --model_type DeepSeek-V3 \
    --model_path ${model_path} \
    --metrics mse \
    --calib_dataset ${calib_dataset} \
    --topk 15 \
    --device npu
```

参数说明：

| 参数 | 说明 |
| --- | --- |
| `attn` | 注意力结构敏感度分析 |
| `--metrics` | 指定分析算法，取值为 `mse` 时使用本算法 |
| `--topk` | 输出的 topk 敏感层数量 |

完整参数见《[敏感层分析工具使用指南参数说明](../../../user_guide/usage_sensitive_layer_wise_analysis.md#命令行预览)》。

**输出**：attention 模块粒度的敏感度排序结果。

### 步骤 3：解读结果并指导调参

**目标**：根据敏感度排序结果辅助回退与 YAML 调参。

**操作**：

1. 查看排序结果，识别 MSE 较高的 attention 模块。
2. 结合业务精度要求确定回退阈值。
3. 在量化 YAML 中对敏感 attention 层配置回退或降低量化强度。

**输出**：回退与调参方案。

## 6. 验收条件

- 分析命令执行成功并输出目标层的敏感度排序。
- 分析结果可用于指导量化配置调整。

## 7. 异常处置

- **报错提示未实现 `AttentionMSEAnalysisInterface`**：当前 `model_type` 的适配器未接入该分析路径，请换用支持列表中的模型类型（如 DeepSeek 系列），或在适配器中按接口实现 hook 类名与输出提取逻辑。

## 8. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| Attention MSE | 基于浮点/量化双路输出的 MSE 分析算法 | [Attention MSE 词条](./term_attention_mse.md) |
| attn 范围分析 | 对注意力模块粒度的敏感度分析 | [敏感层分析使用指南](../../../user_guide/usage_sensitive_layer_wise_analysis.md) |
| AttentionMSEAnalysisInterface | 模型适配器需实现的 attn 分析接口 | [Attention MSE 词条](./term_attention_mse.md) |

## 9. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| `msmodelslim analyze attn` | 注意力结构敏感度分析命令，`--metrics mse` 启用本算法 | [Attention MSE 词条](./term_attention_mse.md) |
| `AttentionMSEAnalysisInterface` | 模型适配器需实现的接口 | [Attention MSE 词条](./term_attention_mse.md) |
