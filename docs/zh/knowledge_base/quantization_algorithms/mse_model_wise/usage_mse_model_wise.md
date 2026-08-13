# 模型级 MSE 使用指南

## 1. 适用范围

本流程适用于在 msModelSlim 中使用模型级 MSE（`mse_model_wise`）敏感层分析算法。模型级 MSE 作为 `msmodelslim analyze layer` 的 metrics 指标，用于以模型最终输出视角评估各层量化敏感度，辅助整层或整块回退决策。

适用角色：算法工程师、模型部署工程师

适用场景：

- 希望通过模型最终输出视角评估各层量化敏感度、辅助整层或整块回退决策的场景。
- 需要端到端量化影响评估的场景。

不适用场景：

- `linear`/`attn` 范围分析（模型级 MSE 仅用于 `layer` 范围）。
- 受链式前向对齐限制无法评估的架构（如 MTP 等特殊结构）。

## 2. 流程关系与前置条件

**上级流程**：模型适配与验证通过后，定稿量化 YAML 前的敏感层分析阶段。

**前置条件**：

- 已安装兼容版本的 msModelSlim 工具（详见《[msModelSlim 工具安装指南](../../../install_guide/install_guide.md)》）。
- 已确认目标 `model_type` 在支持矩阵范围内。
- 已准备好校准数据集。
- 已确认设备显存足够（链式前向会显著增加前向次数与中间缓存）。

**后续操作**：根据分析结果进行整层/整块回退与 YAML 调参，进入量化流程。

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型权重目录 | 模型下载或本地路径 | HuggingFace 格式，含 `config.json` 及 `*.safetensors` | 可被目标 Transformers 版本加载 |
| 输入 | 校准数据集 | 工具默认或用户指定 | JSONL 格式，每条含文本 prompt | 可被工具加载并完成前向推理 |
| 交付件 | 敏感层分析结果 | 命令行输出 | Decoder 块粒度的 score 排序结果 | 结果可读且包含目标层 |

## 4. 流程总览

```mermaid
flowchart LR
    A[准备模型与校准集] --> B[执行 analyze 命令]
    B --> C[链式前向逐层对比]
    C --> D[输出层敏感度排序]
```

## 5. 操作步骤

### 步骤 1：执行敏感层分析命令

**目标**：使用 `mse_model_wise` 指标完成模型最终输出视角的层敏感度分析。

**操作**：

```bash
msmodelslim analyze layer \
    --model_type Qwen3-32B \
    --model_path ${model_path} \
    --metrics mse_model_wise \
    --quant_modules "*mlp*" \
    --calib_dataset ${calib_dataset} \
    --topk 15 \
    --device npu
```

参数说明：

| 参数 | 说明 |
| --- | --- |
| `layer` | Decoder 块级敏感度分析 |
| `--metrics` | 指定分析算法，取值为 `mse_model_wise` 时使用本算法 |
| `--quant_modules` | 通配符列表，指定参与量化对比的模块范围 |
| `--topk` | 输出的 topk 敏感层数量 |

完整参数见《[敏感层分析工具使用指南参数说明](../../../user_guide/usage_sensitive_layer_wise_analysis.md#命令行预览)》。

**输出**：Decoder 块粒度的敏感度排序结果。

### 步骤 2：解读结果并指导调参

**目标**：根据敏感度排序结果辅助整层/整块回退与 YAML 调参。

**操作**：

1. 查看排序结果，识别对模型最终输出影响较大的层。
2. 结合业务精度要求确定回退阈值。
3. 在量化 YAML 中对敏感块配置整层或整块回退。

**输出**：回退与调参方案。

## 6. 验收条件

- 分析命令执行成功并输出目标层的敏感度排序。
- 分析结果可用于指导量化配置调整。

## 7. 异常处置

- **分析中途出现 warning 且后续层未出现在结果中**：多为链式前向在某层无法对齐输入输出；可查阅日志定位层号，对该层或特殊结构（如 MTP）优先回退或换用 `mse_layer_wise` 做块级评估。
- **显存溢出（OOM）**：控制校准批次数与序列长度，或使用显存更大的设备。

## 8. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| 模型级 MSE | 以模型最终输出衡量层敏感度的分析算法 | [模型级 MSE 词条](./term_mse_model_wise.md) |
| 链式前向 | 将上一层输出作为下一层输入的逐层前向 | [模型级 MSE 词条](./term_mse_model_wise.md) |
| layer 范围分析 | 对 Decoder 块粒度的敏感度分析 | [敏感层分析使用指南](../../../user_guide/usage_sensitive_layer_wise_analysis.md) |

## 9. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| `msmodelslim analyze layer` | Decoder 块级敏感度分析命令，`--metrics mse_model_wise` 启用本算法 | [模型级 MSE 词条](./term_mse_model_wise.md) |
| `--quant_modules` | 指定参与量化对比模块范围的参数 | [敏感层分析使用指南](../../../user_guide/usage_sensitive_layer_wise_analysis.md) |
