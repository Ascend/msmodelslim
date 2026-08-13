# 层级 MSE 使用指南

## 1. 适用范围

本流程适用于在 msModelSlim 中使用层级 MSE（`mse_layer_wise`）敏感层分析算法。层级 MSE 作为 `msmodelslim analyze layer` 的 metrics 指标，用于 Decoder 块粒度的敏感度排序，辅助整层或整块回退决策。

适用角色：算法工程师、模型部署工程师

适用场景：

- 希望通过 Decoder 块内子模块输出角度比较各层敏感度、辅助整层或整块回退决策的场景。
- 需要块级量化重构误差评估的场景。

不适用场景：

- `linear`/`attn` 范围分析（层级 MSE 仅用于 `layer` 范围）。

## 2. 流程关系与前置条件

**上级流程**：模型适配与验证通过后，定稿量化 YAML 前的敏感层分析阶段。

**前置条件**：

- 已安装兼容版本的 msModelSlim 工具（详见《[msModelSlim 工具安装指南](../../../install_guide/install_guide.md)》）。
- 已确认目标 `model_type` 在支持矩阵范围内。
- 已准备好校准数据集。
- 已确定 `quant_modules` 通配符配置（参与量化对比的子模块集合）。

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
    B --> C[块内浮点/量化对比]
    C --> D[输出 Decoder 块排序]
```

## 5. 操作步骤

### 步骤 1：执行敏感层分析命令

**目标**：使用 `mse_layer_wise` 指标完成 Decoder 块级敏感度分析。

**操作**：

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

参数说明：

| 参数 | 说明 |
| --- | --- |
| `layer` | Decoder 块级敏感度分析 |
| `--metrics` | 指定分析算法，取值为 `mse_layer_wise` 时使用本算法 |
| `--quant_modules` | 通配符列表，指定参与量化对比的模块范围 |
| `--topk` | 输出的 topk 敏感层数量 |

完整参数见《[敏感层分析工具使用指南参数说明](../../../user_guide/usage_sensitive_layer_wise_analysis.md#命令行预览)》。

**输出**：Decoder 块粒度的敏感度排序结果。

### 步骤 2：解读结果并指导调参

**目标**：根据敏感度排序结果辅助整层/整块回退与 YAML 调参。

**操作**：

1. 查看排序结果，识别 score 较高的 Decoder 块。
2. 结合业务精度要求确定回退阈值。
3. 在量化 YAML 中对敏感块配置整层或整块回退。

**输出**：回退与调参方案。

## 6. 验收条件

- 分析命令执行成功并输出目标层的敏感度排序。
- 分析结果可用于指导量化配置调整。

## 7. 异常处置

- **换 `quant_modules` 后层排序变了**：属于预期，调整 `quant_modules` 后参与量化对比的子模块集合发生变化，块内 MSE 聚合结果随之改变。请按目标量化方案固定一版配置后再解读；同一命令仍不稳定时再排查校准顺序与随机性。
- **结果为空**：检查 `--quant_modules` 通配符是否与模型层名匹配。

## 8. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| 层级 MSE | 块内聚合量化重构 MSE 的 layer 分析算法 | [层级 MSE 词条](./term_mse_layer_wise.md) |
| quant_modules | 参与量化对比的子模块通配符配置 | [层级 MSE 词条](./term_mse_layer_wise.md) |
| layer 范围分析 | 对 Decoder 块粒度的敏感度分析 | [敏感层分析使用指南](../../../user_guide/usage_sensitive_layer_wise_analysis.md) |

## 9. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| `msmodelslim analyze layer` | Decoder 块级敏感度分析命令，`--metrics mse_layer_wise` 启用本算法 | [层级 MSE 词条](./term_mse_layer_wise.md) |
| `--quant_modules` | 指定参与量化对比模块范围的参数 | [敏感层分析使用指南](../../../user_guide/usage_sensitive_layer_wise_analysis.md) |
