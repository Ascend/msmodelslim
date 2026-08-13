# Quantile 使用指南

## 1. 适用范围

本流程适用于在 msModelSlim 中使用 Quantile 敏感层分析算法。Quantile 作为 `msmodelslim analyze linear` 的 metrics 指标，用于线性层粒度的敏感度排序，对离群点相对稳健。

适用角色：算法工程师、模型部署工程师

适用场景：

- 激活分布尾部较重、希望降低离群点对单层分数主导影响的场景。
- 需要更稳健的线性层敏感度排序的场景。

不适用场景：

- `layer`/`attn` 范围分析（Quantile 仅用于 `linear` 范围）。

## 2. 流程关系与前置条件

**上级流程**：模型适配与验证通过后，定稿量化 YAML 前的敏感层分析阶段。

**前置条件**：

- 已安装兼容版本的 msModelSlim 工具（详见《[msModelSlim 工具安装指南](../../../install_guide/install_guide.md)》）。
- 已确认目标 `model_type` 在支持矩阵范围内。
- 已准备好校准数据集。

**后续操作**：根据分析结果进行层回退与 YAML 调参，进入量化流程。

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型权重目录 | 模型下载或本地路径 | HuggingFace 格式，含 `config.json` 及 `*.safetensors` | 可被目标 Transformers 版本加载 |
| 输入 | 校准数据集 | 工具默认或用户指定 | JSONL 格式，每条含文本 prompt | 可被工具加载并完成前向推理 |
| 交付件 | 敏感层分析结果 | 命令行输出 | 线性层粒度的 score 排序结果 | 结果可读且包含目标层 |

## 4. 流程总览

```mermaid
flowchart LR
    A[准备模型与校准集] --> B[执行 analyze 命令]
    B --> C[统计分位数]
    C --> D[输出层敏感度排序]
```

## 5. 操作步骤

### 步骤 1：执行敏感层分析命令

**目标**：使用 Quantile 指标完成线性层敏感度分析。

**操作**：

```bash
msmodelslim analyze linear \
    --model_type Qwen2.5-7B-Instruct \
    --model_path ${model_path} \
    --metrics quantile \
    --calib_dataset ${calib_dataset} \
    --pattern "*.down_proj*" "*.o_proj*" \
    --topk 15 \
    --device npu
```

参数说明：

| 参数 | 说明 |
| --- | --- |
| `linear` | 线性层敏感度分析 |
| `--metrics` | 指定分析算法，取值为 `quantile` 时使用本算法 |
| `--pattern` | 层名通配符，过滤待分析线性层 |
| `--topk` | 输出的 topk 敏感层数量 |

完整参数见《[敏感层分析工具使用指南参数说明](../../../user_guide/usage_sensitive_layer_wise_analysis.md#命令行预览)》。

**输出**：线性层粒度的敏感度排序结果。

### 步骤 2：解读结果并指导调参

**目标**：根据敏感度排序结果辅助回退与 YAML 调参。

**操作**：

1. 查看排序结果，识别 score 较高的敏感层。
2. 结合业务精度要求确定回退阈值。
3. 在量化 YAML 中对敏感层配置回退或降低量化强度。

**输出**：回退与调参方案。

## 6. 验收条件

- 分析命令执行成功并输出目标层的敏感度排序。
- 分析结果可用于指导量化配置调整。

## 7. 异常处置

- **结果为空**：检查 `--pattern` 通配符是否与模型层名匹配。
- **模型不支持**：确认 `model_type` 在支持矩阵范围内。

## 8. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| Quantile | 基于分位数与 IQR 构造 score 的敏感层分析算法 | [Quantile 词条](./term_quantile.md) |
| 四分位距（IQR） | 上四分位数与下四分位数之差 | [Quantile 词条](./term_quantile.md) |
| linear 范围分析 | 对线性层粒度的敏感度分析 | [敏感层分析使用指南](../../../user_guide/usage_sensitive_layer_wise_analysis.md) |

## 9. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| `msmodelslim analyze linear` | 线性层敏感度分析命令，`--metrics quantile` 启用本算法 | [Quantile 词条](./term_quantile.md) |
| `--metrics` | 指定分析算法的参数 | [敏感层分析使用指南](../../../user_guide/usage_sensitive_layer_wise_analysis.md) |
