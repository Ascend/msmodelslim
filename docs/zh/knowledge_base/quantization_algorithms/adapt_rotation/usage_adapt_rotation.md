# Adapt Rotation 使用指南

## 1. 适用范围

本流程适用于在 msModelSlim 中使用 Adapt Rotation（自适应旋转优化）离群值抑制算法。Adapt Rotation 作为 `type: "adapt_rotation"` 处理器，在 [QuaRot](../quarot/term_quarot.md) 基础上通过数据驱动优化旋转矩阵，用于提升低比特（如 W4A4）量化精度。

适用角色：算法工程师、模型部署工程师

适用场景：

- 需要对激活离群值做更精细抑制、进一步提升低比特量化精度的场景。
- 需要以两阶段流程（Stage1 优化 / Stage2 应用）完成旋转配置的场景。

不适用场景：

- 目标模型未实现 `AdaptRotationInterface`（无法收集激活或应用优化旋转）。

## 2. 流程关系与前置条件

**上级流程**：模型适配与验证通过后、配置量化 YAML 阶段，作为离群值抑制前置处理器集成。

**前置条件**：

- 已安装兼容版本的 msModelSlim 工具（详见《[msModelSlim 工具安装指南](../../../install_guide/install_guide.md)》）。
- 目标 `model_type` 的适配器已实现 `AdaptRotationInterface`（含 `get_hidden_dim()`）。
- 已准备好 Stage1 所需的校准数据集。

**后续操作**：Stage2 应用优化旋转后，继续执行下游量化（如 `autoround_quant`）、保存与部署。

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型权重目录 | 模型下载或本地路径 | HuggingFace 格式，含 `config.json` 及 `*.safetensors` | 可被目标 Transformers 版本加载 |
| 输入 | 校准数据集 | 工具默认或用户指定 | JSONL 格式，每条含文本 prompt | 可被工具加载并完成前向推理 |
| 交付件 | 优化后的旋转矩阵 | 上下文机制（Context） | Stage1 写入 `ctx["adapt_rotation"].state["adapted_matrix"]` | 可被 Stage2 读取并覆盖默认旋转 |
| 交付件 | 应用旋转后的模型 | 量化流程输出 | 完成层融合与旋转的模型 | 可继续执行下游量化 |

## 4. 流程总览

```mermaid
flowchart LR
    A[准备模型与校准集] --> B[Stage1 收集激活并优化]
    B --> C[写入 Context 传递旋转矩阵]
    C --> D[Stage2 应用优化旋转]
    D --> E[下游量化与保存]
```

## 5. 操作步骤

### 步骤 1：确认适配器实现分析接口

**目标**：确认目标 `model_type` 的适配器支持 Adapt Rotation 两阶段流程。

**操作**：

1. 确认适配器实现了 `AdaptRotationInterface`（继承 `QuaRotInterface` 并实现 `get_hidden_dim()`）。
2. 若需要在线旋转，确认适配器实现了 `LAOSOnlineRotationInterface`。
3. 确认 Stage1 在 `ContextManager` 下运行，以便将 `adapted_matrix` 传递给 Stage2。

各接口的具体约定，请参阅《[Adapt Rotation 词条](./term_adapt_rotation.md)》。

**输出**：适配器接口确认。

### 步骤 2：配置并执行 Stage1 旋转优化

**目标**：基于校准数据优化旋转矩阵。

**操作**：

1. 在 YAML 的 `spec.prior` 中配置 `type: "adapt_rotation"`、`stage: 1` 及该阶段的 `dataset`。
2. 按需设置 `layer_type`、`steps`、`quant_dtype`、`block_size`、`max_samples` 等参数。
3. 确保 Stage1 的 `quant_dtype` 与下游量化的 `act.dtype` 一致（w4a4 用 `int4`，w8a8 用 `int8`）。

```yaml
spec:
  prior:
    - process:
        - type: "adapt_rotation"
          stage: 1
          layer_type: ["up_proj"]
          steps: 20
          quant_dtype: "int4"
          block_size: -1
          max_samples: 2048
      dataset: boolq.jsonl
```

**输出**：优化后的旋转矩阵写入 Context。

### 步骤 3：配置并执行 Stage2 应用旋转

**目标**：将优化后的旋转矩阵应用到模型并继续量化流程。

**操作**：

1. 在 YAML 的 `spec.process` 中配置 `type: "adapt_rotation"`、`stage: 2`。
2. 按需设置 `online`、`block_size`、`max_tp_size` 等参数。
3. 确认 Stage1 与 Stage2 在同一量化流程中按顺序执行。

```yaml
  process:
    - type: "adapt_rotation"
      stage: 2
      online: False
      block_size: -1
      max_tp_size: 1
```

**输出**：完成层融合与旋转的模型。

### 步骤 4：执行下游量化并验收

**目标**：完成量化、保存并验证精度。

**操作**：

1. 在 Stage2 之后接下游量化处理器（如 `autoround_quant`）。
2. 配置 `save` 阶段保存量化模型。
3. 在验证集上评估量化精度，若不达标则调整 `layer_type`、`steps` 等参数后重跑。

**输出**：量化模型及精度验收结果。

## 6. 验收条件

- Stage1 能收集到非空激活并成功输出优化旋转矩阵。
- Stage2 能读取 `adapted_matrix` 并完成层融合与旋转。
- 量化流程执行成功，精度满足业务要求。

## 7. 异常处置

- **Stage1 未收集到激活**：`layer_type` 未匹配到模型中的 Linear 层，调整 `layer_type`（如 `up_proj`、`gate_proj`）。
- **Context 为空**：Stage1 未在 prior 阶段运行或未配置 `ContextManager`，确保两阶段在同一流程中顺序执行。
- **quant_dtype 与下游量化不一致**：将 Stage1 的 `quant_dtype` 设置为与下游 `qconfig.act.dtype` 一致。
- **MoE 模型执行缓慢**：缩小 `layer_type` 匹配范围，尽量只选择共享层而非专家非共享线性层。

## 8. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| Adapt Rotation | 数据驱动优化旋转矩阵的离群值抑制算法 | [Adapt Rotation 词条](./term_adapt_rotation.md) |
| Hadamard 优化 | 通过 Newton-Schulz 迭代求正交极因子以优化旋转矩阵 | [Adapt Rotation 词条](./term_adapt_rotation.md) |
| AdaptRotationInterface | 模型适配器需实现的接口（含 `get_hidden_dim()`） | [Adapt Rotation 词条](./term_adapt_rotation.md) |
| QuaRot | Adapt Rotation 的上位算法 | [QuaRot 词条](../quarot/term_quarot.md) |

## 9. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| `type: "adapt_rotation"` stage 1 | 收集激活并优化旋转矩阵 | [Adapt Rotation 词条](./term_adapt_rotation.md) |
| `type: "adapt_rotation"` stage 2 | 应用优化后的旋转矩阵 | [Adapt Rotation 词条](./term_adapt_rotation.md) |
| `AdaptRotationInterface` | 模型适配器需实现的接口 | [Adapt Rotation 词条](./term_adapt_rotation.md) |
