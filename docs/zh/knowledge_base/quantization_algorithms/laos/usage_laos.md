# LAOS 使用指南

## 1. 适用范围

本流程适用于在 msModelSlim 中配置和使用 LAOS W4A4 低比特量化方案。LAOS 通过组合 `adapt_rotation`（旋转优化）与 `autoround_quant`（低比特量化）处理器，用于 Qwen3 稠密系列模型的超低比特量化。

适用角色：算法工程师、模型部署工程师

适用场景：

- W4A4 等超低比特量化场景，需要保持较高模型精度的场景。
- 激活值存在显著离群值、需要数据驱动旋转抑制的场景。

不适用场景：

- 非 Qwen3 稠密系列模型（不保证可泛化）。
- NPU 显存小于 64G 的设备（算法包含训练过程）。

## 2. 流程关系与前置条件

**上级流程**：模型适配与验证通过后，确定量化方案阶段。

**前置条件**：

- 已安装兼容版本的 msModelSlim 工具（详见《[msModelSlim 工具安装指南](../../../install_guide/install_guide.md)》）。
- 已确认目标模型为 Qwen3 稠密系列（如 Qwen3-8B/14B/32B）且支持 `AdaptRotationInterface`。
- 已准备好足够的校准数据（如 `laos_calib.jsonl`）。
- 已确认 NPU 显存 ≥64G。

**后续操作**：量化流程执行 → 精度验证 → 部署上线或进入调优。

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型权重目录 | 模型下载或本地路径 | HuggingFace 格式，含 `config.json` 及 `*.safetensors` | 可被目标 Transformers 版本加载 |
| 输入 | 校准数据集 | 工具默认（`laos_calib.jsonl`）或用户指定 | JSONL 格式，每条含文本 prompt | 可被工具加载并完成前向推理 |
| 输入 | 量化 YAML 配置文件 | 用户编写 | 符合 msModelSlim YAML 规范，含 `adapt_rotation` 与 `autoround_quant` 配置 | 可通过工具 `--config_path` 参数加载 |
| 交付件 | 量化权重目录 | `--save_path` 指定路径 | 含 `quant_model_description.json` 及 `*.safetensors`，已应用旋转与低比特量化 | 推理冒烟通过 |

## 4. 流程总览

```mermaid
flowchart LR
    A[编写 YAML 配置] --> B[执行量化命令]
    B --> C[Stage1 旋转优化]
    C --> D[Stage2 应用旋转]
    D --> E[AutoRound 量化]
    E --> F[验证量化结果]
```

## 5. 操作步骤

### 步骤 1：编写 YAML 配置文件

**目标**：编写包含 LAOS 两段式配置的 YAML 文件。

**操作**：

1. 在 `spec.prior` 下配置 `adapt_rotation` Stage1，指定 `stage: 1`，配置 `steps`、`layer_type` 等参数。
2. 在 `spec.process` 下配置 `adapt_rotation` Stage2，指定 `stage: 2`。
3. 在 `spec.process` 中配置 `autoround_quant`，配置 `iters` 与 `strategies` 混合量化策略。
4. 配置 `save` 与 `dataset`。

YAML 配置示例：

```yaml
spec:
  prior:
    - process:
        - type: "adapt_rotation"
          stage: 1
          steps: 20
          layer_type: ["up_proj"]

  process:
    - type: "adapt_rotation"
      stage: 2
      online: false
      block_size: -1
      max_tp_size: 1

    - type: "autoround_quant"
      iters: 400
      enable_round_tuning: true
      strategies:
        - qconfig: *default_w8a8_dynamic
          include: ["*self_attn*", "*.down_proj", ...]
        - qconfig: *default_w4a4_dynamic
          include: ["*.up_proj", "*.gate_proj"]
          exclude: [...]

  save:
    - type: "ascendv1_saver"
      part_file_size: 4

  dataset: laos_calib.jsonl
```

YAML 配置字段详解如下：

| 字段名 | 作用 | 说明 |
| --- | --- | --- |
| adapt_rotation（Stage1） | 旋转矩阵优化 | `type: "adapt_rotation"`、`stage: 1`，配置 `steps`（默认 `20`）与 `layer_type`（如 `["up_proj"]`）。 |
| adapt_rotation（Stage2） | 应用优化旋转 | `type: "adapt_rotation"`、`stage: 2`，配置 `online`（默认 `false`）、`block_size`（默认 `-1`）、`max_tp_size`（默认 `1`）。 |
| autoround_quant | 低比特量化 | 配置 `iters`（默认 `400`）、`enable_round_tuning`（默认 `true`）及 `strategies` 混合量化策略。 |
| save | 权重保存 | `type: "ascendv1_saver"`，可配置 `part_file_size`。 |
| dataset | 校准数据集 | 指定校准数据集名称，如 `laos_calib.jsonl`。 |

**输出**：YAML 配置文件 `${CONFIG_PATH}`。

### 步骤 2：执行量化命令

**目标**：使用上一步编写的 YAML 配置文件启动量化流程。

**操作**：

```bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --device npu \
  --model_type ${MODEL_TYPE} \
  --config_path ${CONFIG_PATH} \
  --trust_remote_code True
```

参数说明：

| 参数 | 必选 | 说明 |
| --- | --- | --- |
| `model_path` | 是 | 浮点模型权重路径 |
| `save_path` | 是 | 量化权重保存路径 |
| `device` | 否 | 量化设备，默认 `npu` |
| `model_type` | 是 | 模型名称，与支持矩阵一致 |
| `config_path` | 是 | 步骤 1 编写的 YAML 配置路径 |
| `trust_remote_code` | 否 | 是否信任远程代码，默认 `False` |

执行流程说明：

1. 工具加载 YAML 配置，解析多阶段流程。
2. Stage1（prior 阶段）收集指定层激活并优化旋转矩阵。
3. Stage2（主阶段）应用优化后的旋转矩阵。
4. `autoround_quant` 执行低比特量化与舍入优化，随后保存量化权重。

**输出**：量化权重目录 `${SAVE_PATH}`，包含量化描述文件与权重分片。

### 步骤 3：验证量化结果

**目标**：确认量化权重文件完整且可加载。

**操作**：

1. 检查输出目录是否包含 `quant_model_description.json` 文件。
2. 检查日志确认 Stage1 与 Stage2 均成功执行。
3. 使用推理框架加载量化权重进行冒烟测试。

**输出**：量化权重验证通过。

## 6. 验收条件

- 量化权重目录包含 `quant_model_description.json` 及所有必需的 `*.safetensors` 分片文件。
- 日志无 Stage1 未收集到激活或 Context 为空告警。
- 量化后模型推理精度优于基线。

## 7. 异常处置

- **Stage1 未收集到激活**：检查 `layer_type` 是否与模型中的 Linear 层名称匹配。
- **Context 为空**：确保 Stage1 在 prior 阶段运行且配置了 `ContextManager`。
- **精度不达标**：尝试调整 `steps`、`iters`、`max_samples`，或检查 `quant_dtype` 是否与下游量化配置一致。
- **显存不足**：确认 NPU 显存 ≥64G，必要时减少校准数据规模。

## 8. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| LAOS | W4A4 低比特量化方案（旋转优化 + 舍入训练） | [LAOS 词条](./term_laos.md) |
| Adapt Rotation | 数据驱动优化旋转矩阵的离群值抑制算法 | [Adapt Rotation 词条](../adapt_rotation/term_adapt_rotation.md) |
| AutoRound | 基于 SignSGD 的低比特权重量化算法 | [AutoRound 词条](../autoround/term_autoround.md) |

## 9. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| `adapt_rotation` 处理器 | 旋转优化处理器（Stage1/Stage2） | [Adapt Rotation 词条](../adapt_rotation/term_adapt_rotation.md) |
| `autoround_quant` 处理器 | 低比特量化与舍入优化处理器 | [AutoRound 词条](../autoround/term_autoround.md) |
| `ascendv1_saver` | 量化权重保存处理器 | [LAOS 词条](./term_laos.md) |
