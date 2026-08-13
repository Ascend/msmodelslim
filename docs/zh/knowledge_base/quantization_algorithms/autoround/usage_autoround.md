# AutoRound 使用指南

## 1. 适用范围

本流程适用于在 msModelSlim 中配置和使用 AutoRound 低比特量化算法。AutoRound 作为权重量化处理器，通过可学习舍入与 SignSGD 优化，用于 4bit 等超低比特量化场景。

适用角色：算法工程师、模型部署工程师

适用场景：

- 4bit 等超低比特权重量化场景。
- 低比特条件下仍需要保持较高模型精度的场景。

不适用场景：

- 非 LLM 线性层以外的量化目标。
- NPU 显存小于 64G 的设备（算法包含训练过程）。

## 2. 流程关系与前置条件

**上级流程**：模型适配与验证通过后，确定量化方案阶段。

**前置条件**：

- 已安装兼容版本的 msModelSlim 工具（详见《[msModelSlim 工具安装指南](../../../install_guide/install_guide.md)》）。
- 已确认目标模型支持线性层量化。
- 已准备好足够的校准数据或训练迭代次数。
- 建议先配置离群值抑制算法（如 QuaRot、Iterative Smooth）作为前置步骤。

**后续操作**：量化流程执行 → 精度验证 → 部署上线或进入调优。

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型权重目录 | 模型下载或本地路径 | HuggingFace 格式，含 `config.json` 及 `*.safetensors` | 可被目标 Transformers 版本加载 |
| 输入 | 校准数据集 | 工具默认或用户指定 | JSONL 格式，每条含文本 prompt | 至少包含足够样本完成迭代优化 |
| 输入 | 量化 YAML 配置文件 | 用户编写 | 符合 msModelSlim YAML 规范，含 `autoround_quant` 处理器配置 | 可通过工具 `--config_path` 参数加载 |
| 交付件 | 量化权重目录 | `--save_path` 指定路径 | 含 `quant_model_description.json` 及 `*.safetensors`，已应用优化后的量化权重 | 推理冒烟通过 |

## 4. 流程总览

```mermaid
flowchart LR
    A[编写 YAML 配置] --> B[执行量化命令]
    B --> C[逐层优化舍入]
    C --> D[应用量化权重]
    D --> E[验证量化结果]
```

## 5. 操作步骤

### 步骤 1：编写 YAML 配置文件

**目标**：编写包含 `autoround_quant` 处理器配置的 YAML 文件。

**操作**：

1. 在 `spec.process` 下配置 `autoround_quant` 处理器，指定 `type: "autoround_quant"`。
2. 配置 `iters`（迭代次数，默认 `10`）、`enable_minmax_tuning`（默认 `True`）、`enable_round_tuning`（默认 `True`）。
3. 配置 `strategies` 量化策略，支持对不同层使用不同量化配置（混合量化）。
4. 建议在 `spec.process` 中先配置离群值抑制算法（如 QuaRot、Iterative Smooth）。

YAML 配置示例（W8A8 与 W4A4 混合量化）：

```yaml
spec:
  process:
    - type: "autoround_quant"     # 固定为 `autoround_quant`，用于指定 Processor 类型。
      iters: 400                  # 优化迭代次数
      enable_minmax_tuning: True  # 是否启用最小最大值调优
      enable_round_tuning: True   # 是否启用舍入调优
      strategies:
        # 策略1：除 up_proj、gate_proj 和 o_proj 层外，其余层均应用 W8A8 量化。
        - qconfig: *default_w8a8_dynamic
          exclude:
            - "*.up_proj"
            - "*.gate_proj"
            - "*.o_proj"
        # 策略2：对up_proj、gate_proj、o_proj层使用W4A4量化
        - qconfig: *default_w4a4_dynamic
          include:
            - "*.up_proj"
            - "*.gate_proj"
            - "*.o_proj"
```

YAML 配置字段详解如下：

| 字段名 | 作用 | 说明 |
| --- | --- | --- |
| type | 处理器类型标识 | 固定为 `"autoround_quant"`。 |
| iters | 优化迭代次数 | 大于 0 的整数，影响优化效果与计算时间，默认 `10`。 |
| enable_minmax_tuning | 最小最大值调优开关 | 布尔值，是否启用最小最大值调优，默认 `True`。 |
| enable_round_tuning | 舍入调优开关 | 布尔值，是否启用舍入调优，默认 `True`。 |
| strategies | 量化策略配置 | 策略列表，支持对不同层使用不同量化配置（如 int4 与 int8 混合量化）。 |

**层过滤机制**：

`include` 定义要包含的层，只有匹配 `include` 模式的层才会被处理；`exclude` 定义要排除的层，匹配 `exclude` 模式的层会被跳过；`exclude` 的优先级高于 `include`。匹配采用 Unix 通配符模式：`*` 匹配任意字符序列、`?` 匹配单个字符、`[abc]` 匹配字符集中的任意字符。若 `include`/`exclude` 未匹配到任何层，工具会进行告警，请核对层名、路径层级、大小写与拼写。

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

1. 工具加载 YAML 配置，解析 `autoround_quant` 处理器。
2. preprocess 阶段逐层采集浮点基准输出并注入可训练量化参数。
3. process 阶段逐层运行 SignSGD 优化，最小化重构误差。
4. postprocess 阶段应用优化后的量化参数并保存量化权重。

**输出**：量化权重目录 `${SAVE_PATH}`，包含量化描述文件与权重分片。

### 步骤 3：验证量化结果

**目标**：确认量化权重文件完整且可加载。

**操作**：

1. 检查输出目录是否包含 `quant_model_description.json` 文件。
2. 检查日志确认优化过程收敛且无层匹配告警。
3. 使用推理框架加载量化权重进行冒烟测试。

**输出**：量化权重验证通过。

## 6. 验收条件

- 量化权重目录包含 `quant_model_description.json` 及所有必需的 `*.safetensors` 分片文件。
- 日志无优化不收敛或层匹配告警。
- 量化后模型推理精度优于未使用 AutoRound 的基线。

## 7. 异常处置

- **优化不收敛**：浮点结果与量化结果差距波动较大，调整学习率或增加迭代次数。
- **精度下降明显**：增加优化步数、调整量化配置、减少 `w4a4` 量化层数，或使用更多更优质的校准数据。
- **group_size 配置错误**：抛出 shape 相关异常，确认 `group_size` 能被待量化 `nn.Linear` 层的 `input_features` 维度整除。
- **层匹配告警**：检查模型定义，确保 `include`/`exclude` 模式匹配到正确的层。

## 8. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| AutoRound | 基于 SignSGD 的低比特权重量化算法 | [AutoRound 词条](./term_autoround.md) |
| SignSGD | 符号梯度下降优化器 | [AutoRound 词条](./term_autoround.md) |
| 混合量化 | 对不同层使用不同量化配置的策略 | [AutoRound 词条](./term_autoround.md) |

## 9. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| `autoround_quant` 处理器 | 用于执行 AutoRound 低比特量化的 Processor 配置 | [AutoRound 词条](./term_autoround.md) |
| 层过滤机制 | include/exclude 通配符匹配规则 | [线性量化词条](../linear_quant/term_linear_quant.md) |
