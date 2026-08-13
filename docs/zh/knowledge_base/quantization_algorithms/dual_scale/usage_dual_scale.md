# DualScale 使用指南

## 1. 适用范围

本流程适用于在 msModelSlim 中配置和使用 DualScale 双尺度量化算法。DualScale 作为 `linear_quant` 处理器的量化方法，通过两级缩放递进结构缓解异常通道影响，用于 Qwen3 稠密系列模型的 W4A4 低比特量化。

适用角色：算法工程师、模型部署工程师

适用场景：

- W4A4 等超低比特量化场景，需要保持较高模型精度的场景。
- 激活值存在结构化异常通道的模型量化场景。

不适用场景：

- 非 Qwen3 稠密系列模型（不保证可泛化）。
- NPU 显存小于 64G 的设备（算法包含训练过程）。

## 2. 流程关系与前置条件

**上级流程**：模型适配与验证通过后，确定量化方案阶段。

**前置条件**：

- 已安装兼容版本的 msModelSlim 工具（详见《[msModelSlim 工具安装指南](../../../install_guide/install_guide.md)》）。
- 已确认目标模型为 Qwen3 稠密系列（如 Qwen3-8B/14B/32B）。
- 已准备好足够的校准数据。
- 已确认 NPU 显存 ≥64G。

**后续操作**：量化流程执行 → 精度验证 → 部署上线或进入调优。

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型权重目录 | 模型下载或本地路径 | HuggingFace 格式，含 `config.json` 及 `*.safetensors` | 可被目标 Transformers 版本加载 |
| 输入 | 校准数据集 | 工具默认或用户指定 | JSONL 格式，每条含文本 prompt | 可被工具加载并完成前向推理 |
| 输入 | 量化 YAML 配置文件 | 用户编写 | 符合 msModelSlim YAML 规范，含 `scope: "dual_scale"` 量化配置 | 可通过工具 `--config_path` 参数加载 |
| 交付件 | 量化权重目录 | `--save_path` 指定路径 | 含 `quant_model_description.json` 及 `*.safetensors`，已应用双尺度量化 | 推理冒烟通过 |

## 4. 流程总览

```mermaid
flowchart LR
    A[编写 YAML 配置] --> B[执行量化命令]
    B --> C[外层缩放计算]
    C --> D[内层量化]
    D --> E[验证量化结果]
```

## 5. 操作步骤

### 步骤 1：编写 YAML 配置文件

**目标**：编写包含 DualScale 量化配置的 YAML 文件。

**操作**：

1. 在 `spec.process` 下配置 `linear_quant` 处理器。
2. 在 `qconfig.act` 与 `qconfig.weight` 中设置 `scope: "dual_scale"`、`dtype: "mxfp4"`、`symmetric: True`、`method: "dualscale"`。
3. 在 `ext` 中配置 `dual_block_size`（外层大块大小）。

YAML 配置示例：

```yaml
spec:
  process:
    - type: "linear_quant"
      qconfig:
        act:
          scope: "dual_scale"
          dtype: "mxfp4"
          symmetric: True
          method: "dualscale"
          ext:
            dual_block_size: 512
        weight:
          scope: "dual_scale"
          dtype: "mxfp4"
          symmetric: True
          method: "dualscale"
          ext:
            dual_block_size: 512
```

YAML 配置字段详解如下：

| 字段名 | 作用 | 说明 |
| --- | --- | --- |
| scope | 量化范围 | 固定为 `"dual_scale"`（双尺度）。 |
| dtype | 量化数据类型 | 固定为 `"mxfp4"`。 |
| symmetric | 是否对称量化 | `true` 为对称，`false` 为非对称。 |
| method | 量化方法 | 固定为 `"dualscale"`。 |
| ext.dual_block_size | 外层大块大小 | 整数，如 `512`。 |

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

1. 工具加载 YAML 配置，解析 `linear_quant` 处理器与 DualScale 配置。
2. 权重在初始化阶段完成静态量化存储（外层尺度 + 内层 block 量化）。
3. 前向时激活按外层缩放、内层量化、外层反量化执行。
4. 保存量化权重。

**输出**：量化权重目录 `${SAVE_PATH}`，包含量化描述文件与权重分片。

### 步骤 3：验证量化结果

**目标**：确认量化权重文件完整且可加载。

**操作**：

1. 检查输出目录是否包含 `quant_model_description.json` 文件。
2. 检查日志确认量化流程正常完成。
3. 使用推理框架加载量化权重进行冒烟测试。

**输出**：量化权重验证通过。

## 6. 验收条件

- 量化权重目录包含 `quant_model_description.json` 及所有必需的 `*.safetensors` 分片文件。
- 日志无配置组合无效告警。
- 量化后模型推理精度优于基线。

## 7. 异常处置

- **配置组合无效**：确认 `scope` 为 `dual_scale`、`dtype` 为 `mxfp4`、`method` 为 `dualscale`。
- **精度不达标**：调整 `dual_block_size`，或确认校准数据质量与模型适配范围。
- **显存不足**：确认 NPU 显存 ≥64G，必要时减少校准数据规模。

## 8. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| DualScale | 两级缩放递进的双尺度量化算法 | [DualScale 词条](./term_dual_scale.md) |
| dual_block_size | 外层大块的大小 | [DualScale 词条](./term_dual_scale.md) |
| 异常通道 | 数值平均高出其他通道数个数量级的通道 | [DualScale 词条](./term_dual_scale.md) |

## 9. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| `linear_quant` 处理器 | 线性层量化处理器，通过 `scope: "dual_scale"` 启用 DualScale | [线性量化词条](../linear_quant/term_linear_quant.md) |
| `MXWeightDualScaleMinmax` | 权重双尺度量化器 | [DualScale 词条](./term_dual_scale.md) |
| `MXActDualScaleMinmax` | 激活双尺度量化器 | [DualScale 词条](./term_dual_scale.md) |
