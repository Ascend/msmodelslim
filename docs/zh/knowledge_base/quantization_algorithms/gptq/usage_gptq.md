# GPTQ 使用指南

## 1. 适用范围

本流程适用于在 msModelSlim 中配置和使用 GPTQ 权重量化算法。GPTQ 作为 `linear_quant` 处理器的权重量化方法，通过逐列量化与二阶误差补偿提升权重量化精度。

适用角色：算法工程师、模型部署工程师

适用场景：

- 对精度要求较高的权重量化场景。
- 权重分布不均匀、传统 MinMax 量化误差较大的场景。

不适用场景：

- int4 低比特权重量化（当前暂不支持）。
- MoE 模型量化（校准集难以覆盖所有专家）。
- per_tensor 量化粒度或非 2D 权重张量。

## 2. 流程关系与前置条件

**上级流程**：模型适配与验证通过后，确定量化方案阶段。

**前置条件**：

- 已安装兼容版本的 msModelSlim 工具（详见《[msModelSlim 工具安装指南](../../../install_guide/install_guide.md)》）。
- 已确认目标模型的线性层为 `torch.nn.Linear` 且权重为 2D 张量。
- 已准备好校准数据集，用于收集激活并构建 Hessian 矩阵。

**后续操作**：量化流程执行 → 精度验证 → 部署上线或进入调优。

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型权重目录 | 模型下载或本地路径 | HuggingFace 格式，含 `config.json` 及 `*.safetensors` | 可被目标 Transformers 版本加载 |
| 输入 | 校准数据集 | 工具默认或用户指定 | JSONL 格式，每条含文本 prompt | 可被工具加载并完成前向推理 |
| 输入 | 量化 YAML 配置文件 | 用户编写 | 符合 msModelSlim YAML 规范，含 `method: "gptq"` 权重量化配置 | 可通过工具 `--config_path` 参数加载 |
| 交付件 | 量化权重目录 | `--save_path` 指定路径 | 含 `quant_model_description.json` 及 `*.safetensors`，已应用 GPTQ 量化 | 推理冒烟通过 |

## 4. 流程总览

```mermaid
flowchart LR
    A[编写 YAML 配置] --> B[执行量化命令]
    B --> C[构建 Hessian]
    C --> D[逐列量化与补偿]
    D --> E[验证量化结果]
```

## 5. 操作步骤

### 步骤 1：编写 YAML 配置文件

**目标**：编写包含 GPTQ 权重量化配置的 YAML 文件。

**操作**：

1. 在 `spec.process` 下配置 `linear_quant` 处理器。
2. 在 `qconfig.weight` 中设置 `method: "gptq"`，并按需配置 `scope`、`dtype`、`symmetric`。
3. 按需在 `ext` 中配置 `percdamp`（默认 `0.01`）、`block_size`（默认 `128`）、`group_size`（默认 `128`）。

YAML 配置示例（per_channel）：

```yaml
spec:
  process:
    - type: "linear_quant"
      qconfig:
        weight:
          scope: "per_channel"   # 量化范围
          dtype: "int8"          # 量化数据类型
          symmetric: true        # 是否对称量化
          method: "gptq"         # 量化算法-GPTQ
          ext: # 可选，扩展参数
            percdamp: 0.01       # 可选，阻尼系数，默认值0.01
            block_size: 128      # 可选，分块大小，默认值128
```

YAML 配置字段详解如下：

| 字段名 | 作用 | 说明 |
| --- | --- | --- |
| scope | 量化范围 | `"per_channel"` 或 `"per_group"`。 |
| dtype | 量化数据类型 | `"int8"`（当前仅支持 int8）。 |
| symmetric | 是否对称量化 | `true` 为对称，`false` 为非对称。 |
| method | 量化方法 | 固定为 `"gptq"`。 |
| ext.percdamp | 阻尼系数 | 逆 Hessian 计算中的阻尼百分比，默认 `0.01`。 |
| ext.block_size | 迭代分块大小 | 每次迭代处理的列块大小，默认 `128`。 |
| ext.group_size | 量化分组大小 | 分组量化的大小，默认 `128`。 |

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

1. 工具加载 YAML 配置，解析 `linear_quant` 处理器与 GPTQ 配置。
2. 量化过程中收集层激活构建 Hessian 矩阵。
3. 逐列量化权重，并通过二阶误差补偿更新未量化权重。
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
- 量化后模型推理精度优于使用 MinMax 的基线。

## 7. 异常处置

- **量化速度慢**：GPTQ 需要计算 Hessian 矩阵并逐列量化，量化速度较慢、计算成本较大，可调整 `block_size` 平衡速度与显存。
- **percdamp 导致精度损失**：`percdamp` 过大可能导致精度损失，过小可能引起数值不稳定，按需在 `0.01` 附近调整。
- **group_size 配置错误**：确认 `group_size` 能被待量化层的 `input_features` 维度整除。

## 8. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| GPTQ | 基于二阶误差补偿的高精度权重量化算法 | [GPTQ 词条](./term_gptq.md) |
| Hessian 矩阵 | 激活的二阶统计量，用于评估量化影响 | [GPTQ 词条](./term_gptq.md) |
| percdamp | 逆 Hessian 计算中的阻尼系数 | [GPTQ 词条](./term_gptq.md) |

## 9. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| `linear_quant` 处理器 | 线性层量化处理器，通过 `method: "gptq"` 启用 GPTQ | [线性量化词条](../linear_quant/term_linear_quant.md) |
| GPTQ 扩展配置 | `percdamp`、`block_size`、`group_size` 等参数 | [GPTQ 词条](./term_gptq.md) |
