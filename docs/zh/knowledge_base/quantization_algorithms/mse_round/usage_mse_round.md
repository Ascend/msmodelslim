# MSE_Round 使用指南

## 1. 适用范围

本流程适用于在 msModelSlim 中配置和使用 MSE_Round 权重量化算法。MSE_Round 作为 `linear_quant` 处理器的权重量化方法，通过 per-block 比较 ceil/floor 两档 shared exponent 的 MSE，提升 MXFP8 权重量化精度。

适用角色：算法工程师、模型部署工程师

适用场景：

- 对 MXFP8 权重量化精度有更高要求的场景。
- block 内最大值分布不均、floor 缩放导致大值截断的模型层。

不适用场景：

- 激活值量化场景（MSE_Round 仅注册于 `mxfp8_per_block_sym` 权重量化方案）。

## 2. 流程关系与前置条件

**上级流程**：模型适配与验证通过后，确定量化方案阶段。

**前置条件**：

- 已安装兼容版本的 msModelSlim 工具（详见《[msModelSlim 工具安装指南](../../../install_guide/install_guide.md)》）。
- 已确认目标模型包含可量化的 `nn.Linear` 模块。
- 已确定量化方案为 W8A8 MXFP8 混精场景。

**后续操作**：量化流程执行 → 精度验证 → 部署上线或进入调优。

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型权重目录 | 模型下载或本地路径 | HuggingFace 格式，含 `config.json` 及 `*.safetensors` | 可被目标 Transformers 版本加载 |
| 输入 | 量化 YAML 配置文件 | 用户编写 | 符合 msModelSlim YAML 规范，含 `method: "mse_round"` 权重量化配置 | 可通过工具 `--config_path` 参数加载 |
| 交付件 | 量化权重目录 | `--save_path` 指定路径 | 含 `quant_model_description.json` 及 `*.safetensors`，已应用 MSE_Round 量化 | 推理冒烟通过 |

## 4. 流程总览

```mermaid
flowchart LR
    A[编写 YAML 配置] --> B[执行量化命令]
    B --> C[计算 ceil/floor]
    C --> D[MSE 择优]
    D --> E[验证量化结果]
```

## 5. 操作步骤

### 步骤 1：编写 YAML 配置文件

**目标**：编写包含 MSE_Round 权重量化配置的 YAML 文件。

**操作**：

1. 在 `spec.process` 下配置 `linear_quant` 处理器。
2. 在 `qconfig.weight` 中设置 `scope: "per_block"`、`dtype: "mxfp8"`、`symmetric: true`、`method: "mse_round"`。
3. 激活值量化使用 `minmax` 方法（W8A8 MXFP 混精场景）。

YAML 配置示例：

```yaml
spec:
  process:
    - type: "linear_quant"
      qconfig:
        act:
          scope: "per_block"
          dtype: "mxfp8"
          symmetric: true
          method: "minmax"
        weight:
          scope: "per_block"
          dtype: "mxfp8"
          symmetric: true
          method: "mse_round"
```

YAML 配置字段详解如下：

| 字段名 | 作用 | 说明 |
| --- | --- | --- |
| scope | 量化范围 | 固定为 `"per_block"`。 |
| dtype | 量化数据类型 | 固定为 `"mxfp8"`。 |
| symmetric | 是否对称量化 | `true`。 |
| method | 量化方法 | 固定为 `"mse_round"`。 |

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

1. 工具加载 YAML 配置，解析 `linear_quant` 处理器与 MSE_Round 配置。
2. 量化器计算每个 block 的 ceil/floor 两档候选 shared exponent。
3. 分别量化-反量化并计算 MSE，选择更优的 shared exponent。
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
- 量化后模型推理精度优于使用固定 floor 缩放（minmax）的基线。

## 7. 异常处置

- **配置组合无效**：确认权重 `scope` 为 `per_block`、`dtype` 为 `mxfp8`、`method` 为 `mse_round`。
- **精度不达标**：确认激活值量化配置合理，必要时配合离群值抑制算法使用。
- **不支持激活量化**：MSE_Round 仅支持权重量化，激活值量化请使用 `minmax`。

## 8. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| MSE_Round | per-block 比较 ceil/floor 两档 MSE 的 MXFP8 权重量化算法 | [MSE_Round 词条](./term_mse_round.md) |
| shared exponent | MXFP 格式中 block 共享的指数缩放参数 | [MSE_Round 词条](./term_mse_round.md) |
| MXFP8 | 基于 E4M3 尾数 + E8M0 指数的块共享指数浮点格式 | [MSE_Round 词条](./term_mse_round.md) |

## 9. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| `linear_quant` 处理器 | 线性层量化处理器，通过 `method: "mse_round"` 启用 MSE_Round | [线性量化词条](../linear_quant/term_linear_quant.md) |
| `MXWeightPerBlockMseRound` | MSE_Round 权重量化实现类 | [MSE_Round 词条](./term_mse_round.md) |
