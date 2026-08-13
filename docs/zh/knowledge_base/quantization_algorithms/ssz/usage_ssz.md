# SSZ 使用指南

## 1. 适用范围

本流程适用于在 msModelSlim 中配置和使用 SSZ 权重量化算法。SSZ 作为 `linear_quant` 处理器的权重量化方法，通过迭代搜索最优 scale/offset 提升权重分布不均匀场景下的量化精度。

适用角色：算法工程师、模型部署工程师

适用场景：

- 对精度要求较高、权重分布不均匀的线性层量化场景。
- INT4 等低比特权重量化场景。

不适用场景：

- per_tensor 或 per_group 量化粒度（当前暂不支持）。
- 非 2D 权重张量。

## 2. 流程关系与前置条件

**上级流程**：模型适配与验证通过后，确定量化方案阶段。

**前置条件**：

- 已安装兼容版本的 msModelSlim 工具（详见《[msModelSlim 工具安装指南](../../../install_guide/install_guide.md)》）。
- 已确认目标模型的线性层权重为 2D 张量。
- 权重必须支持 MinMax 观察器初始化（先调用 `init_weight` 再调用 `forward`）。

**后续操作**：量化流程执行 → 精度验证 → 部署上线或进入调优。

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型权重目录 | 模型下载或本地路径 | HuggingFace 格式，含 `config.json` 及 `*.safetensors` | 可被目标 Transformers 版本加载 |
| 输入 | 量化 YAML 配置文件 | 用户编写 | 符合 msModelSlim YAML 规范，含 `method: "ssz"` 权重量化配置 | 可通过工具 `--config_path` 参数加载 |
| 交付件 | 量化权重目录 | `--save_path` 指定路径 | 含 `quant_model_description.json` 及 `*.safetensors`，已应用 SSZ 量化 | 推理冒烟通过 |

## 4. 流程总览

```mermaid
flowchart LR
    A[编写 YAML 配置] --> B[执行量化命令]
    B --> C[MinMax 初始化]
    C --> D[迭代搜索参数]
    D --> E[验证量化结果]
```

## 5. 操作步骤

### 步骤 1：编写 YAML 配置文件

**目标**：编写包含 SSZ 权重量化配置的 YAML 文件。

**操作**：

1. 在 `spec.process` 下配置 `linear_quant` 处理器。
2. 在 `qconfig.weight` 中设置 `method: "ssz"`，配置 `scope: "per_channel"` 及 `dtype`、`symmetric`。

YAML 配置示例：

```yaml
spec:
  process:
    - type: "linear_quant"
      qconfig:
        weight:
          scope: "per_channel"
          dtype: "int8"
          symmetric: true
          method: "ssz"
```

YAML 配置字段详解如下：

| 字段名 | 作用 | 说明 |
| --- | --- | --- |
| scope | 量化范围 | 仅支持 `"per_channel"`。 |
| dtype | 量化数据类型 | `"int8"` 或 `"int4"`。 |
| symmetric | 是否对称量化 | `true` 为对称，`false` 为非对称。 |
| method | 量化方法 | 固定为 `"ssz"`。 |

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

1. 工具加载 YAML 配置，解析 `linear_quant` 处理器与 SSZ 配置。
2. 使用 MinMax 观察器初始化 scale 和 offset。
3. 迭代执行最小二乘求解与贪心更新，直到收敛。
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
- 日志无权重维度错误或初始化顺序错误告警。
- 量化后模型推理精度优于使用 MinMax 的基线。

## 7. 异常处置

- **权重维度错误**：输入的权重维度错误导致量化失败，确保权重是 2D 张量。
- **量化配置错误**：检查 `dtype`、`scope`、`method`、`symmetric` 参数设置是否正确。
- **初始化顺序错误**：必须先调用 `init_weight`，再调用 `forward`。
- **收敛问题**：如果算法不收敛，可调整 `SCALE_SEARCH_CONVERGE_THRESHOLD` 参数。

## 8. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| SSZ | 通过迭代搜索优化 scale/offset 的权重量化算法 | [SSZ 词条](./term_ssz.md) |
| 最小二乘法 | 计算当前最优 scale/offset 的求解方法 | [SSZ 词条](./term_ssz.md) |
| 收敛阈值 | 判断迭代是否收敛的误差阈值 | [SSZ 词条](./term_ssz.md) |

## 9. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| `linear_quant` 处理器 | 线性层量化处理器，通过 `method: "ssz"` 启用 SSZ | [线性量化词条](../linear_quant/term_linear_quant.md) |
| `ssz_calculate_qparam` | SSZ 量化参数计算核心函数 | [SSZ 词条](./term_ssz.md) |
