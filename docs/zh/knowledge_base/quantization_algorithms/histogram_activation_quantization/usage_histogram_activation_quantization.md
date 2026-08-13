# Histogram 使用指南

## 1. 适用范围

本流程适用于在 msModelSlim 中配置和使用 Histogram 直方图激活值量化算法。Histogram 作为 `linear_quant` 处理器的激活值量化方法，通过直方图分析自动搜索最优截断区间，用于提升激活值静态量化的精度。

适用角色：算法工程师、模型部署工程师

适用场景：

- 激活值存在离群值、MinMax 量化精度损失较大的场景。
- 需要更高精度的激活值静态量化场景。

不适用场景：

- 需要 per_token 或 int4 的激活值量化（当前仅支持 per_tensor/int8）。
- 权重量化场景（Histogram 仅支持激活值量化）。

## 2. 流程关系与前置条件

**上级流程**：模型适配与验证通过后，确定量化方案阶段。

**前置条件**：

- 已安装兼容版本的 msModelSlim 工具（详见《[msModelSlim 工具安装指南](../../../install_guide/install_guide.md)》）。
- 已确认目标模型包含可量化的 `nn.Linear` 模块。
- 已准备好校准数据集，用于构建直方图统计。

**后续操作**：量化流程执行 → 精度验证 → 部署上线或进入调优。

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型权重目录 | 模型下载或本地路径 | HuggingFace 格式，含 `config.json` 及 `*.safetensors` | 可被目标 Transformers 版本加载 |
| 输入 | 校准数据集 | 工具默认或用户指定 | JSONL 格式，每条含文本 prompt | 可被工具加载并完成前向推理 |
| 输入 | 量化 YAML 配置文件 | 用户编写 | 符合 msModelSlim YAML 规范，含 `method: "histogram"` 激活配置 | 可通过工具 `--config_path` 参数加载 |
| 交付件 | 量化权重目录 | `--save_path` 指定路径 | 含 `quant_model_description.json` 及 `*.safetensors`，已应用直方图量化 | 推理冒烟通过 |

## 4. 流程总览

```mermaid
flowchart LR
    A[编写 YAML 配置] --> B[执行量化命令]
    B --> C[直方图统计]
    C --> D[搜索截断区间]
    D --> E[验证量化结果]
```

## 5. 操作步骤

### 步骤 1：编写 YAML 配置文件

**目标**：编写包含 Histogram 激活值量化配置的 YAML 文件。

**操作**：

1. 在 `spec.process` 下配置 `linear_quant` 处理器。
2. 在 `qconfig.act` 中设置 `method: "histogram"`，`scope` 配置为 `"per_tensor"`、`dtype` 配置为 `"int8"`。
3. 权重配置中不要使用 `method: "histogram"`（不支持直方图权重量化）。

YAML 配置示例：

```yaml
spec:
  process:
    - type: "linear_quant"
      qconfig:
        act:
          scope: "per_tensor"  # 目前只支持per_tensor
          dtype: "int8"        # 目前只支持int8
          symmetric: false     # 支持对称/非对称量化，取值分别为True/False
          method: "histogram"  # 配置为"histogram", 即启用直方图激活值量化
        weight:
          scope: "per_channel"
          dtype: "int8"
          symmetric: true
          method: "minmax"     # 不支持直方图权重量化，此处不应配置为"histogram"
```

YAML 配置字段详解如下：

| 字段名 | 作用 | 说明 |
| --- | --- | --- |
| qconfig.act.scope | 激活量化范围 | 仅支持 `"per_tensor"`。 |
| qconfig.act.dtype | 激活量化数据类型 | 仅支持 `"int8"`。 |
| qconfig.act.symmetric | 是否对称量化 | `true` 为对称，`false` 为非对称。 |
| qconfig.act.method | 量化方法 | 固定为 `"histogram"`，启用直方图激活值量化。 |

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

1. 工具加载 YAML 配置，解析 `linear_quant` 处理器与 Histogram 配置。
2. 校准阶段通过 `HistogramObserver` 统计激活值直方图。
3. 搜索最优截断区间并计算量化参数。
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
- 日志无配置错误（`ValidationError`）告警。
- 量化后模型推理精度优于使用 MinMax 的基线。

## 7. 异常处置

- **配置错误**：日志出现 `ValidationError`，检查是否将 histogram 错误配置到 weight 处，或使用了 histogram 不支持的配置（如 int4）。
- **不支持权重量化**：确认权重配置中未使用 `method: "histogram"`。
- **精度下降**：确认校准数据足够且分布具有代表性，必要时配合离群值抑制算法使用。

## 8. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| Histogram | 基于直方图搜索最优截断区间的激活值量化算法 | [Histogram 词条](./term_histogram_activation_quantization.md) |
| 截断区间 | 过滤离群值后的量化范围 | [Histogram 词条](./term_histogram_activation_quantization.md) |
| L2 范数误差 | 量化前后分布的 L2 距离度量 | [Histogram 词条](./term_histogram_activation_quantization.md) |

## 9. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| `linear_quant` 处理器 | 线性层量化处理器，通过 `method: "histogram"` 启用直方图量化 | [线性量化词条](../linear_quant/term_linear_quant.md) |
| `HistogramObserver` | 直方图统计与截断值搜索观察器 | [Histogram 词条](./term_histogram_activation_quantization.md) |
