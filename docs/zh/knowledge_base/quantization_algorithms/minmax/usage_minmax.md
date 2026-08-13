# MinMax 使用指南

## 1. 适用范围

本流程适用于在 msModelSlim 中配置和使用 MinMax 最小最大值量化算法。MinMax 作为 `linear_quant` 处理器的激活值或权重量化方法，通过统计张量极值计算量化参数，是大多数常规量化场景的首选。

适用角色：算法工程师、模型部署工程师

适用场景：

- 基础量化场景，INT8 等常规位宽的量化。
- 对计算开销敏感、需要快速完成量化的场景。

不适用场景：

- 张量存在极端离群值、需要更高量化精度的低比特场景（建议配合离群值抑制或换用 SSZ 等算法）。

## 2. 流程关系与前置条件

**上级流程**：模型适配与验证通过后，确定量化方案阶段。

**前置条件**：

- 已安装兼容版本的 msModelSlim 工具（详见《[msModelSlim 工具安装指南](../../../install_guide/install_guide.md)》）。
- 已确认目标模型包含可量化的 `nn.Linear` 模块。
- 静态量化需准备好校准数据集；动态量化可直接在线计算量化参数。

**后续操作**：量化流程执行 → 精度验证 → 部署上线或进入调优。

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型权重目录 | 模型下载或本地路径 | HuggingFace 格式，含 `config.json` 及 `*.safetensors` | 可被目标 Transformers 版本加载 |
| 输入 | 校准数据集 | 工具默认或用户指定（静态量化必选） | JSONL 格式，每条含文本 prompt | 可被工具加载并完成前向推理 |
| 输入 | 量化 YAML 配置文件 | 用户编写 | 符合 msModelSlim YAML 规范，含 `method: "minmax"` 量化配置 | 可通过工具 `--config_path` 参数加载 |
| 交付件 | 量化权重目录 | `--save_path` 指定路径 | 含 `quant_model_description.json` 及 `*.safetensors`，已应用 MinMax 量化 | 推理冒烟通过 |

## 4. 流程总览

```mermaid
flowchart LR
    A[编写 YAML 配置] --> B[执行量化命令]
    B --> C[统计 min/max]
    C --> D[量化并部署]
    D --> E[验证量化结果]
```

## 5. 操作步骤

### 步骤 1：编写 YAML 配置文件

**目标**：编写包含 MinMax 量化配置的 YAML 文件。

**操作**：

1. 在 `spec.process` 下配置 `linear_quant` 处理器。
2. 在 `qconfig.act` 与 `qconfig.weight` 中设置 `method: "minmax"`，并配置 `scope`、`dtype`、`symmetric`。
3. 按需配置 `include`/`exclude` 通配符控制量化层范围。

YAML 配置示例：

```yaml
spec:
  process:
    - type: "linear_quant"
      qconfig:
        act:
          scope: "per_tensor"   # 激活值量化范围
          dtype: "int8"         # 激活值量化数据类型
          symmetric: false      # 激活值非对称量化
          method: "minmax"      # 激活值使用 minmax
        weight:
          scope: "per_channel"  # 权重量化范围
          dtype: "int8"         # 权重量化数据类型
          symmetric: true       # 权重对称量化
          method: "minmax"      # 权重使用 minmax
      include: ["*"]            # 包含的层
      exclude: []               # 排除的层
```

YAML 配置字段详解如下：

| 字段名 | 作用 | 说明 |
| --- | --- | --- |
| qconfig.act.method | 激活值量化方法 | 固定为 `"minmax"`，指定激活值使用 MinMax 算法。 |
| qconfig.weight.method | 权重量化方法 | 固定为 `"minmax"`，指定权重使用 MinMax 算法。 |

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

1. 工具加载 YAML 配置，解析 `linear_quant` 处理器与 MinMax 配置。
2. 量化器统计张量极值，按 `scope` 粒度计算 scale 与 offset。
3. 对命中规则的层执行量化并生成量化 IR。
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
- 日志无层匹配告警。
- 量化后模型推理精度符合业务要求。

## 7. 异常处置

- **精度问题**：MinMax 在低比特（如 INT8、INT4）下精度下降明显，建议配合离群值抑制算法（如 SmoothQuant）或使用更高级的量化算法（如 SSZ）。
- **层匹配告警**：`include`/`exclude` 未匹配到任何层时工具告警，检查层名、路径层级、大小写与拼写。

## 8. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| MinMax | 通过统计极值确定量化范围的量化算法 | [MinMax 词条](./term_minmax.md) |
| 缩放因子 | 决定量化步长的参数 | [MinMax 词条](./term_minmax.md) |
| 对称/非对称量化 | 零点为 0 或可调整的两种量化模式 | [MinMax 词条](./term_minmax.md) |

## 9. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| `linear_quant` 处理器 | 线性层量化处理器，通过 `method: "minmax"` 启用 MinMax | [线性量化词条](../linear_quant/term_linear_quant.md) |
| `qconfig.act.method` / `qconfig.weight.method` | 指定激活值/权重使用 MinMax 算法 | [MinMax 词条](./term_minmax.md) |
