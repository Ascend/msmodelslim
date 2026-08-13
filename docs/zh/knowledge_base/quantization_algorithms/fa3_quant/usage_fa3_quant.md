# FA3 Quant 使用指南

## 1. 适用范围

本流程适用于在 msModelSlim 中配置和使用 FA3 Quant 注意力激活量化算法。FA3 Quant 作为激活量化处理器，对注意力机制中的 Q、K、V 激活进行 per-head 量化，用于降低显存占用并提升推理效率。

适用角色：算法工程师、模型部署工程师

适用场景：

- 长序列推理场景下需要降低 Attention 中间激活显存占用的场景。
- 基于 MLA 架构的模型（如 DeepSeek 系列）的全量化方案。

不适用场景：

- 模型适配器未实现 `FA3QuantAdapterInterface` 接口。
- 非 MLA 架构的注意力机制。

## 2. 流程关系与前置条件

**上级流程**：模型适配与验证通过后，确定量化方案阶段。

**前置条件**：

- 已安装兼容版本的 msModelSlim 工具（详见《[msModelSlim 工具安装指南](../../../install_guide/install_guide.md)》）。
- 已确认目标模型基于 MLA 架构，且适配器实现了 `FA3QuantAdapterInterface` 接口。
- 已准备好校准数据集，用于收集每个 head 的激活统计信息。

**后续操作**：量化流程执行 → 精度验证 → 部署上线或进入调优。

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型权重目录 | 模型下载或本地路径 | HuggingFace 格式，含 `config.json` 及 `*.safetensors` | 可被目标 Transformers 版本加载 |
| 输入 | 校准数据集 | 工具默认或用户指定 | JSONL 格式，每条含文本 prompt | 可被工具加载并完成前向推理 |
| 输入 | 量化 YAML 配置文件 | 用户编写 | 符合 msModelSlim YAML 规范，含 `fa3_quant` 处理器配置 | 可通过工具 `--config_path` 参数加载 |
| 交付件 | 量化权重目录 | `--save_path` 指定路径 | 含 `quant_model_description.json` 及 `*.safetensors`，已应用 FA3 量化 | 推理冒烟通过 |

## 4. 流程总览

```mermaid
flowchart LR
    A[编写 YAML 配置] --> B[执行量化命令]
    B --> C[注入占位器并校准]
    C --> D[计算量化参数]
    D --> E[验证量化结果]
```

## 5. 操作步骤

### 步骤 1：编写 YAML 配置文件

**目标**：编写包含 `fa3_quant` 处理器配置的 YAML 文件。

**操作**：

1. 在 `spec.process` 下配置 `fa3_quant` 处理器，指定 `type: "fa3_quant"`。
2. 配置 `qconfig`（统一量化配置）或 `details`（逐激活值详细配置），二者不可同时配置。
3. 按需配置 `include`/`exclude` 控制参与量化的注意力层。

YAML 配置示例（统一配置）：

```yaml
spec:
  process:
    - type: "fa3_quant"
      qconfig:
          dtype: "fp8_e4m3"
          scope: "per_token"
          symmetric: True
          method: "minmax"
      include: [ "*" ]                           # 包含的注意力层
      exclude: [ "model.layers.0.self_attn" ]   # 排除的注意力层
```

YAML 配置字段详解如下：

| 字段名 | 作用 | 说明 |
| --- | --- | --- |
| type | 处理器类型标识 | 固定为 `"fa3_quant"`。 |
| qconfig | 量化统一配置 | Q/K/V 的统一量化配置，与 `details` 不可同时配置。 |
| details | 量化详细配置 | 按 `fa_q`/`fa_k`/`fa_v` 分别配置各激活值的量化方式。 |
| include | 包含的注意力层 | 字符串列表，支持通配符匹配，指定执行 FA3 量化的注意力层。 |
| exclude | 排除的注意力层 | 字符串列表，支持通配符匹配，优先级高于 `include`。 |

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

1. 工具加载 YAML 配置，解析 `fa3_quant` 处理器。
2. 注入阶段在 MLA 计算关键位置插入占位器。
3. 校准阶段收集每个 head 的激活统计信息，使用 Recall Window 算法寻找量化范围。
4. 部署阶段计算量化参数并替换监听器，随后保存量化权重。

**输出**：量化权重目录 `${SAVE_PATH}`，包含量化描述文件与权重分片。

### 步骤 3：验证量化结果

**目标**：确认量化权重文件完整且可加载。

**操作**：

1. 检查输出目录是否包含 `quant_model_description.json` 文件。
2. 检查日志确认校准阶段正常完成。
3. 使用推理框架加载量化权重进行冒烟测试。

**输出**：量化权重验证通过。

## 6. 验收条件

- 量化权重目录包含 `quant_model_description.json` 及所有必需的 `*.safetensors` 分片文件。
- 日志无配置冲突（`qconfig` 与 `details` 同时配置）告警。
- 量化后模型推理精度不低于未使用 FA3 Quant 的基线。

## 7. 异常处置

- **配置冲突**：`qconfig` 与 `details` 字段不支持同时配置，检查 YAML 只保留其一。
- **模型不适配**：确认模型基于 MLA 架构且适配器实现了 `FA3QuantAdapterInterface`。
- **精度下降**：检查 `quant_type` 配置，必要时调整 Q/K/V 各自的量化粒度与格式。

## 8. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| FA3 Quant | 针对注意力激活的 per-head 量化算法 | [FA3 Quant 词条](./term_fa3_quant.md) |
| per-head 量化 | 对每个注意力头独立计算量化参数 | [FA3 Quant 词条](./term_fa3_quant.md) |
| quant_type | 各激活值的量化格式与策略组合标识 | [FA3 Quant 词条](./term_fa3_quant.md) |

## 9. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| `fa3_quant` 处理器 | 用于执行 FA3 量化的 Processor 配置 | [FA3 Quant 词条](./term_fa3_quant.md) |
| `FA3QuantAdapterInterface` | 模型适配器需实现的 FA3 注入接口 | [FA3 Quant 词条](./term_fa3_quant.md) |
