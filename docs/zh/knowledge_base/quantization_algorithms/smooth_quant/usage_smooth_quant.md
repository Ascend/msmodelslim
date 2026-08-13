# SmoothQuant 使用指南

## 1. 适用范围

本流程适用于在 msModelSlim 中配置和使用 SmoothQuant 平滑量化算法。SmoothQuant 作为离群值抑制算法，通常作为量化前的预处理步骤，用于将激活离群值迁移到权重，提升低比特量化的精度。

适用角色：算法工程师、模型部署工程师

适用场景：

- 激活值存在显著离群值，直接量化精度损失较大的场景。
- W8A8、W4A8 等量化方案的前置离群值抑制步骤。

不适用场景：

- 模型未实现 `SmoothQuantInterface` 接口，或无 `norm-linear` 子图映射。
- 需要处理 `ov`、`up-down`、`linear-linear` 等非 `norm-linear` 子图类型。

## 2. 流程关系与前置条件

**上级流程**：模型适配与验证通过后，确定量化方案阶段。

**前置条件**：

- 已安装兼容版本的 msModelSlim 工具（详见《[msModelSlim 工具安装指南](../../../install_guide/install_guide.md)》）。
- 已确认目标模型适配器实现了 `SmoothQuantInterface` 接口，并正确配置 `norm-linear` 子图映射。
- 已准备好校准数据集，用于收集激活统计信息。

**后续操作**：量化流程执行 → 精度验证 → 部署上线或进入调优。

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型权重目录 | 模型下载或本地路径 | HuggingFace 格式，含 `config.json` 及 `*.safetensors` | 可被目标 Transformers 版本加载 |
| 输入 | 校准数据集 | 工具默认或用户指定 | JSONL 格式，每条含文本 prompt | 可被工具加载并完成前向推理 |
| 输入 | 量化 YAML 配置文件 | 用户编写 | 符合 msModelSlim YAML 规范，含 `smooth_quant` 处理器配置 | 可通过工具 `--config_path` 参数加载 |
| 交付件 | 平滑后的量化权重目录 | `--save_path` 指定路径 | 含 `quant_model_description.json` 及 `*.safetensors`，已应用平滑缩放 | 推理冒烟通过 |

## 4. 流程总览

```mermaid
flowchart LR
    A[编写 YAML 配置] --> B[执行量化命令]
    B --> C[收集激活统计]
    C --> D[计算并融合平滑缩放]
    D --> E[验证量化结果]
```

## 5. 操作步骤

### 步骤 1：编写 YAML 配置文件

**目标**：编写包含 `smooth_quant` 处理器配置的 YAML 文件。

**操作**：

1. 在 `spec.process` 下配置 `smooth_quant` 处理器，指定 `type: "smooth_quant"`。
2. 按需配置 `alpha`（平滑强度，默认 `0.5`）、`symmetric`（是否对称量化，默认 `True`）。
3. 按需配置 `include`/`exclude` 通配符，控制参与平滑的层范围。

YAML 配置示例：

```yaml
spec:
  process:
    - type: "smooth_quant"                 # 固定为 `smooth_quant`，用于指定 Processor 类型。
      alpha: 0.5                           # 平衡参数，控制激活和权重的相对重要性，浮点数，0~1，默认0.5。
      symmetric: True                      # 是否启用对称量化，默认True，True为对称，False为非对称。
      include: ["*"]                       # 包含的层，支持通配符匹配，默认为["*"]（全量）。
      exclude: ["*self_attn*"]             # 排除的层，支持通配符匹配，默认为空。
```

**注意**：SmoothQuant 仅支持 `norm-linear` 子图类型，不支持其他子图类型（如 `ov`、`up-down`、`linear-linear`），因而不支持指定 `enable_subgraph_type` 字段。

YAML 配置字段详解如下：

| 字段名 | 作用 | 说明 |
| --- | --- | --- |
| type | 处理器类型标识 | 固定为 `"smooth_quant"`。 |
| alpha | 平衡参数 | 0~1 之间的浮点数，控制激活和权重的相对重要性，默认 `0.5`。 |
| symmetric | 是否对称量化 | 布尔值，`True` 为对称，`False` 为非对称，默认 `True`。 |
| include | 包含的层 | 字符串列表，支持通配符匹配，默认为 `["*"]`（全量）。 |
| exclude | 排除的层 | 字符串列表，支持通配符匹配，默认为空。 |

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

1. 工具加载 YAML 配置，解析 `smooth_quant` 处理器。
2. 预处理阶段为 `norm-linear` 子图中的线性模块安装前向钩子，收集激活统计信息。
3. 后处理阶段计算平滑缩放因子，对归一化层做反向缩放、对线性层做正向缩放。
4. 继续执行下游量化处理器并保存量化权重。

**输出**：量化权重目录 `${SAVE_PATH}`，包含量化描述文件与权重分片。

### 步骤 3：验证量化结果

**目标**：确认量化权重文件完整且可加载。

**操作**：

1. 检查输出目录是否包含 `quant_model_description.json` 文件。
2. 检查日志确认无层匹配告警（如 `patterns are not matched any module`）。
3. 使用推理框架加载量化权重进行冒烟测试。

**输出**：量化权重验证通过。

## 6. 验收条件

- 量化权重目录包含 `quant_model_description.json` 及所有必需的 `*.safetensors` 分片文件。
- 日志无子图配置错误或模块名不匹配告警。
- 量化后模型推理精度不低于未使用 SmoothQuant 的基线。

## 7. 异常处置

- **模块名称不匹配**：`include/exclude` 未命中时日志提示未匹配模式，核对完整模块名称是否与 `named_modules()` 返回的路径一致。
- **子图配置错误**：`get_adapter_config_for_subgraph()` 返回的配置不正确，检查 `source` 与 `targets` 字段。
- **模块不存在**：配置中指定的模块名称在模型中不存在，通过 `model.named_modules()` 验证。
- **映射关系错误**：确认 `source` 为归一化层、`targets` 为其后续线性层。

## 8. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| SmoothQuant | 通过协同缩放激活与权重抑制离群值的算法 | [SmoothQuant 词条](./term_smooth_quant.md) |
| 离群值抑制 | 通过数值变换减少激活离群值、降低量化误差 | [量化算法总览](../README.md) |
| norm-linear 子图 | 归一化层到多个线性层的映射结构 | [SmoothQuant 词条](./term_smooth_quant.md) |

## 9. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| `smooth_quant` 处理器 | 用于执行平滑量化的 Processor 配置 | [SmoothQuant 词条](./term_smooth_quant.md) |
| `SmoothQuantInterface` | 模型适配器需实现的离群值抑制接口 | [SmoothQuant 词条](./term_smooth_quant.md) |
