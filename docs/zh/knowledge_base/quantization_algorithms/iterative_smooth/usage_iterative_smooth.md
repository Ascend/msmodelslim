# Iterative Smooth 使用指南

## 1. 适用范围

本流程适用于在 msModelSlim 中配置和使用 Iterative Smooth 迭代平滑算法。Iterative Smooth 作为离群值抑制算法，通常作为量化前的预处理步骤，用于将激活离群值迁移到权重，提升低比特量化的精度。

适用角色：算法工程师、模型部署工程师

适用场景：

- 需要同时处理注意力 `ov`、MLP `up-down`、连续线性层等多种子图结构的场景。
- 需要对无前置融合层的独立线性层做平滑（非融合子图）的场景。

不适用场景：

- 模型未实现 `IterSmoothInterface` 接口，或无子图映射配置。
- 目标模块无 `weight` 属性或模块名无法通过 `named_modules()` 定位。

## 2. 流程关系与前置条件

**上级流程**：模型适配与验证通过后，确定量化方案阶段。

**前置条件**：

- 已安装兼容版本的 msModelSlim 工具（详见《[msModelSlim 工具安装指南](../../../install_guide/install_guide.md)》）。
- 已确认目标模型适配器实现了 `IterSmoothInterface` 接口，并正确配置子图映射。
- 已准备好校准数据集，用于收集激活统计信息。

**后续操作**：量化流程执行 → 精度验证 → 部署上线或进入调优。

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型权重目录 | 模型下载或本地路径 | HuggingFace 格式，含 `config.json` 及 `*.safetensors` | 可被目标 Transformers 版本加载 |
| 输入 | 校准数据集 | 工具默认或用户指定 | JSONL 格式，每条含文本 prompt | 可被工具加载并完成前向推理 |
| 输入 | 量化 YAML 配置文件 | 用户编写 | 符合 msModelSlim YAML 规范，含 `iter_smooth` 处理器配置 | 可通过工具 `--config_path` 参数加载 |
| 交付件 | 平滑后的量化权重目录 | `--save_path` 指定路径 | 含 `quant_model_description.json` 及 `*.safetensors`，已应用平滑缩放 | 推理冒烟通过 |

## 4. 流程总览

```mermaid
flowchart LR
    A[编写 YAML 配置] --> B[执行量化命令]
    B --> C[子图发现与统计]
    C --> D[按优先级平滑]
    D --> E[验证量化结果]
```

## 5. 操作步骤

### 步骤 1：编写 YAML 配置文件

**目标**：编写包含 `iter_smooth` 处理器配置的 YAML 文件。

**操作**：

1. 在 `spec.process` 下配置 `iter_smooth` 处理器，指定 `type: "iter_smooth"`。
2. 按需配置 `alpha`（默认 `0.9`）、`scale_min`（默认 `1e-5`）、`symmetric`（默认 `True`）。
3. 按需配置 `enable_subgraph_type` 与 `include`/`exclude`，控制参与平滑的子图与层范围。

YAML 配置示例：

```yaml
spec:
  process:
    - type: "iter_smooth"                  # 固定为 `iter_smooth`，用于指定 Processor。
      alpha: 0.9                           # 平衡参数，控制激活和权重的相对重要性，默认0.9。
      scale_min: 1e-5                      # 缩放因子的最小值，防止数值不稳定，默认1e-5。
      symmetric: True                      # 是否启用对称量化，默认True。
      enable_subgraph_type:                # 开启的子图类型。
        - 'norm-linear'
        - 'linear-linear'
        - 'ov'
        - 'up-down'
      include: ["*"]                       # 包含的层，支持通配符。
      exclude: ["*self_attn*"]             # 排除的层，支持通配符。
```

YAML 配置字段详解如下：

| 字段名 | 作用 | 说明 |
| --- | --- | --- |
| type | 处理器类型标识 | 固定为 `"iter_smooth"`。 |
| alpha | 平衡参数 | 大于 0 的浮点数，控制激活和权重的相对重要性，默认 `0.9`。 |
| scale_min | 缩放因子最小值 | 大于 0 的浮点数，防止数值不稳定，默认 `1e-5`。 |
| symmetric | 是否对称量化 | 布尔值，`True` 为对称，`False` 为非对称，默认 `True`。 |
| enable_subgraph_type | 开启的子图类型 | 支持 `norm-linear`、`linear-linear`、`ov`、`up-down`。 |
| include | 包含的层 | 字符串列表，支持通配符匹配。 |
| exclude | 排除的层 | 字符串列表，支持通配符匹配。 |

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

1. 工具加载 YAML 配置，解析 `iter_smooth` 处理器。
2. 预处理阶段发现子图并收集激活统计信息。
3. 后处理阶段按 `up-down` → `ov` → `norm-linear` → `linear-linear` 优先级平滑各子图。
4. 继续执行下游量化处理器并保存量化权重。

**输出**：量化权重目录 `${SAVE_PATH}`，包含量化描述文件与权重分片。

### 步骤 3：验证量化结果

**目标**：确认量化权重文件完整且可加载。

**操作**：

1. 检查输出目录是否包含 `quant_model_description.json` 文件。
2. 检查日志确认无层匹配告警。
3. 使用推理框架加载量化权重进行冒烟测试。

**输出**：量化权重验证通过。

## 6. 验收条件

- 量化权重目录包含 `quant_model_description.json` 及所有必需的 `*.safetensors` 分片文件。
- 日志无子图配置错误或模块名不匹配告警。
- 量化后模型推理精度不低于未使用 Iterative Smooth 的基线。

## 7. 异常处置

- **模块名不匹配**：`include/exclude` 未命中时日志提示未匹配模式，核对完整模块名是否与 `named_modules()` 返回的路径一致。
- **子图类型不支持**：确保配置的子图类型在 `ENABLE_SUBGRAPH_TYPES` 列表中。
- **非融合子图不生效**：确认 `mapping.source` 显式传入 `None` 且 `mapping.targets` 非空，并确认目标模块在 `include` 范围内且未被 `exclude` 过滤。
- **映射关系错误**：检查 `MappingConfig` 中的 `source` 与 `targets` 是否指向正确的模块。

## 8. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| Iterative Smooth | 支持多子图类型的迭代平滑离群值抑制算法 | [Iterative Smooth 词条](./term_iterative_smooth.md) |
| 子图类型 | norm-linear、linear-linear、ov、up-down 等可平滑的结构 | [Iterative Smooth 词条](./term_iterative_smooth.md) |
| 非融合子图 | `source=None`、仅对若干独立线性层做平滑的结构 | [Iterative Smooth 词条](./term_iterative_smooth.md) |

## 9. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| `iter_smooth` 处理器 | 用于执行迭代平滑的 Processor 配置 | [Iterative Smooth 词条](./term_iterative_smooth.md) |
| `IterSmoothInterface` | 模型适配器需实现的迭代平滑接口 | [Iterative Smooth 词条](./term_iterative_smooth.md) |
