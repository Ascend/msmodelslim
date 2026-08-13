# Flex AWQ SSZ 使用指南

## 1. 适用范围

本流程适用于在 msModelSlim 中配置和使用 Flex AWQ SSZ 灵活激活感知权重量化平滑算法。Flex AWQ SSZ 作为离群值抑制算法，通常作为量化前的预处理步骤，通过真实量化器评估参数、自动搜索最优 `alpha` 抑制激活离群值，提升低比特量化的精度。

适用角色：算法工程师、模型部署工程师

适用场景：

- 需要对平滑参数进行真实量化器评估、追求更高精度的低比特量化场景。
- 需要同时对注意力 `ov`、MLP `up-down`、连续线性层等多种结构做离群值抑制的场景。

不适用场景：

- 模型未实现 `FlexSmoothQuantInterface` 接口，或无子图映射配置。
- 未配置 `qconfig`（必填参数）。

## 2. 流程关系与前置条件

**上级流程**：模型适配与验证通过后，确定量化方案阶段。

**前置条件**：

- 已安装兼容版本的 msModelSlim 工具（详见《[msModelSlim 工具安装指南](../../../install_guide/install_guide.md)》）。
- 已确认目标模型适配器实现了 `FlexSmoothQuantInterface` 接口，并正确配置子图映射。
- 已确定下游量化方案（如 W4A4、W8A8），以便配置 `qconfig` 与 `alpha`。
- 已准备好校准数据集，用于收集激活统计信息。

**后续操作**：量化流程执行 → 精度验证 → 部署上线或进入调优。

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型权重目录 | 模型下载或本地路径 | HuggingFace 格式，含 `config.json` 及 `*.safetensors` | 可被目标 Transformers 版本加载 |
| 输入 | 校准数据集 | 工具默认或用户指定 | JSONL 格式，每条含文本 prompt | 可被工具加载并完成前向推理 |
| 输入 | 量化 YAML 配置文件 | 用户编写 | 符合 msModelSlim YAML 规范，含 `flex_awq_ssz` 处理器及 `qconfig` 配置 | 可通过工具 `--config_path` 参数加载 |
| 交付件 | 平滑后的量化权重目录 | `--save_path` 指定路径 | 含 `quant_model_description.json` 及 `*.safetensors`，已应用平滑缩放 | 推理冒烟通过 |

## 4. 流程总览

```mermaid
flowchart LR
    A[编写 YAML 配置] --> B[执行量化命令]
    B --> C[收集激活均值]
    C --> D[搜索并融合缩放]
    D --> E[验证量化结果]
```

## 5. 操作步骤

### 步骤 1：编写 YAML 配置文件

**目标**：编写包含 `flex_awq_ssz` 处理器与 `qconfig` 配置的 YAML 文件。

**操作**：

1. 在 `spec.process` 下配置 `flex_awq_ssz` 处理器，指定 `type: "flex_awq_ssz"`。
2. 配置必填的 `qconfig`，包含 `act` 与 `weight` 的量化配置（权重通常使用 `method: "ssz"`）。
3. 可选配置 `alpha`（默认自动搜索）以及 `enable_subgraph_type`、`include`/`exclude`。

YAML 配置示例：

```yaml
spec:
  process:
    - type: "flex_awq_ssz"                # 固定为 `flex_awq_ssz`，用于指定 Processor。
      alpha: 0.8                          # 激活缩放的系数，取值范围为0-1之间，默认值为None（自动搜索），也支持用户自行配置。
      qconfig:                            # 量化配置，为必填参数。
        act:                              # 激活值量化配置。
          scope: "per_token"              # 量化范围：per_token 或 per_tensor。
          dtype: "int8"                   # 量化数据类型：int8。
          symmetric: True                 # 是否对称量化：True 或 False。
          method: "minmax"                # 量化方法：minmax 或其他方法。
        weight:                           # 权重量化配置。
          scope: "per_channel"            # 量化范围：per_channel。
          dtype: "int4"                   # 量化数据类型：int4 或 int8。
          symmetric: True                 # 是否对称量化：True。
          method: "ssz"                   # 量化方法：ssz（Smooth Scale Zero）。
          ext:                            # 扩展配置（可选）。
            step: 10                      # SSZ方法的步长参数。
      enable_subgraph_type:               # 字符串列表，指定启用的子图类型，默认启用所有四种类型。
        - 'norm-linear'
        - 'linear-linear'
        - 'ov'
        - 'up-down'
      include: ["*"]                      # 包含的层，支持通配符。
      exclude: ["*self_attn*"]            # 排除的层，支持通配符。
```

YAML 配置字段详解如下：

| 字段名 | 作用 | 说明 |
| --- | --- | --- |
| type | 处理器类型标识 | 固定为 `"flex_awq_ssz"`。 |
| alpha | 激活缩放权重系数 | 0~1 之间的浮点数，默认 `None`（自动搜索）。 |
| qconfig | 量化配置 | 必填参数，包含激活值（`act`）和权重（`weight`）的量化配置，用于真实量化器评估。 |
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

1. 工具加载 YAML 配置，解析 `flex_awq_ssz` 处理器与 `qconfig`。
2. 预处理阶段发现子图并收集激活统计信息。
3. 后处理阶段使用真实量化器搜索最优 `alpha` 并融合缩放。
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
- 日志无 `qconfig` 缺失或模块名不匹配告警。
- 量化后模型推理精度不低于未使用 Flex AWQ SSZ 的基线。

## 7. 异常处置

- **qconfig 配置缺失**：报错提示 `qconfig` 为必填参数，在 YAML 配置中添加 `qconfig` 字段（含 `act` 与 `weight`）。
- **模块名不匹配**：`include/exclude` 未命中时日志提示未匹配模式，核对完整模块名是否与 `named_modules()` 返回的路径一致。
- **子图类型不支持**：确保配置的子图类型在支持列表中（`norm-linear`、`linear-linear`、`ov`、`up-down`）。
- **映射关系错误**：检查 `MappingConfig` 中的 `source` 与 `targets` 是否指向正确的模块。

## 8. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| Flex AWQ SSZ | 结合 AWQ 与 SSZ、使用真实量化器评估参数的平滑算法 | [Flex AWQ SSZ 词条](./term_flex_awq_ssz.md) |
| qconfig | 激活与权重的量化配置，供真实量化器评估使用 | [Flex AWQ SSZ 词条](./term_flex_awq_ssz.md) |
| 离群值抑制 | 通过数值变换减少激活离群值、降低量化误差 | [量化算法总览](../README.md) |

## 9. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| `flex_awq_ssz` 处理器 | 用于执行灵活激活感知平滑的 Processor 配置 | [Flex AWQ SSZ 词条](./term_flex_awq_ssz.md) |
| `FlexSmoothQuantInterface` | 模型适配器需实现的平滑接口（与 Flex Smooth Quant 相同） | [Flex AWQ SSZ 词条](./term_flex_awq_ssz.md) |
