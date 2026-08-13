# AWQ 使用指南

## 1. 适用范围

本流程适用于在 msModelSlim 中配置和使用 AWQ 激活感知权重量化算法。AWQ 作为离群值抑制算法，通常作为权重量化前的预处理步骤，通过激活感知搜索最优缩放因子，提升低比特量化的精度。

适用角色：算法工程师、模型部署工程师

适用场景：

- 需要自动搜索最优权重缩放因子、保护重要通道的低比特量化场景。
- 作为权重量化的前置步骤，为 MinMax、SSZ 等权重量化算法提供更优的权重分布。

不适用场景：

- 模型适配器未实现 `AWQInterface` 接口。
- 目标模块无 `weight` 属性或模块名无法通过 `named_modules()` 定位。

## 2. 流程关系与前置条件

**上级流程**：模型适配与验证通过后，确定量化方案阶段。

**前置条件**：

- 已安装兼容版本的 msModelSlim 工具（详见《[msModelSlim 工具安装指南](../../../install_guide/install_guide.md)》）。
- 已确认目标模型适配器实现了 `AWQInterface` 接口，并正确配置子图映射。
- 已确定下游权重量化方案（如 INT8、INT4），以便配置 `weight_qconfig`。
- 已准备好校准数据集，用于收集激活统计信息。

**后续操作**：量化流程执行 → 精度验证 → 部署上线或进入调优。

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型权重目录 | 模型下载或本地路径 | HuggingFace 格式，含 `config.json` 及 `*.safetensors` | 可被目标 Transformers 版本加载 |
| 输入 | 校准数据集 | 工具默认或用户指定 | JSONL 格式，每条含文本 prompt | 可被工具加载并完成前向推理 |
| 输入 | 量化 YAML 配置文件 | 用户编写 | 符合 msModelSlim YAML 规范，含 `awq` 处理器配置 | 可通过工具 `--config_path` 参数加载 |
| 交付件 | 平滑后的量化权重目录 | `--save_path` 指定路径 | 含 `quant_model_description.json` 及 `*.safetensors`，已应用最优缩放 | 推理冒烟通过 |

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

**目标**：编写包含 `awq` 处理器配置的 YAML 文件。

**操作**：

1. 在 `spec.process` 下配置 `awq` 处理器，指定 `type: "awq"`。
2. 配置 `weight_qconfig`（AWQ 搜索阶段使用的权重量化配置）。
3. 按需配置 `n_grid`（默认 `20`）、`enable_subgraph_type` 与 `include`/`exclude`。

YAML 配置示例：

```yaml
spec:
  process:
    - type: "awq"
      weight_qconfig:
        scope: "per_channel"
        dtype: "int8"
        symmetric: true
        method: "minmax"
      n_grid: 20
      enable_subgraph_type:
        - "norm-linear"
        - "linear-linear"
        - "ov"
        - "up-down"
      include: ["*"]
      exclude: []
```

YAML 配置字段详解如下：

| 字段名 | 作用 | 说明 |
| --- | --- | --- |
| type | 处理器类型标识 | 固定为 `"awq"`。 |
| weight_qconfig | 权重量化配置 | AWQ 搜索阶段使用的权重量化配置，字段定义与 `linear_quant` 的 `qconfig.weight` 一致。 |
| n_grid | 网格搜索步数 | 正整数，默认 `20`，数值越大搜索越细致但耗时增加。 |
| enable_subgraph_type | 启用的子图类型 | 支持 `norm-linear`、`linear-linear`、`ov`、`up-down`。 |
| include | 包含的层 | 字符串列表，支持通配符匹配。 |
| exclude | 排除的层 | 字符串列表，支持通配符匹配，优先级高于 `include`。 |

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

1. 工具加载 YAML 配置，解析 `awq` 处理器。
2. 预处理阶段为目标线性层安装 forward hook，收集激活均值并缓存祖先模块输入。
3. 后处理阶段按子图优先级搜索最优缩放因子，通过 `SubgraphFusionFactory` 融合。
4. 继续执行下游量化处理器并保存量化权重。

**输出**：量化权重目录 `${SAVE_PATH}`，包含量化描述文件与权重分片。

### 步骤 3：验证量化结果

**目标**：确认量化权重文件完整且可加载。

**操作**：

1. 检查输出目录是否包含 `quant_model_description.json` 文件。
2. 检查日志确认无层匹配告警或祖先模块未找到告警。
3. 使用推理框架加载量化权重进行冒烟测试。

**输出**：量化权重验证通过。

## 6. 验收条件

- 量化权重目录包含 `quant_model_description.json` 及所有必需的 `*.safetensors` 分片文件。
- 日志无激活统计信息缺失或中间参数缓存为空告警。
- 量化后模型推理精度不低于未使用 AWQ 的基线。

## 7. 异常处置

- **模块名不匹配**：`include/exclude` 未命中时日志提示未匹配模式，核对完整模块名。
- **祖先模块未找到**：日志提示 `No name found for inspect module of subgraph`，检查 `targets` 中的模块是否具有合理的共同路径前缀。
- **激活统计信息缺失**：日志提示 `No activation mean for target module`，确保校准数据足够且前向推理正常。
- **中间参数缓存为空**：日志提示 `No kwargs cache for parent module`，检查 LCA 发现的祖先模块是否被正确触发。

## 8. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| AWQ | 激活感知权重量化，基于激活均值搜索最优缩放因子 | [AWQ 词条](./term_awq_smooth.md) |
| weight_qconfig | AWQ 搜索阶段使用的权重量化配置 | [AWQ 词条](./term_awq_smooth.md) |
| 最低公共祖先（LCA） | 用于块级误差评估的祖先模块 | [AWQ 词条](./term_awq_smooth.md) |

## 9. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| `awq` 处理器 | 用于执行激活感知权重量化的 Processor 配置 | [AWQ 词条](./term_awq_smooth.md) |
| `AWQInterface` | 模型适配器需实现的激活感知量化接口 | [AWQ 词条](./term_awq_smooth.md) |
