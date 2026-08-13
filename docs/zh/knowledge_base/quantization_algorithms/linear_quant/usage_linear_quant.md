# 线性量化使用指南

## 1. 适用范围

本流程适用于在 msModelSlim 中配置和使用线性量化算法。线性量化通过 `linear_quant` 处理器对模型的线性层（`nn.Linear`）权重与激活进行量化，是大多数量化方案的基础。

适用角色：算法工程师、模型部署工程师

适用场景：

- 基础量化场景，需要对线性层权重与激活进行量化。
- 作为 MinMax、Histogram、SSZ、GPTQ、PDMIX 等量化算法的宿主处理器。

不适用场景：

- 非 `nn.Linear` 模块的量化目标。
- 需要动态计算量化参数以外特殊量化策略的场景（应选择对应的算法）。

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
| 输入 | 量化 YAML 配置文件 | 用户编写 | 符合 msModelSlim YAML 规范，含 `linear_quant` 处理器配置 | 可通过工具 `--config_path` 参数加载 |
| 交付件 | 量化权重目录 | `--save_path` 指定路径 | 含 `quant_model_description.json` 及 `*.safetensors`，已应用线性量化 | 推理冒烟通过 |

## 4. 流程总览

```mermaid
flowchart LR
    A[编写 YAML 配置] --> B[执行量化命令]
    B --> C[统计量化参数]
    C --> D[量化并部署]
    D --> E[验证量化结果]
```

## 5. 操作步骤

### 步骤 1：编写 YAML 配置文件

**目标**：编写包含 `linear_quant` 处理器配置的 YAML 文件。

**操作**：

1. 在 `spec.process` 下配置 `linear_quant` 处理器，指定 `type: "linear_quant"`。
2. 配置 `qconfig.act`（激活值量化配置）与 `qconfig.weight`（权重量化配置），指定 `scope`、`dtype`、`symmetric`、`method`。
3. 按需配置 `include`/`exclude` 通配符控制量化层范围。

YAML 配置示例（W8A8 动态量化）：

```yaml
spec:
  process:
    - type: "linear_quant"         # 处理器类型：线性层量化
      qconfig:
        act:                       # 激活值量化配置
          scope: "per_token"       # 动态量化标识：每个 token 独立量化参数
          dtype: "int8"            # 数据类型：int8
          symmetric: false         # 是否对称量化：false
          method: "minmax"         # 量化方法：minmax
        weight:                    # 权重量化配置
          scope: "per_channel"     # 权重量化粒度：逐通道量化
          dtype: "int8"            # 数据类型：int8
          symmetric: true          # 对称量化：true
          method: "minmax"         # 量化方法：minmax
      include: ["*"]               # 包含的层。默认：["*"]
      exclude: ["*down_proj*"]     # 排除的层。默认：[]
```

YAML 配置字段详解如下：

| 字段名 | 作用 | 说明 |
| --- | --- | --- |
| type | 处理器类型标识 | 固定为 `"linear_quant"`。 |
| qconfig.act.scope | 激活量化范围 | `"per_tensor"`（静态）、`"per_token"`（动态）、`"pd_mix"`（PDMIX 混合）。 |
| qconfig.act.dtype | 激活量化数据类型 | `"int8"`、`"int4"`、`"float"`（16位浮点激活）。 |
| qconfig.act.symmetric | 激活是否对称量化 | `true` 为对称，`false` 为非对称。 |
| qconfig.act.method | 激活量化方法 | `"minmax"` 或 `"histogram"`。 |
| qconfig.weight.scope | 权重量化范围 | `"per_tensor"`、`"per_channel"`、`"per_group"`。 |
| qconfig.weight.dtype | 权重量化数据类型 | `"int8"` 或 `"int4"`。 |
| qconfig.weight.symmetric | 权重是否对称量化 | `true` 为对称，`false` 为非对称。 |
| qconfig.weight.method | 权重量化方法 | `"minmax"`、`"ssz"`、`"gptq"`。 |
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

1. 工具加载 YAML 配置，解析 `linear_quant` 处理器与 `qconfig`。
2. 静态量化收集激活统计并固定量化参数；动态量化在推理时在线计算。
3. 对命中 `include`/`exclude` 规则的层执行量化并生成量化 IR。
4. 保存量化权重。

**输出**：量化权重目录 `${SAVE_PATH}`，包含量化描述文件与权重分片。

### 步骤 3：验证量化结果

**目标**：确认量化权重文件完整且可加载。

**操作**：

1. 检查输出目录是否包含 `quant_model_description.json` 文件。
2. 检查日志确认无层匹配告警或无效配置组合告警。
3. 使用推理框架加载量化权重进行冒烟测试。

**输出**：量化权重验证通过。

## 6. 验收条件

- 量化权重目录包含 `quant_model_description.json` 及所有必需的 `*.safetensors` 分片文件。
- 日志无 `patterns are not matched any module` 层匹配告警。
- 量化后模型推理精度符合业务要求。

## 7. 异常处置

- **量化组合无效**：检测到无效的量化配置组合时抛出 `UnsupportedError`，根据异常信息调整配置参数。
- **层匹配告警**：`include`/`exclude` 未匹配到任何层时工具告警，检查层名、路径层级、大小写与拼写。
- **精度下降**：动态量化（`per_token`）精度通常优于静态量化（`per_tensor`），可按需调整；或配合离群值抑制算法使用。

## 8. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| 线性量化 | 将浮点数值映射到离散整数的量化方法 | [线性量化词条](./term_linear_quant.md) |
| 静态量化 | 推理前固定量化参数的量化方式 | [线性量化词条](./term_linear_quant.md) |
| 动态量化 | 推理时在线计算量化参数的量化方式 | [线性量化词条](./term_linear_quant.md) |

## 9. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| `linear_quant` 处理器 | 用于执行线性层量化的 Processor 配置 | [线性量化词条](./term_linear_quant.md) |
| 层过滤机制 | include/exclude 通配符匹配规则 | [线性量化词条](./term_linear_quant.md) |
| 量化方法 | MinMax、Histogram、SSZ、GPTQ 等 `method` 取值 | [线性量化词条](./term_linear_quant.md) |
