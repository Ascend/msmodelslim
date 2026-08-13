# PDMIX 使用指南

## 1. 适用范围

本流程适用于在 msModelSlim 中配置和使用 PDMIX 激活值阶段间混合量化算法。PDMIX 作为 `linear_quant` 处理器的激活值量化方法，在 Prefilling 阶段使用动态量化、Decoding 阶段使用静态量化，用于平衡精度与推理性能。

适用角色：算法工程师、模型部署工程师

适用场景：

- 长上下文或分布漂移场景下，静态量化精度损失大、需要回退大量层的场景。
- 生成式模型推理加速场景，希望在控制精度损失的同时获取静态量化的性能收益。

不适用场景：

- 非 MindIE 推理部署场景。
- 非 W8A8 PDMIX 模式的其他量化配置组合。

## 2. 流程关系与前置条件

**上级流程**：模型适配与验证通过后，确定量化方案阶段。

**前置条件**：

- 已安装兼容版本的 msModelSlim 工具（详见《[msModelSlim 工具安装指南](../../../install_guide/install_guide.md)》）。
- 已确认目标模型由 `torch.nn.Linear` 实现，且推理部署环境为 MindIE。
- 已准备好校准数据集，用于计算 Decoding 阶段的静态量化参数。

**后续操作**：量化流程执行 → 精度验证 → 部署上线或进入调优。

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型权重目录 | 模型下载或本地路径 | HuggingFace 格式，含 `config.json` 及 `*.safetensors` | 可被目标 Transformers 版本加载 |
| 输入 | 校准数据集 | 工具默认或用户指定 | JSONL 格式，每条含文本 prompt | 可被工具加载并完成前向推理 |
| 输入 | 量化 YAML 配置文件 | 用户编写 | 符合 msModelSlim YAML 规范，含 `scope: "pd_mix"` 激活配置 | 可通过工具 `--config_path` 参数加载 |
| 交付件 | 量化权重目录 | `--save_path` 指定路径 | 含 `quant_model_description.json` 及 `*.safetensors`，已应用 PDMIX 量化 | 推理冒烟通过 |

## 4. 流程总览

```mermaid
flowchart LR
    A[编写 YAML 配置] --> B[执行量化命令]
    B --> C[计算静态量化参数]
    C --> D[部署 PDMIX IR]
    D --> E[验证量化结果]
```

## 5. 操作步骤

### 步骤 1：编写 YAML 配置文件

**目标**：编写包含 PDMIX 激活值量化配置的 YAML 文件。

**操作**：

1. 在 `spec.process` 下配置 `linear_quant` 处理器。
2. 在 `qconfig.act` 中设置 `scope: "pd_mix"`、`dtype: "int8"`、`symmetric: false`、`method: "minmax"`。
3. 在 `qconfig.weight` 中配置权重 per_channel INT8 对称量化。

YAML 配置示例：

```yaml
spec:
  process:
    - type: "linear_quant"     # 线性层量化模式处理器
      qconfig:
        act: # 激活值量化配置
          scope: "pd_mix"      # prefilling: per-token；decoding: per-tensor
          dtype: "int8"        # 暂时仅支持 INT8
          symmetric: false     # PDMIX 量化总体为非对称
          method: "minmax"     # 暂时仅支持 MinMax 算法
        weight: # 权重量化配置
          scope: "per_channel" # 暂时仅支持搭配权重 per_channel 量化
          dtype: "int8"        # 暂时仅支持搭配权重 INT8 量化
          symmetric: true      # 仅支持搭配权重对称量化
          method: "minmax"     # 权重量化算法
```

YAML 配置字段详解如下：

| 字段名 | 作用 | 说明 |
| --- | --- | --- |
| qconfig.act.scope | 激活量化范围 | 固定为 `"pd_mix"`（prefilling 用 `per_token`，decoding 用 `per_tensor`）。 |
| qconfig.act.dtype | 激活量化数据类型 | 仅支持 `"int8"`。 |
| qconfig.act.symmetric | 激活是否对称量化 | `false`（PDMIX 量化总体为非对称）。 |
| qconfig.act.method | 激活量化方法 | 仅支持 `"minmax"`。 |
| qconfig.weight.scope | 权重量化范围 | 仅支持 `"per_channel"`。 |
| qconfig.weight.dtype | 权重量化数据类型 | 仅支持 `"int8"`。 |
| qconfig.weight.symmetric | 权重是否对称量化 | 仅支持 `true`。 |
| qconfig.weight.method | 权重量化方法 | `"minmax"`。 |

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

1. 工具加载 YAML 配置，解析 `linear_quant` 处理器与 PDMIX 激活配置。
2. 校准阶段计算 Decoding 阶段使用的静态量化参数。
3. 部署 `W8A8PDMixFakeQuantLinear` IR，Prefilling 使用 per-token 动态量化、Decoding 使用 per-tensor 静态量化。
4. 保存量化权重。

**输出**：量化权重目录 `${SAVE_PATH}`，包含量化描述文件与权重分片。

### 步骤 3：验证量化结果

**目标**：确认量化权重文件完整且可加载。

**操作**：

1. 检查输出目录是否包含 `quant_model_description.json` 文件。
2. 检查日志确认量化流程正常完成。
3. 在 MindIE 推理环境中加载量化权重进行冒烟测试。

**输出**：量化权重验证通过。

## 6. 验收条件

- 量化权重目录包含 `quant_model_description.json` 及所有必需的 `*.safetensors` 分片文件。
- 日志无无效配置组合告警。
- 量化后模型推理精度与性能符合业务要求。

## 7. 异常处置

- **配置组合无效**：除 `qconfig.weight.method` 外，其他配置组合未有对应实现，检查 YAML 配置与 PDMIX 支持范围是否一致。
- **部署环境不支持**：确认推理部署环境为 MindIE。
- **精度下降**：确认权重量化方式与 PDMIX 匹配（per_channel INT8 对称），或检查校准数据质量。

## 8. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| PDMIX | Prefilling 动态、Decoding 静态的激活值混合量化算法 | [PDMIX 词条](./term_pdmix.md) |
| per_token | 每个 token 独立量化参数（动态量化） | [PDMIX 词条](./term_pdmix.md) |
| per_tensor | 整个张量共用量化参数（静态量化） | [PDMIX 词条](./term_pdmix.md) |

## 9. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| `linear_quant` 处理器 | 线性层量化处理器，通过 `scope: "pd_mix"` 启用 PDMIX | [线性量化词条](../linear_quant/term_linear_quant.md) |
| `ActPDMixMinmax` | PDMIX 激活值量化校准实现 | [PDMIX 词条](./term_pdmix.md) |
