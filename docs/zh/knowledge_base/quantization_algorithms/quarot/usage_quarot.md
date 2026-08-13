# QuaRot 使用指南

## 1. 适用范围

本流程适用于在 msModelSlim 中配置和使用 QuaRot 旋转量化算法。QuaRot 作为离群值抑制算法，通常作为量化前的预处理步骤，通过正交旋转平滑激活分布，提升低比特量化的精度。

适用角色：算法工程师、模型部署工程师

适用场景：

- 激活值存在显著离群值、需要低比特（如 W4A4）量化的大语言模型场景。
- 作为 AutoRound 等低比特量化算法的前置离群值抑制步骤。

不适用场景：

- 模型未实现 `QuaRotInterface` 接口。
- 启用在线旋转但模型未实现 `LAOSOnlineRotationInterface` 接口。

## 2. 流程关系与前置条件

**上级流程**：模型适配与验证通过后，确定量化方案阶段。

**前置条件**：

- 已安装兼容版本的 msModelSlim 工具（详见《[msModelSlim 工具安装指南](../../../install_guide/install_guide.md)》）。
- 已确认目标模型适配器实现了 `QuaRotInterface`（启用在线旋转时还需实现 `LAOSOnlineRotationInterface`）。
- 已确定下游量化方案（如 W4A4、W8A8）。

**后续操作**：量化流程执行 → 精度验证 → 部署上线或进入调优。

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型权重目录 | 模型下载或本地路径 | HuggingFace 格式，含 `config.json` 及 `*.safetensors` | 可被目标 Transformers 版本加载 |
| 输入 | 校准数据集 | 工具默认或用户指定 | JSONL 格式，每条含文本 prompt | 可被工具加载并完成前向推理 |
| 输入 | 量化 YAML 配置文件 | 用户编写 | 符合 msModelSlim YAML 规范，含 `quarot` 处理器配置 | 可通过工具 `--config_path` 参数加载 |
| 交付件 | 旋转后的量化权重目录 | `--save_path` 指定路径 | 含 `quant_model_description.json` 及 `*.safetensors`，已应用旋转矩阵 | 推理冒烟通过 |

## 4. 流程总览

```mermaid
flowchart LR
    A[编写 YAML 配置] --> B[执行量化命令]
    B --> C[获取旋转映射并融合]
    C --> D[逐层执行旋转]
    D --> E[验证量化结果]
```

## 5. 操作步骤

### 步骤 1：编写 YAML 配置文件

**目标**：编写包含 `quarot` 处理器配置的 YAML 文件。

**操作**：

1. 在 `spec.process` 下配置 `quarot` 处理器，指定 `type: "quarot"`。
2. 按需配置 `online`（默认 `False`）、`block_size`（默认 `-1`）、`max_tp_size`（默认 `4`）。
3. 启用在线旋转时配置 `down_proj_online_layers`；按需配置 `export_extra_info`（默认 `True`）。

YAML 配置示例：

```yaml
spec:
  process:
    - type: "quarot"                     # 固定为 `quarot`，用于指定 Processor 类型。
      online: False                      # 控制是否启用在线旋转，默认为 False。
      block_size: -1                     # 旋转矩阵启用块对角矩阵时每个块的大小, 取值范围为-1或2的幂次方，如果大于0必须为2的幂，若为-1，表示不进行块对角矩阵处理
      max_tp_size: 4                     # 最大张量并行大小，默认为4，仅在启用在线旋转时生效，必须大于0且为2的幂
      down_proj_online_layers: [ ]       # 用于指定哪些层的down_proj使用在线旋转，默认为空
      export_extra_info: True            # 是否导出全局旋转矩阵：使用 ascend_v1_saver 时在量化权重路径下生成 optional 目录及 quant_model_description.json 追加字段，默认为 True
```

YAML 配置字段详解如下：

| 字段名 | 作用 | 说明 |
| --- | --- | --- |
| type | 处理器类型标识 | 固定为 `"quarot"`。 |
| online | 在线旋转开关 | 布尔值，`True` 启用在线旋转，默认 `False`。 |
| block_size | 旋转矩阵对角块大小 | `-1` 或大于 0 的 2 的幂，`-1` 表示不进行块对角矩阵处理。 |
| max_tp_size | 最大张量并行大小 | 仅在线旋转时生效，大于 0 且为 2 的幂或等于 1，默认 `4`。 |
| down_proj_online_layers | 使用在线旋转的 down 层 | 层索引列表，指定哪些层的 `down_proj` 使用在线旋转，默认 `[]`。 |
| export_extra_info | 是否导出全局旋转信息 | 默认 `True`，为 `True` 时导出 `optional/quarot.safetensors` 并追加描述字段。 |

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

1. 工具加载 YAML 配置，解析 `quarot` 处理器。
2. pre_run 阶段获取旋转映射并执行层融合与旋转。
3. preprocess 阶段逐层执行旋转操作（启用在线旋转时注入旋转算子）。
4. 继续执行下游量化处理器并保存量化权重。

**输出**：量化权重目录 `${SAVE_PATH}`，包含量化描述文件与权重分片。

### 步骤 3：验证量化结果

**目标**：确认量化权重文件完整且可加载。

**操作**：

1. 检查输出目录是否包含 `quant_model_description.json` 文件。
2. 检查日志确认无旋转矩阵创建失败或张量并行配置错误告警。
3. 使用推理框架加载量化权重进行冒烟测试。

**输出**：量化权重验证通过。

## 6. 验收条件

- 量化权重目录包含 `quant_model_description.json` 及所有必需的 `*.safetensors` 分片文件。
- 启用 `export_extra_info: True` 时，`optional/quarot.safetensors` 生成且描述文件中包含 `optional.quarot` 域。
- 量化后模型推理精度不低于未使用 QuaRot 的基线。

## 7. 异常处置

- **旋转矩阵创建失败**：提示输入模型的维度暂未被支持，确认指定维度的 Hadamard 矩阵存在，必要时在 `hadamard_txt` 中补充。
- **张量并行配置错误**：推理引擎 TP 部署时精度异常，确认 `tp_size` 为 2 的幂且小于等于 `max_tp_size`。
- **在线旋转性能问题**：启用在线旋转后推理性能下降，可考虑关闭在线旋转（`online: False`）或仅对部分层使用在线旋转。
- **模型适配失败**：确认适配器正确实现所有 `QuaRotInterface` 接口方法。

## 8. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| QuaRot | 基于正交旋转的离群值抑制算法 | [QuaRot 词条](./term_quarot.md) |
| Hadamard 矩阵 | 用于旋转的正交矩阵 | [QuaRot 词条](./term_quarot.md) |
| 在线旋转 | 推理时动态注入旋转算子的模式 | [QuaRot 词条](./term_quarot.md) |

## 9. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| `quarot` 处理器 | 用于执行旋转量化的 Processor 配置 | [QuaRot 词条](./term_quarot.md) |
| `QuaRotInterface` | 模型适配器需实现的基础旋转接口 | [QuaRot 词条](./term_quarot.md) |
| `LAOSOnlineRotationInterface` | 模型适配器需实现的在线旋转接口 | [QuaRot 词条](./term_quarot.md) |
