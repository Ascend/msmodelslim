# KV Smooth 使用指南

## 1. 适用范围

本流程适用于在 msModelSlim 中配置和使用 KV Smooth 缓存平滑算法。KV Smooth 作为离群值抑制算法，通常作为 KV Cache 量化的前置步骤，用于压缩 Key 的动态范围，提升缓存量化的精度。

适用角色：算法工程师、模型部署工程师

适用场景：

- 需要抑制 Key 离群值、提升 KV Cache 量化精度的场景。
- 长序列推理场景下，需要压缩 KV Cache 动态范围以降低显存占用的场景。

不适用场景：

- 注意力前向未接受或未使用 `past_key_values`/`past_key_value`。
- 目标通路不符合 `Linear/Norm → RoPE → KVCache` 结构。

## 2. 流程关系与前置条件

**上级流程**：模型适配与验证通过后，确定量化方案阶段。

**前置条件**：

- 已安装兼容版本的 msModelSlim 工具（详见《[msModelSlim 工具安装指南](../../../install_guide/install_guide.md)》）。
- 已确认目标模型适配器实现了 `KVSmoothFusedInterface` 接口，并正确配置融合子图。
- 已准备好校准数据集，用于推理标定观测抑制缩放尺度。

**后续操作**：量化流程执行 → 精度验证 → 部署上线或进入调优。

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型权重目录 | 模型下载或本地路径 | HuggingFace 格式，含 `config.json` 及 `*.safetensors` | 可被目标 Transformers 版本加载 |
| 输入 | 校准数据集 | 工具默认或用户指定 | JSONL 格式，每条含文本 prompt | 可被工具加载并完成前向推理 |
| 输入 | 量化 YAML 配置文件 | 用户编写 | 符合 msModelSlim YAML 规范，含 `kv_smooth` 处理器配置 | 可通过工具 `--config_path` 参数加载 |
| 交付件 | 平滑后的量化权重目录 | `--save_path` 指定路径 | 含 `quant_model_description.json` 及 `*.safetensors`，已应用缓存平滑 | 推理冒烟通过 |

## 4. 流程总览

```mermaid
flowchart LR
    A[编写 YAML 配置] --> B[执行量化命令]
    B --> C[观测 key_states]
    C --> D[计算并融合缩放]
    D --> E[验证量化结果]
```

## 5. 操作步骤

### 步骤 1：编写 YAML 配置文件

**目标**：编写包含 `kv_smooth` 处理器配置的 YAML 文件。

**操作**：

1. 在 `spec.process` 下配置 `kv_smooth` 处理器，指定 `type: "kv_smooth"`。
2. 按需配置 `smooth_factor`（默认 `1.0`，越大平滑越激进）。
3. 按需配置 `include`/`exclude` 通配符，控制参与平滑的注意力模块。

YAML 配置示例：

```yaml
spec:
  process:
    - type: "kv_smooth"
      smooth_factor: 1.0                    # 控制平滑激进程度，>0，越大平滑越激进
      include: ["*"]                        # 包含的层，支持通配符
      exclude: ["model.layers.0.self_attn"] # 排除的层，支持通配符
```

**注意**：`smooth_factor` 必须大于 0；`exclude` 的优先级高于 `include`。

YAML 配置字段详解如下：

| 字段名 | 作用 | 说明 |
| --- | --- | --- |
| type | 处理器类型标识 | 固定为 `"kv_smooth"`。 |
| smooth_factor | 平滑激进程度 | 大于 0 的浮点数，越大平滑越激进，默认 `1.0`。 |
| include | 包含的层 | 字符串列表，支持通配符匹配，默认为 `["*"]`。 |
| exclude | 排除的层 | 字符串列表，支持通配符匹配，默认为空，优先级高于 `include`。 |

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

1. 工具加载 YAML 配置，解析 `kv_smooth` 处理器。
2. 观察阶段（`preprocess`）注入观察器，在 `Cache.update()` 调用时捕获 `key_states` 并统计每通道最大绝对值。
3. 平滑阶段（`postprocess`）计算缩放向量并融合进 RoPE 之前的 Q/K 投影或归一化权重。
4. 继续执行下游量化处理器并保存量化权重。

**输出**：量化权重目录 `${SAVE_PATH}`，包含量化描述文件与权重分片。

### 步骤 3：验证量化结果

**目标**：确认量化权重文件完整且可加载。

**操作**：

1. 检查输出目录是否包含 `quant_model_description.json` 文件。
2. 检查日志确认无回退未命中或头维度信息缺失告警。
3. 使用推理框架加载量化权重进行冒烟测试。

**输出**：量化权重验证通过。

## 6. 验收条件

- 量化权重目录包含 `quant_model_description.json` 及所有必需的 `*.safetensors` 分片文件。
- 日志无 `past_key_values and past_key_value both are None` 或模块名不一致告警。
- 量化后模型推理精度不低于未使用 KV Smooth 的基线。

## 7. 异常处置

- **回退未命中**：告警日志出现 `are not matched any module`，核对完整模块名与 `include`/`exclude` 配置。
- **头维度信息缺失**：抛出 `UnsupportedError`，确认模型适配器实现了 `KVSmoothFusedInterface`。
- **注意力不适用**：日志告警 `past_key_values and past_key_value both are None`，检查注意力层 `forward` 是否传入 `past_key_values`。
- **模块名不一致**：抛出 `ToDoError`，确认 `fused_from_query_states_name`/`fused_from_key_states_name` 与实际子模块命名一致。

## 8. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| KV Smooth | 针对 KV Cache 量化的离群值抑制算法 | [KV Smooth 词条](./term_kv_smooth.md) |
| key_states | 写入 KV Cache 的 Key 状态 | [KV Smooth 词条](./term_kv_smooth.md) |
| 融合通路 | Linear/Norm → RoPE → KVCache 的平滑融合路径 | [KV Smooth 词条](./term_kv_smooth.md) |

## 9. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| `kv_smooth` 处理器 | 用于执行缓存平滑的 Processor 配置 | [KV Smooth 词条](./term_kv_smooth.md) |
| `KVSmoothFusedInterface` | 模型适配器需实现的缓存平滑接口 | [KV Smooth 词条](./term_kv_smooth.md) |
