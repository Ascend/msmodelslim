# KVCache Quant 使用指南

## 1. 适用范围

本流程适用于在 msModelSlim 中配置和使用 KVCache Quant 缓存量化算法。KVCache Quant 作为缓存量化处理器，对写入 KV Cache 的 Key/Value 状态进行 INT8 量化，用于降低缓存显存占用、提升长序列推理效率。

适用角色：算法工程师、模型部署工程师

适用场景：

- 长序列推理场景下需要降低 KV Cache 显存占用的场景。
- 与线性量化配合实现全量化方案，提升长序列推理效率。

不适用场景：

- 注意力模块 `forward` 未接受或未调用 `cache.update()`。
- 需要 INT8 以外的缓存量化格式。

## 2. 流程关系与前置条件

**上级流程**：模型适配与验证通过后，确定量化方案阶段。

**前置条件**：

- 已安装兼容版本的 msModelSlim 工具（详见《[msModelSlim 工具安装指南](../../../install_guide/install_guide.md)》）。
- 已确认目标模型的注意力模块接受 `DynamicCache` 对象并调用 `cache.update()`。
- 已准备好校准数据集，用于缓存量化校准。

**后续操作**：量化流程执行 → 精度验证 → 部署上线或进入调优。

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型权重目录 | 模型下载或本地路径 | HuggingFace 格式，含 `config.json` 及 `*.safetensors` | 可被目标 Transformers 版本加载 |
| 输入 | 校准数据集 | 工具默认或用户指定 | JSONL 格式，每条含文本 prompt | 可被工具加载并完成前向推理 |
| 输入 | 量化 YAML 配置文件 | 用户编写 | 符合 msModelSlim YAML 规范，含 `dynamic_cache` 处理器配置 | 可通过工具 `--config_path` 参数加载 |
| 交付件 | 量化权重目录 | `--save_path` 指定路径 | 含 `quant_model_description.json` 及 `*.safetensors`，已应用缓存量化 | 推理冒烟通过 |

## 4. 流程总览

```mermaid
flowchart LR
    A[编写 YAML 配置] --> B[执行量化命令]
    B --> C[检测注意力层]
    C --> D[校准并部署量化]
    D --> E[验证量化结果]
```

## 5. 操作步骤

### 步骤 1：编写 YAML 配置文件

**目标**：编写包含 `dynamic_cache` 处理器配置的 YAML 文件。

**操作**：

1. 在 `spec.process` 下配置 `dynamic_cache` 处理器，指定 `type: "dynamic_cache"`。
2. 配置 `qconfig`，其中 `scope` 仅支持 `"per_channel"`、`dtype` 仅支持 `"int8"`、`method` 仅支持 `"minmax"`。
3. 按需配置 `include`/`exclude` 控制参与量化的注意力层。

YAML 配置示例：

```yaml
spec:
  process:
    - type: "dynamic_cache" # 固定为 `dynamic_cache`，用于指定 Processor。
      qconfig: # 量化配置，支持 per_channel 量化
        scope: "per_channel" # 量化粒度：仅支持per_channel
        dtype: "int8" # 量化数据类型，目前支持 int8
        symmetric: True # 对称量化，默认 True
        method: "minmax" # 量化方法，目前支持 minmax
      include: ["*"]
      exclude: ["model.layers.0.self_attn"]
```

YAML 配置字段详解如下：

| 字段名 | 作用 | 说明 |
| --- | --- | --- |
| type | 处理器类型标识 | 固定为 `"dynamic_cache"`。 |
| qconfig | 缓存量化配置 | 包含 `scope`、`dtype`、`symmetric`、`method` 等字段。 |
| scope | 量化粒度 | 仅支持 `"per_channel"`，按隐藏层维度计算量化参数。 |
| dtype | 量化数据类型 | 仅支持 `"int8"`。 |
| symmetric | 对称量化开关 | 布尔值，默认 `true`。 |
| method | 量化方法 | 仅支持 `"minmax"`。 |
| include | 包含的注意力层 | 字符串列表，支持通配符匹配。 |
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

1. 工具加载 YAML 配置，解析 `dynamic_cache` 处理器。
2. 检测阶段自动识别注意力层并为每个注意力层创建量化器。
3. 校准阶段在 `cache.update()` 调用时拦截 Key/Value 状态并收集统计信息。
4. 部署阶段转换为伪量化 IR 并保存量化权重。

**输出**：量化权重目录 `${SAVE_PATH}`，包含量化描述文件与权重分片。

### 步骤 3：验证量化结果

**目标**：确认量化权重文件完整且可加载。

**操作**：

1. 检查输出目录是否包含 `quant_model_description.json` 文件。
2. 检查日志确认缓存量化正常执行。
3. 使用推理框架加载量化权重进行冒烟测试。

**输出**：量化权重验证通过。

## 6. 验收条件

- 量化权重目录包含 `quant_model_description.json` 及所有必需的 `*.safetensors` 分片文件。
- 日志无缓存未被量化告警。
- 量化后模型推理精度不低于未使用 KVCache Quant 的基线。

## 7. 异常处置

- **缓存未被量化**：确认注意力前向接受了一个 `cache` 参数并正确调用 `cache.update()`。
- **新缓存类型不支持**：确认自定义缓存实现了标准的 `update` 接口，并正确处理返回值。
- **自定义命名不识别**：基于模块类名称模式匹配（`*self_attn*`），自定义命名需在适配器中适配。

## 8. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| KVCache Quant | 针对 KV Cache 的 INT8 量化算法 | [KVCache Quant 词条](./term_kvcache_quant.md) |
| DynamicCache | Transformers 标准动态缓存机制 | [KVCache Quant 词条](./term_kvcache_quant.md) |
| per_channel 量化 | 按隐藏层维度计算量化参数 | [KVCache Quant 词条](./term_kvcache_quant.md) |

## 9. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| `dynamic_cache` 处理器 | 用于执行缓存量化的 Processor 配置 | [KVCache Quant 词条](./term_kvcache_quant.md) |
| 自定义缓存接口 | 自定义 Cache 需实现的 `update` 接口 | [KVCache Quant 词条](./term_kvcache_quant.md) |
