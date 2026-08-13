# compressed-tensors

> **词条类别**：量化数据格式
>
> **英文名称**：compressed-tensors
>
> **应用领域**：HuggingFace / vLLM 生态量化权重交换
>
> **msModelSlim 实现**：`msmodelslim/format/compressed_tensors_format/`

## 1. 概述

compressed-tensors 是与 HuggingFace / vLLM 生态兼容的[量化格式](../README.md)，字段约定遵循 [vllm-project/compressed-tensors](https://github.com/vllm-project/compressed-tensors) 规范。msModelSlim **导出时内置**该规范的结构定义，量化过程**无需**安装 `compressed-tensors` 包。核心特征是将方案写入 `config.json` → `quantization_config`，权重写入 HF 风格 safetensors。

配置、执行与部署步骤见《[compressed-tensors 使用指南](compressed_tensors_usage.md)》。各量化模式的原理与公式见《[量化模式](../../quantization_mode/term_quantization_mode.md)》及下文支持表中的词条链接。

## 2. 词条介绍

### 2.1 原理

#### 2.1.1 核心思想

compressed-tensors 本质上是一套**面向 HuggingFace / vLLM 生态的量化模型落盘约定**：它不执行校准或伪量化计算，而是在一键量化流水线末尾，将已量化完成的 QIR 模块转换为推理框架可直接加载的两类交付件——**注入 `config.json` 的 `quantization_config`** 与 **HF 风格权重文件** `model*.safetensors`。前者描述 `quant_method: "compressed-tensors"`、scheme（`QuantizationArgs` / `config_groups`）等元数据，供 vLLM 等按 HF 自动检测路径识别量化方案；后者按模块前缀写入 int8 / scale / zero-point 等张量。推理侧先读 `quantization_config` 确定 scheme，再按键名从 safetensors 取数。

因此，用户可将 compressed-tensors 理解为：**msModelSlim 量化结果的 HF 生态标准导出包**——算法负责量化计算，本格式负责与 [vllm-project/compressed-tensors](https://github.com/vllm-project/compressed-tensors) 规范对齐的落盘与描述。

#### 2.1.2 关键性质

- 与 HF `from_pretrained` / vLLM 自动检测路径兼容。
- 当前仅线性层量化；`targets` 固定为 `["Linear"]`。
- 不支持分布式导出；`kv_cache_scheme` 恒为 `null`。
- 仅部分 QIR preset 实现了导出 handler（见下文支持表）。

### 2.2 <span id="export-artifacts">导出产物（交付件）</span>
#### 目录与文件说明

执行一键量化（`compressed_tensors`）后，在指定的 `save_path` 目录下典型生成以下文件：

```text
save_directory/
├── config.json                          # 注入 quantization_config 字段
├── model.safetensors                    # 或 model-00001-of-xxxxx.safetensors（分片）
├── model.safetensors.index.json         # 分片时生成
└── （从源模型复制的 HF 辅助文件）
    └── *.json / *.py / *.txt / *.jinja
```

| 文件名                            | 说明                                                |
| ------------------------------ | ------------------------------------------------- |
| `config.json`                  | 原始 HF 配置，并注入 `quantization_config`（量化方案元数据）       |
| `model.safetensors`            | **量化权重**；较大时可分片为 `model-*-of-*.safetensors`       |
| `model.safetensors.index.json` | 分片索引（仅分片时生成）                                      |
| HF 辅助文件                        | 自源模型复制的 `*.json` / `*.py` / `*.txt` / `*.jinja` 等 |

#### config.json → quantization_config

##### 文件结构示例

`quantization_config` 典型结构如下：

```json
{
  "version": "0.13.0",
  "quant_method": "compressed-tensors",
  "sparsity_config": {},
  "transform_config": {},
  "config_groups": {
    "group_0": {
      "targets": ["Linear"],
      "weights": { "num_bits": 8, "type": "int", "strategy": "channel", "symmetric": true, "dynamic": false },
      "input_activations": { "num_bits": 8, "type": "int", "strategy": "tensor", "symmetric": false, "dynamic": false },
      "format": "int-quantized"
    }
  },
  "format": "int-quantized",
  "quantization_status": "compressed",
  "global_compression_ratio": null,
  "ignore": ["re:...(?![.\\w])"],
  "kv_cache_scheme": null
}
```

##### <span id="global-metadata">顶层字段说明</span>

| 字段                                     | 说明                                                            |
| -------------------------------------- | ------------------------------------------------------------- |
| `version`                              | Schema 版本，固定 `"0.13.0"`                                       |
| `quant_method`                         | 固定 `"compressed-tensors"`                                     |
| `config_groups`                        | 按唯一 scheme 分组，键名为 `group_0`、`group_1` 等                       |
| `format`                               | 根格式，如 `int-quantized`、`mixed-precision`                       |
| `quantization_status`                  | 导出时为 `"compressed"`                                           |
| `global_compression_ratio`             | 全局压缩率（0–1 浮点数），可选信息字段，**不参与推理加载**；msModelSlim 当前不计算，恒为 `null` |
| `ignore`                               | 未量化但同类型层名的 regex 列表                                           |
| `kv_cache_scheme`                      | KV Cache 量化方案，**当前不支持**，恒为 `null`                             |
| `sparsity_config` / `transform_config` | 空对象占位                                                         |

##### QuantizationScheme（config_groups 内）

每个 `config_groups` 条目描述一组层的量化方案。**msModelSlim 当前仅支持线性层（**`nn.Linear` **/ QIR FakeQuantLinear）量化**，因此 `targets` 固定为 `["Linear"]`。

| 字段                   | 说明                            |
| -------------------- | ----------------------------- |
| `targets`            | 目标模块类型；**当前固定为** `["Linear"]` |
| `weights`            | 权重量化参数（`QuantizationArgs`）    |
| `input_activations`  | 输入激活量化参数；仅权重量化时为 `null`       |
| `output_activations` | 输出激活量化参数                      |
| `format`             | 层压缩格式，如 `int-quantized`       |

##### QuantizationArgs 参数说明

| 参数                | 类型        | 默认值     | 说明                                                                                     |
| ----------------- | --------- | ------- | -------------------------------------------------------------------------------------- |
| `num_bits`        | int       | `8`     | 量化位宽                                                                                   |
| `type`            | enum      | `"int"` | 量化类型：`"int"` / `"float"`                                                               |
| `symmetric`       | bool      | `true`  | 是否对称量化                                                                                 |
| `strategy`        | enum      | 自动推断    | 量化粒度：`tensor` / `channel` / `group` / `block` / `token` / `tensor_group` / `attn_head` |
| `group_size`      | int       | `null`  | group 策略的组大小                                                                           |
| `block_structure` | list[int] | `null`  | block 策略专用，长度为 2 的整数列表，形如 `[rows, cols]`                                               |
| `dynamic`         | bool      | `false` | 是否动态量化：`false` 静态，`true` 动态；msModelSlim 导出时由 QIR preset 显式写入                           |
| `actorder`        | enum      | `null`  | 激活排序（`group` / `weight` 等）                                                             |
| `scale_dtype`     | string    | `null`  | scale 张量数据类型（torch dtype 字符串）                                                          |
| `zp_dtype`        | string    | 自动推断    | zero-point 张量数据类型；对称量化时导出为 `null`                                                      |
| `observer`        | string    | 自动推断    | 校准方法；静态量化默认 `memoryless_minmax`，动态量化为 `null`                                           |
| `observer_kwargs` | object    | `{}`    | 传给 observer 的额外参数                                                                      |

### 2.3 <span id="engine-support">推理引擎支持情况</span>

compressed-tensors **均可导出**下表中的 Preset；下表描述的是产物能否被目标推理引擎按 HF `quantization_config` 加载。具体模型 × Preset × 引擎组合以支持矩阵与官方最佳实践为准。

| 格式 Preset | vLLM（HF 生态） | 说明 |
| --- | --- | --- |
| W8A8 Static | √ | 激活已离线校准；`act.scope: per_tensor` |
| W8A8 Dynamic | √ | 激活动态；`act.scope: per_token` |

> **图例**：`√` 表示该引擎存在可加载路径或已有验证实践。选型时先确认框架支持 `quant_method: "compressed-tensors"`，再在下文「[量化模式支持情况](#mode-support)」核对交付件字段。Ascend 私有路径请改用《[AscendV1](../ascendv1/term_ascendv1.md)》。

### 2.4 <span id="mode-support">量化模式支持情况</span>

> **交付件说明**：「交付件：quantization_config」→ `config.json` 内 scheme；「交付件：safetensors」→ `model*.safetensors`。模式原理见《[量化模式](../../quantization_mode/term_quantization_mode.md)》词条。

| 格式 Preset    | compressed-tensors 是否支持导出 | 量化模式词条 | 交付件：quantization_config                   | 交付件：safetensors                       |
| ------------ | ------------------------- | ---------------- | ----------------------------------------- | ------------------------------------- |
| W8A8 Static  | 支持                        | [W8A8 静态量化](../../quantization_mode/linear_layer_quantization/term_w8a8_static.md) | [W8A8 Static scheme](#desc-w8a8-static)   | [W8A8 Static 权重张量](#st-w8a8-static)   |
| W8A8 Dynamic | 支持                        | [W8A8 动态量化](../../quantization_mode/linear_layer_quantization/term_w8a8_dynamic.md) | [W8A8 Dynamic scheme](#desc-w8a8-dynamic) | [W8A8 Dynamic 权重张量](#st-w8a8-dynamic) |

### 2.5 各量化模式交付件格式

约定：

- `{prefix}` 为模块前缀（例如 `model.layers.0.self_attn.q_proj`）。
- **quantization_config**：写入 `config.json`；顶层与 `QuantizationArgs` 见上文「[顶层字段说明](#global-metadata)」；下文给出各 Preset 在 `config_groups` 中的典型 scheme。
- **safetensors**：文件 `model*.safetensors`（可分片）；键为 `{prefix}.<param>`，存实际数值张量。

#### W8A8 Static

##### <span id="desc-w8a8-static">config.json → quantization_config</span>
典型 `config_groups` 条目（`weights.dynamic = false`，含静态激活）：

| 字段路径                                                                         | 典型取值                                           | 说明     |
| ---------------------------------------------------------------------------- | ---------------------------------------------- | ------ |
| `targets`                                                                    | `["Linear"]`                                   | 仅线性层   |
| `weights.num_bits` / `type` / `strategy` / `symmetric` / `dynamic`           | `8` / `"int"` / `"channel"` / `true` / `false` | 权重量化   |
| `input_activations.num_bits` / `type` / `strategy` / `symmetric` / `dynamic` | `8` / `"int"` / `"tensor"` / `false` / `false` | 静态激活量化 |
| `format`                                                                     | `"int-quantized"`                              | 层压缩格式  |

##### <span id="st-w8a8-static">model*.safetensors</span>

| 张量名                         | 数据类型    | 说明                         |
| --------------------------- | ------- | -------------------------- |
| `{prefix}.weight`           | int8    | 量化权重                       |
| `{prefix}.weight_scale`     | float32 | 权重量化 scale（unsqueeze 为 2D） |
| `{prefix}.input_scale`      | float32 | 激活量化 scale                 |
| `{prefix}.input_zero_point` | -       | 仅当 `input_offset != 0` 时写入 |
| `{prefix}.bias`             | float32 | 可选                         |

#### W8A8 Dynamic

##### <span id="desc-w8a8-dynamic">config.json → quantization_config</span>
典型 `config_groups` 条目（激活为动态，导出时 `input_activations.dynamic = true` 或按 QIR 约定写入）：

| 字段路径                                                               | 典型取值                                           | 说明                              |
| ------------------------------------------------------------------ | ---------------------------------------------- | ------------------------------- |
| `targets`                                                          | `["Linear"]`                                   | 仅线性层                            |
| `weights.num_bits` / `type` / `strategy` / `symmetric` / `dynamic` | `8` / `"int"` / `"channel"` / `true` / `false` | 权重量化（per-channel）               |
| `input_activations`                                                | 动态激活参数（`dynamic: true`）                        | 激活动态；scale / zero-point **不落盘** |
| `format`                                                           | `"int-quantized"`                              | 层压缩格式                           |

##### <span id="st-w8a8-dynamic">model*.safetensors</span>

| 张量名                     | 数据类型    | 说明                              |
| ----------------------- | ------- | ------------------------------- |
| `{prefix}.weight`       | int8    | 量化权重                            |
| `{prefix}.weight_scale` | float32 | 权重量化 scale（1D 时 unsqueeze 为 2D） |
| `{prefix}.bias`         | float32 | 可选                              |

> 动态激活的 scale / zero-point **不写入**权重文件，推理时 per-token 动态计算。

### 2.6 适用场景与限制

#### 2.6.1 适用场景

- 向 vLLM 等 HF 生态框架交付可互换量化权重。
- 需要与 compressed-tensors 规范对齐的 `quantization_config` 交换。

#### 2.6.2 使用限制

- 不支持分布式导出（`support_distributed() = False`）。
- KV Cache 量化暂不支持（`kv_cache_scheme = null`）。
- 仅 W8A8 Static / W8A8 Dynamic 两种 QIR 有 handler。
- 本词条交付件分两列说明：`quantization_config` 与 `model*.safetensors`；不展开量化模式原理与算子说明（见《[量化模式](../../quantization_mode/term_quantization_mode.md)》）。

## 3. 关联流程

| 流程                                                            | 说明                |
| ------------------------------------------------------------- | ----------------- |
| 《[compressed-tensors 使用指南](compressed_tensors_usage.md)》      | 确认模式支持、配置与执行 |
| 《[量化格式接入指南](../iformat_integration_guide.md)》                 | IFormat 1-shot 参考 |
| 《[一键量化使用指南](../../../user_guide/usage_quick_quantization.md)》 | save 配置总览         |

## 4. 关联词条

- [量化格式](../README.md)：上位概念，本词条所属目录。
- [AscendV1](../ascendv1/term_ascendv1.md)：其他，同属量化格式的并列落盘协议。
- [MindIE-SD](../mindie_sd/term_mindie_sd.md)：其他，同属量化格式的并列落盘协议。
- [量化模式](../../quantization_mode/term_quantization_mode.md)：配套术语，本格式 Preset 与交付件字段对应各量化模式；详见本页「[量化模式支持情况](#mode-support)」。
