# 一键量化生成结果

## 简介

本文档详细介绍一键量化功能生成的输出文件结构，帮助用户理解量化结果的组织方式以及各个文件的作用。

## 生成文件概述

执行一键量化命令后，会在指定的 `save_path` 目录下生成以下文件：

```bash
├── config.json                          # 原始模型配置文件
├── generation_config.json               # 原始生成配置文件
├── quant_model_description.json         # 量化权重描述文件
├── quant_model_weights.safetensors      # 量化权重文件（若权重较大可能分片，通过 index.json 索引）
├── tokenizer_config.json                # 原始分词器配置文件
├── tokenizer.json                       # 原始分词器词汇表
├── {model_type}_best_practice.yaml      # 量化配置协议
├── vocab.json                           # 原始词汇映射文件（部分模型）
├── optional/                            # 可选导出目录（部分算法启用时生成）
│   └── quarot.safetensors               # QuaRot 全局旋转矩阵（启用 export_extra_info 时生成）
└── debug_info/                          # 调试信息目录（仅在启用 --debug 参数时生成）
    ├── debug_info.json                  # 调试信息元数据（JSON格式）
    └── debug_info.safetensors           # 调试信息张量数据（SafeTensors格式）
```

### 文件说明

| 文件名 | 说明 |
|--------|------|
| `config.json` | 原始模型的配置文件，包含模型架构、层数、隐藏维度等关键参数 |
| `generation_config.json` | 原始模型的生成配置文件，包含采样策略、最大生成长度等推理相关参数 |
| `quant_model_description.json` | **量化权重描述文件**，记录每个权重张量的量化类型和元数据 |
| `quant_model_weights.safetensors` | **量化权重文件**，包含实际存储的量化后的模型权重数据（若权重较大可能分片保存为多个文件，通过 `model.safetensors.index.json` 索引）。 |
| `tokenizer_config.json` | 原始分词器的配置文件，包含特殊 token、词表大小等信息 |
| `tokenizer.json` | 原始分词器的词汇表文件，定义 token 与 ID 的映射关系 |
| `{model_type}_best_practice.yaml` | **量化配置协议文件**，记录本次量化所使用的完整配置信息，参考[量化配置协议详解](usage.md#量化配置协议详解) |
| `vocab.json` | 原始词汇映射文件，部分模型（如 GPT 风格模型）会包含此文件 |
| `optional/quarot.safetensors` | **QuaRot 全局旋转矩阵文件**（仅在使用 QuaRot 且 `export_extra_info: True` 时生成），存储全局旋转矩阵 `Q`。详见[QuaRot 旋转量化](#quarot-旋转量化) |
| `debug_info/` | **调试信息目录**（仅在启用 `--debug` 参数时生成），包含量化过程中的上下文信息，用于问题排查和算法分析。详见[调试信息输出](#调试信息输出) |

## quant_model_description.json 详解

`quant_model_description.json` 是量化权重描述文件，它记录了模型中每个权重张量的量化类型和相关元数据信息，是推理框架加载量化模型的重要依据。

### 文件结构示例

```json
{
  "model_quant_type": "W8A8",
  "version": "1.0.0",
  "group_size": 128,
  "kv_quant_type": "KV8",
  "model.layers.0.self_attn.qkv_proj.weight": "W8A8",
  "model.layers.0.self_attn.o_proj.weight": "W8A8",
  "model.layers.0.mlp.gate_proj.weight": "W8A8",
  "model.layers.0.mlp.up_proj.weight": "W8A8",
  "model.layers.0.mlp.down_proj.weight": "W8A8",
  "metadata": {},
  "optional": {}
}
```

> [!Note] 说明
> `*.weight` 字段名称由模型本身决定。

### 字段类型说明

#### 全局元数据字段

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `model_quant_type` | `string` | 模型整体的量化类型，标识模型采用的量化方案 |
| `version` | `string` | 量化工具版本号，格式为 `x.x.x` |
| `group_size` | `int` | 量化分组大小，用于分组量化时的参数 |
| `kv_quant_type` | `string` | KV Cache 量化类型，如 `KV8` 表示 8bit KV Cache |
| `kv_cache_type` | `string` | KV Cache 量化类型的别名，与 `kv_quant_type` 含义相同 |
| `fa_quant_type` | `string` | Flash Attention 量化类型，如 `FAQuant` |
| `reduce_quant_type` | `string` | 通信量化类型，如 `per_channel` |
| `metadata` | `object` | 其他元数据信息，如 QuaRot 相关信息 |
| `optional` | `object` | 可选功能模块信息，如 QuaRot 全局旋转矩阵路径 |

#### 量化类型枚举值

| 枚举值 | 说明 |
|--------|------|
| `FLOAT` | 浮点数（未量化） |
| `W16A16S` | W16A16s稀疏量化，权重16bit稀疏 |
| `W8A8` | W8A8量化，权重8bit，激活8bit |
| `W8A8_DYNAMIC` | W8A8动态量化，权重8bit，激活per-token动态量化 |
| `W8A8_MIX` | W8A8混合量化，结合静态和动态量化 |
| `W8A16` | W8A16量化，权重8bit，激活16bit |
| `W4A4_DYNAMIC` | W4A4动态量化，权重4bit，激活per-token动态量化 |
| `WFP8AFP8_DYNAMIC` | WFP8/AFP8动态量化 |
| `W8A8_MXFP8` | W8A8 MXFP8量化 |
| `W4A8_MXFP` | W4A8 MXFP量化 |
| `W4A4_MXFP4` | W4A4 MXFP4量化 |
| `W4A8_DYNAMIC` | W4A8动态量化，权重4bit，激活per-token动态量化 |
| `C8` | KV Cache 8bit量化 |
| `FAQuant` | Flash Attention量化 |
| `FLATQUANT_DYNAMIC` | FlatQuant动态量化 |
| `FLATQUANT` | FlatQuant静态量化 |

#### 权重张量量化类型

除了上述全局元数据字段外，JSON 文件中的其他键值对表示模型中各个权重张量的量化类型。键为权重名称，值为该权重采用的量化类型。

例如：

```json
{
  "model.layers.0.self_attn.qkv_proj.weight": "W8A8"
}
```

表示 `model.layers.0.self_attn.qkv_proj.weight` 这个权重使用了 W8A8 量化方式。

## 权重类型详解

基于 `quant_model_description.json` 中的字段，本节详细介绍不同量化模式的参数结构。**不同量化模式的量化参数不同，safetensors权重文件和json描述文件也不同。**

> [!Tip] 完整格式规范
> 各量化模式的**公式推导、参数语义与对照总表**请参阅独立的 [AscendV1 格式说明](../../quantization_formats/ascendv1.md)。本文档侧重输出文件概览与参数结构速查。

### 各量化模式参数详解

以下内容基于 `AscendV1Saver` 实现，详细介绍各量化模式的参数结构。完整公式请参见《[AscendV1 格式说明](../../quantization_formats/ascendv1.md)》。

#### FLOAT（未量化）

FLOAT 表示未进行量化处理的权重，保持原始浮点精度。

**量化参数**（在 safetensors 中的存储）：

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `weight` | float16/bfloat16 | 原始浮点权重 |
| `bias` | float16/bfloat16 | 偏置（可选） |

**典型权重名称**：

```text
model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.q_proj.bias
```

**quant_model_description.json 标识**：`FLOAT`

---

#### W16A16S（稀疏量化）

W16A16S 是权重浮点稀疏量化模式，权重保持浮点精度并进行稀疏处理。

**量化参数**（在 safetensors 中的存储）：

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `weight` | float16/bfloat16 | 稀疏处理后的权重（非零值） |
| `scale` | float16/bfloat16 | 缩放因子 |

**典型权重名称**：

```text
model.layers.0.mlp.gate_proj.weight
model.layers.0.mlp.gate_proj.scale
```

**quant_model_description.json 标识**：`W16A16S`

---

#### W8A8（静态量化）

W8A8 是静态量化模式，对权重和激活值都进行 int8 量化。

**量化参数**（在 safetensors 中的存储）：

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `weight` | int8 | 量化后的权重数据 |
| `quant_bias` | int32 | 量化偏置 |
| `input_scale` | float16/bfloat16 | 激活值量化缩放因子 |
| `input_offset` | float16/bfloat16 | 激活值量化偏移因子 |
| `deq_scale` | int64/float32 | 综合反量化缩放因子 (input_scale × weight_scale) |
| `bias` | float32 | 原始浮点偏置（可选） |

**典型权重名称**：

```text
model.layers.0.self_attn.qkv_proj.weight
model.layers.0.self_attn.qkv_proj.quant_bias
model.layers.0.self_attn.qkv_proj.input_scale
model.layers.0.self_attn.qkv_proj.input_offset
model.layers.0.self_attn.qkv_proj.deq_scale
```

**quant_model_description.json 标识**：`W8A8`

---

#### W8A8_DYNAMIC（动态量化）

W8A8_DYNAMIC 是权重 int8 量化、激活值 per-token 动态量化的模式。

**量化参数**（在 safetensors 中的存储）：

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `weight` | int8 | 量化后的权重数据 |
| `weight_scale` | float32 | 权重量化缩放因子 |
| `weight_offset` | float32 | 权重量化偏移因子（对称量化时为全0） |
| `bias` | float32 | 原始浮点偏置（可选） |

**典型权重名称**：

```text
model.layers.0.self_attn.qkv_proj.weight
model.layers.0.self_attn.qkv_proj.weight_scale
model.layers.0.self_attn.qkv_proj.weight_offset
```

**反量化公式**：

```python
deq_weight = (weight - weight_offset) * weight_scale
```

**quant_model_description.json 标识**：`W8A8_DYNAMIC`

---

#### W8A8_MIX（混合量化）

W8A8_MIX 是结合静态和动态量化的混合模式。

**量化参数**（在 safetensors 中的存储）：

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `weight` | int8 | 量化后的权重数据 |
| `quant_bias` | int32 | 量化偏置 |
| `input_scale` | float16/bfloat16 | 激活值量化缩放因子 |
| `input_offset` | float16/bfloat16 | 激活值量化偏移因子 |
| `deq_scale` | int64/float32 | 综合反量化缩放因子 |
| `weight_scale` | float32 | 权重量化缩放因子 |
| `weight_offset` | float32 | 权重量化偏移因子 |
| `bias` | float32 | 原始浮点偏置（可选） |

**典型权重名称**：

```text
model.layers.0.self_attn.qkv_proj.weight
model.layers.0.self_attn.qkv_proj.quant_bias
model.layers.0.self_attn.qkv_proj.input_scale
model.layers.0.self_attn.qkv_proj.input_offset
model.layers.0.self_attn.qkv_proj.deq_scale
model.layers.0.self_attn.qkv_proj.weight_scale
model.layers.0.self_attn.qkv_proj.weight_offset
```

**quant_model_description.json 标识**：`W8A8_MIX`

---

#### W8A16（权重量化）

W8A16 是权重量化模式，只对权重进行 int8 量化，激活值保持浮点精度。

**量化参数**（在 safetensors 中的存储）：

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `weight` | int8 | 量化后的权重数据 |
| `weight_scale` | float32 | 权重量化缩放因子 |
| `weight_offset` | float32 | 权重量化偏移因子（对称量化时为全0） |
| `bias` | float32 | 原始浮点偏置（可选） |

**典型权重名称**：

```text
model.layers.0.self_attn.qkv_proj.weight
model.layers.0.self_attn.qkv_proj.weight_scale
model.layers.0.self_attn.qkv_proj.weight_offset
```

**反量化公式**：

```python
deq_weight = (weight - weight_offset) * weight_scale
```

**quant_model_description.json 标识**：`W8A16`

---

#### W4A4_DYNAMIC（W4A4 动态量化）

W4A4_DYNAMIC 是权重 int4 量化、激活值 per-token 动态量化的极低比特模式。

**量化参数**（在 safetensors 中的存储）：

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `weight` | int8 | 量化后的权重数据（int4 打包存储） |
| `weight_scale` | float32 | 权重量化缩放因子 |
| `weight_offset` | float32 | 权重量化偏移因子 |
| `bias` | float32 | 原始浮点偏置（可选） |

**典型权重名称**：

```text
model.layers.0.self_attn.qkv_proj.weight
model.layers.0.self_attn.qkv_proj.weight_scale
model.layers.0.self_attn.qkv_proj.weight_offset
```

**重要说明**：激活值量化参数在推理时动态计算，不保存到权重文件中。

**quant_model_description.json 标识**：`W4A4_DYNAMIC`

---

#### WFP8AFP8_DYNAMIC（FP8 动态量化）

WFP8AFP8_DYNAMIC 是 FP8 浮点动态量化模式。

**量化参数**（在 safetensors 中的存储）：

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `weight` | float8_e4m3fn | FP8 格式权重 |
| `weight_scale` | float32 | 权重量化缩放因子 |
| `weight_offset` | float32 | 权重量化偏移因子 |
| `bias` | float32 | 原始浮点偏置（可选） |

**典型权重名称**：

```text
model.layers.0.self_attn.qkv_proj.weight
model.layers.0.self_attn.qkv_proj.weight_scale
model.layers.0.self_attn.qkv_proj.weight_offset
```

**quant_model_description.json 标识**：`WFP8AFP8_DYNAMIC`

---

#### W8A8_MXFP8（MXFP8 量化）

W8A8_MXFP8 是使用 MX（Mixtral 格式）FP8 的量化模式。

**量化参数**（在 safetensors 中的存储）：

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `weight` | float8_e4m3fn | FP8 格式权重 |
| `weight_scale` | uint8 | 缩放因子（+127 偏移后存储） |
| `bias` | float32 | 原始浮点偏置（可选） |

**说明**：`weight_scale` 进行了 +127 偏移处理，使其从 -127~128 偏移到 0~255，正好覆盖 `uint8` 的取值范围。

**典型权重名称**：

```text
model.layers.0.self_attn.qkv_proj.weight
model.layers.0.self_attn.qkv_proj.weight_scale
```

**quant_model_description.json 标识**：`W8A8_MXFP8`

---

#### W4A8_MXFP（W4A8 MXFP 量化）

W4A8_MXFP 是权重 int4 + 激活 int8 的 MXFP 量化模式。

**量化参数**（在 safetensors 中的存储）：

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `weight` | float8_e4m3fn | FP8 格式权重 |
| `weight_scale` | uint8 | 缩放因子（+127 偏移后存储） |
| `bias` | float32 | 原始浮点偏置（可选） |

**典型权重名称**：

```text
model.layers.0.self_attn.qkv_proj.weight
model.layers.0.self_attn.qkv_proj.weight_scale
```

**quant_model_description.json 标识**：`W4A8_MXFP`

---

#### W4A4_MXFP4（W4A4 MXFP4 量化）

W4A4_MXFP4 是极低比特的 MXFP4 量化模式。

**量化参数**（在 safetensors 中的存储）：

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `weight` | float8_e4m3fn | FP8 格式权重 |
| `weight_scale` | uint8 | 缩放因子（+127 偏移后存储） |
| `bias` | float32 | 原始浮点偏置（可选） |

**典型权重名称**：

```text
model.layers.0.self_attn.qkv_proj.weight
model.layers.0.self_attn.qkv_proj.weight_scale
```

**quant_model_description.json 标识**：`W4A4_MXFP4`

---

#### C8（KV Cache 量化）

KV Cache 8bit 量化是针对 Key-Value 缓存的量化模式。

**量化参数**（在 safetensors 中的存储）：

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `kv_cache_scale` | float32/float16 | KV Cache 量化缩放因子 |
| `kv_cache_offset` | float32/float16 | KV Cache 量化偏移因子 |

**典型权重名称**：

```text
model.layers.0.self_attn.k_proj.kv_cache_scale
model.layers.0.self_attn.k_proj.kv_cache_offset
model.layers.0.self_attn.v_proj.kv_cache_scale
model.layers.0.self_attn.v_proj.kv_cache_offset
```

**quant_model_description.json 标识**：`C8`

---

#### W4A8_DYNAMIC（W4A8 动态量化）

W4A8_DYNAMIC 是权重 int4 量化、激活值 per-token 动态量化的模式。

**量化参数**（在 safetensors 中的存储）：

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `weight` | int8 | 量化后的权重数据（int4 打包存储） |
| `weight_scale` | float32 | 权重量化缩放因子 |
| `weight_offset` | float32 | 权重量化偏移因子 |
| `scale_bias` | float32 | 缩放偏置因子（用于反量化时的额外调整） |
| `bias` | float32 | 原始浮点偏置（可选） |

**典型权重名称**：

```text
model.layers.0.self_attn.qkv_proj.weight
model.layers.0.self_attn.qkv_proj.weight_scale
model.layers.0.self_attn.qkv_proj.weight_offset
model.layers.0.self_attn.qkv_proj.scale_bias
```

**反量化公式**：

```python
deq_weight = (weight - weight_offset) * weight_scale + scale_bias
```

**quant_model_description.json 标识**：`W4A8_DYNAMIC`

---

#### FlatQuant_DYNAMIC（FlatQuant 动态量化）

FlatQuant_DYNAMIC 是 FlatQuant 动态量化模式。

**量化参数**（在 safetensors 中的存储）：

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `weight` | int8/int32 | 量化后的权重数据 |
| `weight_scale` | float32 | 权重量化缩放因子 |
| `weight_offset` | float32 | 权重量化偏移因子 |
| `input_scale` | float32 | 激活值量化缩放因子（动态） |
| `input_offset` | float32 | 激活值量化偏移因子（动态） |
| `deq_scale` | float32 | 综合反量化缩放因子 |
| `quant_bias` | int32 | 量化偏置 |
| `left_trans` | float32 | 线性变换左矩阵 |
| `right_trans` | float32 | 线性变换右矩阵 |
| `clip_ratio` | float32 | 裁剪比例因子 |
| `bias` | float32 | 原始浮点偏置（可选） |

**说明**：

- FlatQuant 是一种结合线性变换的量化方法
- `left_trans` 和 `right_trans` 是用于特征变换的矩阵
- `clip_ratio` 用于控制量化范围

**quant_model_description.json 标识**：`W8A8_FLATQUANT_DYNAMIC` 或 `W4A8_FLATQUANT_DYNAMIC`

---

#### NonFusionSmoothQuant（平滑量化）

NonFusionSmoothQuant 是一种平滑量化模式，用于减少量化误差。

**量化参数**（在 safetensors 中的存储）：

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `div.mul_scale` | float32 | 平滑缩放因子 |
| 其他参数 | - | 由内部 Linear 层的量化类型决定 |

**典型权重名称**：

```text
model.layers.0.self_attn.q_proj.div.mul_scale
model.layers.0.self_attn.q_proj.linear.weight
```

**quant_model_description.json 标识**：`FLOAT`（内层权重）

---

#### FAQuant（Flash Attention 量化）

FAQuant 是 Flash Attention 量化模式。

**量化参数**（在 safetensors 中的存储）：

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `scale` | float16/bfloat16 | 量化缩放因子 |
| `offset` | float16/bfloat16 | 量化偏移因子 |

**典型权重名称**：

```text
model.layers.0.self_attn.q_proj.scale
model.layers.0.self_attn.q_proj.offset
```

**quant_model_description.json 标识**：`FAQuant`

> 注：
>
> 1. ✓ 表示该量化模式包含此参数，- 表示不包含，✓ (+127) 表示需要 +127 偏移处理
> 2. NonFusionSmoothQuant 的参数由内部 Linear 层的量化类型决定，额外包含 `div.mul_scale` 参数
> 3. QuaRot 的旋转矩阵参数根据具体实现可能包含部分或全部旋转矩阵

## QuaRot （旋转量化）

### 参数详解

QuaRot 是一种基于旋转的量化方法，用于保持量化后模型的功能等价性。

**量化参数**（在 safetensors 中的存储）：

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `heads_rotation` | float32 | 多头注意力旋转矩阵 |
| `kronecker_rotation_m` | float32 | Kronecker 旋转矩阵 M |
| `kronecker_rotation_n` | float32 | Kronecker 旋转矩阵 N |
| `global_rotation` | float32 | 全局旋转矩阵（保存在 optional 目录） |

**说明**：

- `heads_rotation` 用于多头注意力的旋转
- `kronecker_rotation_m` 和 `kronecker_rotation_n` 用于 MLP 层的旋转
- `global_rotation` 保存在 `optional/quarot.safetensors` 文件中

### 文件说明

#### optional/quarot.safetensors

当使用 QuaRot 算法且配置 `export_extra_info: True` 时，量化工具会在 `save_path` 目录下额外生成 `optional/` 子目录，以 SafeTensors 格式存储 QuaRot 使用的全局旋转矩阵 `Q`。其目录结构如下：

```bash
optional/
└── quarot.safetensors       # QuaRot 全局旋转矩阵文件
```

全局旋转矩阵 `Q`：

| 键名 | 数据类型 | 说明 |
|------|----------|------|
| `global_rotation` | float32 | QuaRot 全局旋转矩阵 `Q` |

#### quant_model_description.json 中的描述字段

**启用online**：`quant_model_description.json` 中新增 `metadata.quarot` 域：

```jsonc
{
  "metadata": {                                 // 其他元数据信息
    "quarot": {                                 // QuaRot 额外导出域
      "max_tp_size": 4,                         // 最大 TP 大小，由quarot量化配置中max_tp_size参数设置
      "heads_rotation": {                       // 多头注意力旋转矩阵
        "layers": [                             // 使用在线旋转的层（o层）
          "model.layers.0.self_attn.o_proj.",
          "model.layers.1.self_attn.o_proj.",
          "model.layers.2.self_attn.o_proj."
        ]
      },
      "kronecker_rotation": {                   // Kronecker 旋转矩阵
        "layers": [                             // 使用在线旋转的层（down层），由quarot量化配置down_proj_online_layers参数指定，并由safetensors文件中相应层的kronecker_rotation_m和kronecker_rotation_n描述
          "model.layers.2.mlp.down_proj."
        ]
      }
    }
  }
}
```

**启用export_extra_info**：`quant_model_description.json` 中新增 `optional.quarot` 域：

```jsonc
{
  "optional": {                                           // 可选导出件总入口
    "quarot": {                                           // QuaRot 额外导出域
      "rotation_map": {                                   // 旋转信息映射表
        "global_rotation": "optional/quarot.safetensors"  // 全局旋转矩阵文件（相对路径）
      }
    }
  }
}
```

### 使用场景

- **推理框架加载**：推理框架读取 `quant_model_description.json` 中的 `optional.quarot.rotation_map`，按路径加载全局旋转矩阵，用于在线旋转计算。
- **算法复现与调试**：可直接加载旋转矩阵，验证 QuaRot 变换的数学等价性。

## 调试信息输出

当在量化命令中添加 `--debug` 参数时，工具会在量化完成后自动保存量化过程中的上下文信息到 `debug_info` 目录。

### 调试信息目录结构

```bash
debug_info/
├── debug_info.json                  # 调试信息元数据（JSON格式）
└── debug_info.safetensors           # 调试信息张量数据（SafeTensors格式）
```

### 调试信息文件说明

#### debug_info.json

包含量化过程中的非张量数据和张量元数据，采用分命名空间（namespace）的结构组织：

**文件结构示例**：

```json
{
  "linear_quant_namespace": {
    "layer_name": "model.layers.0.self_attn.qkv_proj",
    "quant_config": {
      "weight_dtype": "int8",
      "act_dtype": "int8"
    },
    "statistics": {
      "weight_min": -0.5,
      "weight_max": 0.5
    },
    "scale_tensor": {
      "_type": "tensor",
      "_file": "debug_info.safetensors",
      "_key": "tensor_0"
    }
  },
  "iter_smooth_namespace": {
    "smoothing_factors": {
      "_type": "tensor",
      "_file": "debug_info.safetensors",
      "_key": "tensor_1"
    }
  }
}
```

**字段说明**：

- **命名空间（namespace）**：每个处理器或模块会创建独立的命名空间，用于隔离不同阶段的调试信息
- **普通字段**：直接存储标量值（整数、浮点数、字符串、布尔值等）
- **张量引用**：对于 PyTorch 张量，存储引用信息：
  - `_type`: 固定值 `"tensor"`，标识这是一个张量引用
  - `_file`: 张量数据所在的文件名（`debug_info.safetensors`）
  - `_key`: 张量在 SafeTensors 文件中的键名

#### debug_info.safetensors

以 SafeTensors 格式存储量化过程中的所有张量数据，包括：

- 量化参数（scale、zero_point 等）
- 统计信息（最小值、最大值、直方图等）
- 中间结果张量
- 离群值抑制算法的平滑因子
- 其他调试用张量

**特点**：

- **高效存储**：SafeTensors 格式支持快速加载和内存映射
- **跨平台兼容**：可在不同框架和平台间共享
- **安全性**：相比 pickle 格式更安全，避免代码注入风险

### 调试信息的使用

调试信息可用于以下场景：

1. **量化精度调优**：分析哪些层的量化误差较大，离群值抑制算法是否生效
2. **算法研究与开发**：对比不同量化算法的效果，开发新的量化策略
3. **问题排查与报告**：快速定位问题所在，向技术支持提供详细的诊断信息
4. **模型分析与优化**：了解模型各层的激活值分布特征，识别量化敏感层

**加载调试信息示例**：

```python
import json
from safetensors import safe_open

# 加载 JSON 元数据
with open("debug_info/debug_info.json", "r") as f:
    debug_meta = json.load(f)

# 加载 SafeTensors 张量数据
with safe_open("debug_info/debug_info.safetensors", framework="pt") as f:
    # 获取所有张量的键名
    tensor_keys = f.keys()

    # 加载特定张量
    for key in tensor_keys:
        tensor = f.get_tensor(key)
        print(f"{key}: shape={tensor.shape}, dtype={tensor.dtype}")
```

**注意事项**：

- 调试信息可能占用较大的磁盘空间（通常为模型大小的 10%-50%）
- 启用调试模式会略微增加量化时间（通常增加 5%-10%）
- 调试信息可能包含模型的敏感信息，请妥善保管

详细使用说明请参考[调试模式使用指南](debug_mode.md)。
