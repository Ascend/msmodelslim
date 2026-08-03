# AscendV1 格式说明

## 简介

AscendV1 是 msModelSlim 面向昇腾 NPU 推理的量化权重格式，由 `AscendV1Saver` 导出。推理框架（MindIE、vLLM Ascend等）通过 `quant_model_description.json` 识别各张量的量化类型，并从 `quant_model_weights.safetensors` 加载对应参数。

## YAML 配置

```yaml
spec:
  save:
    - type: "ascendv1_saver"
      part_file_size: 4
```

### 配置参数

| 参数 | **类型** | 默认值 | 说明 |
|------|------|--------|------|
| `type` | string | `"ascendv1_saver"` | 保存器类型标识，固定值 |
| `part_file_size` | int | `4` | 权重分片大小（GB）；`0` 表示不分片 |
| `ext` | object | `{}` | 可选扩展配置，键值对格式，键名可自定义，值为 JSON 兼容类型；当前 `AscendV1Saver` 不读取此字段，常规导出可省略。<br>示例：`ext: { custom_tag: "experiment-v1" }` |

## 导出产物

执行一键量化（`ascendv1_saver`）后，在指定的 `save_path` 目录下典型生成以下文件：

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
| `quant_model_weights.safetensors` | **量化权重文件**，包含实际存储的量化后的模型权重数据（若权重较大可能分片保存为多个文件，通过 index.json 索引） |
| `tokenizer_config.json` | 原始分词器的配置文件，包含特殊 token、词表大小等信息 |
| `tokenizer.json` | 原始分词器的词汇表文件，定义 token 与 ID 的映射关系 |
| `{model_type}_best_practice.yaml` | **量化配置协议文件**，记录本次量化所使用的完整配置信息，参考《[量化配置协议详解](../../../user_guide/usage_quick_quantization.md#5-量化配置协议详解)》 |
| `vocab.json` | 原始词汇映射文件，部分模型（如 GPT 风格模型）会包含此文件 |
| `optional/quarot.safetensors` | **QuaRot 全局旋转矩阵文件**（仅在使用 QuaRot 且 `export_extra_info: True` 时生成），存储全局旋转矩阵 `Q`。详见[QuaRot 旋转量化](#quarot-旋转量化) |
| `debug_info/` | **调试信息目录**（仅在启用 `--debug` 参数时生成），包含量化过程中的上下文信息，用于问题排查和算法分析。详见[调试信息输出](#调试信息输出) |

`quant_model_description.json` 中，每个张量键对应一个量化类型标识；同一 Linear 层的所有参数（weight、scale 等）共享相同的类型标识。

## quant_model_description.json

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

### 全局元数据字段

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `model_quant_type` | string | 模型整体量化类型（混合量化时取优先级最高者） |
| `version` | string | 格式版本，当前 `"1.0.0"` |
| `group_size` | int | 分组量化时的 group 大小 |
| `kv_quant_type` / `kv_cache_type` | string | KV Cache 量化类型 |
| `fa_quant_type` | string | Flash Attention 量化类型 |
| `reduce_quant_type` | string | 通信量化类型 |
| `metadata` | object | 扩展元数据（如 QuaRot 信息） |
| `optional` | object | 可选导出件（如 QuaRot 全局旋转矩阵路径） |

### 量化类型优先级

当模型中存在多种量化类型时，`model_quant_type` 按以下优先级（列表越靠后优先级越高）选取：

```text
FLOAT → W16A16S → W8A16 → W8A8_DYNAMIC → W8A8_MIX → W8A8
→ WFP8AFP8_DYNAMIC → W8A8_MXFP8 → W4A8_MXFP
→ W4A4_DYNAMIC → W4A4_MXFP4 → W4A4_MXFP4_DUALSCALE
```

### 量化类型枚举

| 枚举值 | 说明 |
|--------|------|
| `FLOAT` | 浮点数（未量化） |
| `W16A16S` | W16A16 稀疏量化 |
| `W8A8` | W8A8 静态量化 |
| `W8A8_DYNAMIC` | W8A8 动态量化（激活 per-token） |
| `W8A8_MIX` | W8A8 混合量化（PDMIX） |
| `W8A16` | 权重 8bit、激活 16bit |
| `W4A4_DYNAMIC` | W4A4 动态量化 |
| `W4A8_DYNAMIC` | W4A8 动态量化 |
| `WFP8AFP8_DYNAMIC` | FP8 动态量化 |
| `W8A8_MXFP8` | MXFP8 量化 |
| `W4A8_MXFP` | W4A8 MXFP 量化 |
| `W4A4_MXFP4` | W4A4 MXFP4 量化 |
| `W4A4_MXFP4_DUALSCALE` | W4A4 MXFP4 双 scale 量化 |
| `C8` | KV Cache 8bit 量化 |
| `FAQuant` | Flash Attention 量化 |

其余键值对为 `{张量名}: {量化类型}`，例如 `"model.layers.0.self_attn.q_proj.weight": "W8A8"`。

> 下文各量化模式均提供 **NPU 算子实现** 小节，链接 CANN 算子文档，便于对照导出权重字段理解推理侧用法。算子可用性以 CANN 版本与芯片型号为准。

---

## 各量化模式详解

### FLOAT（未量化）

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `weight` | float16/bfloat16 | 原始浮点权重 |
| `bias` | float16/bfloat16 | 偏置（可选） |

#### NPU 算子实现

权重保持浮点精度，推理侧使用常规浮点 MatMul，无专用量化算子。

---

### W16A16S（稀疏量化）

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `weight` | float16/bfloat16 | 稀疏处理后的权重 |
| `scale` | float16/bfloat16 | 缩放因子 |

#### NPU 算子实现

- [aclnnMatmulCompressDequant](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/aolapi/context/ops-nn/aclnnMatmulCompressDequant.md) — 稀疏/压缩权重解压缩后 MatMul 与反量化。

---

### W8A8（静态量化）

W8A8 对权重和激活均进行 int8 静态量化，是昇腾推理最常用的格式之一。

#### 量化参数

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `weight` | int8 | 量化后的权重 |
| `quant_bias` | int32 | 量化偏置 |
| `input_scale` | float32 | 激活量化 scale |
| `input_offset` | float32 | 激活量化 zero-point |
| `deq_scale` | int64/float32 | 综合反量化 scale |
| `bias` | float32 | 原始浮点偏置（可选，标识为 FLOAT） |

#### 量化与反量化公式

**权重量化**（per-channel 对称）：

$$quant\_weight = \text{round}\left(\frac{weight}{weight\_scale}\right)$$

**激活量化**（per-tensor 非对称）：

$$quant\_act = \text{round}\left(\frac{act}{input\_scale} + input\_offset\right)$$

**导出时派生参数**（`AscendV1Saver.on_w8a8_static`）：

$$deq\_scale = input\_scale \times weight\_scale$$

$$correction = \left(\sum_{dim=1} quant\_weight\right) \times input\_offset$$

$$quant\_bias = \text{round}\left(\frac{bias}{deq\_scale} - correction\right)$$

**推理反量化**（概念公式，$\cdot$ 表示矩阵乘法）：

$$output = (quant\_act \cdot quant\_weight + quant\_bias) \times deq\_scale$$

#### deq_scale 存储规则

- 模型全局 dtype 为 **bfloat16** 时：`deq_scale` 以 **float32** 存储。
- 否则：将 float32 的位模式 reinterpret 为 **int64** 存储，以满足昇腾量化矩阵乘算子对 `deqScale` 的 **UINT64** 入参要求（推理侧按算子约定直接使用，不会 cast 回 float32）。

#### 特性介绍

- [W8A8 量化特性（MindIE LLM）](https://www.hiascend.com/document/detail/zh/mindie/20RC2/mindiellm/llmdev/mindie_llm0288.html) — 导出字段与推理集成说明。

#### NPU 算子实现

- [aclnnQuantMatmulV2](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/aolapi/context/ops-nn/aclnnQuantMatmulV2.md) — W8A8 静态量化 MatMul；`deq_scale` 的 UINT64 格式要求见该算子 `deqScale` 入参说明。

---

### W8A8_DYNAMIC（动态量化）

权重 int8 per-channel 静态量化，激活 per-token 动态量化。

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `weight` | int8 | 量化权重 |
| `weight_scale` | float32 | 权重量化 scale |
| `weight_offset` | float32 | 权重量化 zero-point（对称量化时为 0） |
| `bias` | float32 | 原始浮点偏置（可选） |

**反量化公式**：

```python
deq_weight = (weight - weight_offset) * weight_scale
```

激活量化参数在推理时动态计算，不写入权重文件。

#### NPU 算子实现

- [aclnnDynamicQuantV2](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/aolapi/context/ops-nn/aclnnDynamicQuantV2.md) — 激活 per-token 动态量化。
- [aclnnGroupedMatmulV4](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/aolapi/context/ops-transformer/aclnnGroupedMatmulV4.md) — 支持 per-token 激活 + per-channel 权重的动态量化 MatMul。

---

### W8A8_MIX（混合量化 / PDMIX）

结合 W8A8 静态激活量化与 W8A8 动态权重量化的混合模式，参数为 W8A8 与 W8A8_DYNAMIC 的并集：

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `weight` | int8 | 量化权重 |
| `quant_bias` | int32 | 量化偏置 |
| `input_scale` | float32 | 激活量化 scale |
| `input_offset` | float32 | 激活量化 zero-point |
| `deq_scale` | int64/float32 | 综合反量化 scale |
| `weight_scale` | float32 | 权重量化 scale |
| `weight_offset` | float32 | 权重量化 zero-point |
| `bias` | float32 | 原始浮点偏置（可选） |

`deq_scale` 与 `quant_bias` 的推导公式与 W8A8 静态相同。

#### NPU 算子实现

- [aclnnQuantMatmulV2](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/aolapi/context/ops-nn/aclnnQuantMatmulV2.md) — 静态激活分支 MatMul。
- [aclnnDynamicQuantV2](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/aolapi/context/ops-nn/aclnnDynamicQuantV2.md) — 动态激活分支量化。

---

### W8A16（权重量化）

仅权重量化，激活保持浮点精度。

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `weight` | int8 | 量化权重 |
| `weight_scale` | float32 | 权重量化 scale |
| `weight_offset` | float32 | 权重量化 zero-point |
| `bias` | float32 | 原始浮点偏置（可选） |

**反量化公式**：

```python
deq_weight = (weight - weight_offset) * weight_scale
```

#### NPU 算子实现

- [aclnnGroupedMatmulV4](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/aolapi/context/ops-transformer/aclnnGroupedMatmulV4.md) — 权重量化 MatMul（激活保持浮点，通过 `antiquantScale` 等对 int8 权重反量化后计算）。

---

### W4A4_DYNAMIC（W4A4 动态量化）

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `weight` | int8 | int4 打包存储 |
| `weight_scale` | float32 | 权重量化 scale |
| `weight_offset` | float32 | 权重量化 zero-point |
| `bias` | float32 | 原始浮点偏置（可选） |

激活量化参数推理时动态计算，不保存。

#### NPU 算子实现

- [aclnnGroupedMatmulV4](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/aolapi/context/ops-transformer/aclnnGroupedMatmulV4.md) — 支持 INT4 权重 + per-token 动态激活的 MatMul。

---

### W4A8_DYNAMIC（W4A8 动态量化）

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `weight` | int8 | int4 打包存储 |
| `weight_scale` | float32 | 权重量化 scale |
| `weight_offset` | float32 | 权重量化 zero-point |
| `scale_bias` | float32 | 反量化额外调整因子 |
| `bias` | float32 | 原始浮点偏置（可选） |

**反量化公式**：

```python
deq_weight = (weight - weight_offset) * weight_scale + scale_bias
```

#### NPU 算子实现

- [aclnnGroupedMatmulV4](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/aolapi/context/ops-transformer/aclnnGroupedMatmulV4.md) — 支持 INT4 权重、`scale_bias` 等反量化参数的 MatMul。

---

### WFP8AFP8_DYNAMIC（FP8 动态量化）

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `weight` | float8_e4m3fn | FP8 权重 |
| `weight_scale` | float32 | 权重量化 scale |
| `weight_offset` | float32 | 权重量化 zero-point |
| `bias` | float32 | 原始浮点偏置（可选） |

#### NPU 算子实现

- [aclnnDynamicMxQuantV2](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/aolapi/context/ops-nn/aclnnDynamicMxQuantV2.md) — FP8 动态量化 MatMul 系列算子。

---

### MXFP 系列（W8A8_MXFP8 / W4A8_MXFP / W4A4_MXFP4）

MX（Microscaling）格式使用 FP8/FP4 权重与 block-wise scale。

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `weight` | float8_e4m3fn / uint8(packed fp4) | 量化权重 |
| `weight_scale` | uint8 | scale（**+127 偏移**后存储，范围 0~255） |
| `bias` | float32 | 原始浮点偏置（可选） |

**scale 偏移**：导出时 `weight_scale_stored = weight_scale + 127`，将 -127~128 映射到 uint8 范围。

#### NPU 算子实现

- [aclnnDynamicMxQuantV2](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/aolapi/context/ops-nn/aclnnDynamicMxQuantV2.md) — MXFP block-wise 量化 MatMul。

#### W4A4_MXFP4_DUALSCALE

在 W4A4_MXFP4 基础上额外包含：

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `weight_dual_scale` | float32 | 第二路 scale |

在 W4A4_MXFP4 的 NPU 算子基础上使用 `weight_dual_scale` 作为第二路 scale 入参。

---

### C8（KV Cache 量化）

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `kv_cache_scale` | float32/float16 | KV Cache 量化 scale |
| `kv_cache_offset` | float32/float16 | KV Cache 量化 zero-point |

#### 特性介绍

- [KV Cache int8（MindIE LLM）](https://www.hiascend.com/document/detail/zh/mindie/20RC2/mindiellm/llmdev/mindie_llm0292.html) — `kv_cache_scale` / `kv_cache_offset` 字段说明。

#### NPU 算子实现

- [aclnnDequantRopeQuantKvcache](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/aolapi/context/ops-transformer/aclnnDequantRopeQuantKvcache.md) — KV Cache 量化写入与 RoPE 融合算子。

---

### FAQuant（Flash Attention 量化）

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `scale` | float16/bfloat16 | 量化 scale |
| `offset` | float16/bfloat16 | 量化 zero-point |

#### 特性介绍

- [Attention 量化（MindIE LLM）](https://www.hiascend.com/document/detail/zh/mindie/20RC2/mindiellm/llmdev/mindie_llm0294.html) — `fa_quant_type` 与 scale/offset 字段说明。

#### NPU 算子实现

- [aclnnFusedInferAttentionScore](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/aolapi/context/ops-transformer/aclnnFusedInferAttentionScore.md) — 全量/增量 Flash Attention 融合算子，支持 `quantScale` / `quantOffset` 等 FA 量化参数。

---

### FlatQuant（动态/静态）

FlatQuant 结合线性变换的量化方法，额外包含变换矩阵：

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `weight` | int8/int32 | 量化权重 |
| `weight_scale` / `weight_offset` | float32 | 权重量化参数 |
| `input_scale` / `input_offset` | float32 | 激活量化参数 |
| `deq_scale` | float32 | 综合反量化 scale |
| `quant_bias` | int32 | 量化偏置 |
| `left_trans` / `right_trans` | float32 | 特征变换矩阵 |
| `clip_ratio` | float32 | 裁剪比例 |
| `bias` | float32 | 原始浮点偏置（可选） |

以 Linear 层输入激活为例，量化前对激活 $x$ 依次施加 Kronecker 仿射变换与可学习激活裁剪（LAC）：

$$x = x \cdot \mathrm{Kronecker}(left\_trans, right\_trans)$$

$$x = \mathrm{clamp}\big(x,\ x.\max() \cdot \mathrm{sigmoid}(clip\_ratio),\ x.\min() \cdot \mathrm{sigmoid}(clip\_ratio)\big)$$

标识：`W8A8_FLATQUANT_DYNAMIC` 或 `W4A8_FLATQUANT_DYNAMIC`。

#### NPU 算子实现

- [aclnnFlatQuant](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/aolapi/context/ops-nn/aclnnFlatQuant.md) — FlatQuant 仿射变换与 LAC 裁剪。
- 内层 Linear 量化 MatMul 参见对应基础模式（如 W8A8 的 [aclnnQuantMatmulV2](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/aolapi/context/ops-nn/aclnnQuantMatmulV2.md)）。

---

### NonFusionSmoothQuant（平滑量化）

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `div.mul_scale` | float32 | 平滑缩放因子 |
| 内层 Linear 参数 | - | 由内层量化类型决定 |

以 Linear 层输入激活为例，量化前对激活 $x$ 施加平滑缩放：

$$x = x \cdot div.mul\_scale$$

内层权重在 description 中标识为 `FLOAT`。

#### NPU 算子实现

- `div.mul_scale` 在推理侧对激活做逐元素缩放（见上文公式），无独立融合算子；内层 Linear 按实际量化类型选用对应 MatMul 算子（如 W8A8 参见 [aclnnQuantMatmulV2](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/aolapi/context/ops-nn/aclnnQuantMatmulV2.md)）。

---

## 参数对照总表

| 参数 | FLOAT | W16A16S | W8A8 | W8A8_DYN | W8A8_MIX | W8A16 | W4A4_DYN | W4A8_DYN | WFP8 | MXFP | C8 | FAQuant |
|------|:-----:|:-------:|:----:|:--------:|:--------:|:-----:|:--------:|:--------:|:----:|:----:|:--:|:-------:|
| weight | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - | - |
| bias | ✓ | - | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - | - |
| quant_bias | - | - | ✓ | - | ✓ | - | - | - | - | - | - | - |
| input_scale | - | - | ✓ | - | ✓ | - | - | - | - | - | - | - |
| input_offset | - | - | ✓ | - | ✓ | - | - | - | - | - | - | - |
| deq_scale | - | - | ✓ | - | ✓ | - | - | - | - | - | - | - |
| weight_scale | - | - | - | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓(+127) | - | - |
| weight_offset | - | - | - | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - | - | - |
| scale_bias | - | - | - | - | - | - | - | ✓ | - | - | - | - |
| scale (sparse) | - | ✓ | - | - | - | - | - | - | - | - | - | - |
| kv_cache_scale/offset | - | - | - | - | - | - | - | - | - | - | ✓ | - |
| scale/offset (FA) | - | - | - | - | - | - | - | - | - | - | - | ✓ |

> W8A8_DYN = W8A8_DYNAMIC；W4A4_DYN = W4A4_DYNAMIC；W4A8_DYN = W4A8_DYNAMIC；WFP8 = WFP8AFP8_DYNAMIC；MXFP = W8A8_MXFP8 / W4A8_MXFP / W4A4_MXFP4 系列。

---

## <span id="quarot-旋转量化">QuaRot（旋转量化）</span>

### 1 参数详解

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

### 2 文件说明

#### 2.1 optional/quarot.safetensors

当使用 QuaRot 算法且配置 `export_extra_info: True` 时，量化工具会在 `save_path` 目录下额外生成 `optional/` 子目录，以 SafeTensors 格式存储 QuaRot 使用的全局旋转矩阵 `Q`。其目录结构如下：

```bash
optional/
└── quarot.safetensors       # QuaRot 全局旋转矩阵文件
```

全局旋转矩阵 `Q`：

| 键名 | 数据类型 | 说明 |
|------|----------|------|
| `global_rotation` | float32 | QuaRot 全局旋转矩阵 `Q` |

#### 2.2 quant_model_description.json 中的描述字段

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

### 3 使用场景

- **推理框架加载**：推理框架读取 `quant_model_description.json` 中的 `optional.quarot.rotation_map`，按路径加载全局旋转矩阵，用于在线旋转计算。
- **算法复现与调试**：可直接加载旋转矩阵，验证 QuaRot 变换的数学等价性。

## 调试信息输出

当在量化命令中添加 `--debug` 参数时，工具会在量化完成后自动保存量化过程中的上下文信息到 `debug_info` 目录。

### 1 调试信息目录结构

```bash
debug_info/
├── debug_info.json                  # 调试信息元数据（JSON格式）
└── debug_info.safetensors           # 调试信息张量数据（SafeTensors格式）
```

### 2 调试信息文件说明

#### 2.1 debug_info.json

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

- **命名空间（namespace）**：每个处理器或模块会创建独立的命名空间，用于隔离不同阶段的调试信息。
- **普通字段**：直接存储标量值（整数、浮点数、字符串、布尔值等）。
- **张量引用**：对于 PyTorch 张量，存储引用信息：
  - `_type`: 固定值 `"tensor"`，标识这是一个张量引用
  - `_file`: 张量数据所在的文件名（`debug_info.safetensors`）
  - `_key`: 张量在 SafeTensors 文件中的键名

#### 2.2 debug_info.safetensors

以 SafeTensors 格式存储量化过程中的所有张量数据，包括：

- 量化参数（scale、zero_point 等）
- 统计信息（最小值、最大值、直方图等）
- 中间结果张量
- 离群值抑制算法的平滑因子
- 其他调试用张量

**特点**：

- **高效存储**：SafeTensors 格式支持快速加载和内存映射。
- **跨平台兼容**：可在不同框架和平台间共享。
- **安全性**：相比 pickle 格式更安全，避免代码注入风险。

### 3 调试信息的使用

调试信息可用于以下场景：

1. **量化精度调优**：分析哪些层的量化误差较大，离群值抑制算法是否生效。
2. **算法研究与开发**：对比不同量化算法的效果，开发新的量化策略。
3. **问题排查与报告**：快速定位问题所在，向技术支持提供详细的诊断信息。
4. **模型分析与优化**：了解模型各层的激活值分布特征，识别量化敏感层。

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

详细使用说明请参考《[调试模式使用指南](../../../user_guide/usage_debug_mode.md)》。

## 相关文档

- 《[格式支持矩阵](../README.md)》
- 《[一键量化使用指南](../../../user_guide/usage_quick_quantization.md)》
- 《[调试模式使用指南](../../../user_guide/usage_debug_mode.md)》
