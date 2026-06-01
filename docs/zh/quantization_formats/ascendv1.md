# AscendV1 格式说明

## 简介

AscendV1 是 msModelSlim 面向昇腾 NPU 推理的量化权重格式，由 `AscendV1Saver` 导出。推理框架（MindIE、vLLM Ascend等）通过 `quant_model_description.json` 识别各张量的量化类型，并从 `quant_model_weights.safetensors` 加载对应参数。

> 关于一键量化输出目录结构、QuaRot 与 debug 信息，请参见《[一键量化生成结果](../feature_guide/quick_quantization_v1/quantization_result.md)》。

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
| `ext` | object | `{}` | 扩展配置 |

## 导出产物

```
save_directory/
├── quant_model_description.json         # 量化权重描述文件
├── quant_model_weights.safetensors      # 量化权重（可分片）
├── quant_model_weights.safetensors.index.json  # 分片索引（可选）
├── config.json                          # 复制的 HF 配置（移除 quantization_config）
└── （其他 HF 辅助文件）
```

`quant_model_description.json` 中，每个张量键对应一个量化类型标识；同一 Linear 层的所有参数（weight、scale 等）共享相同的类型标识。

## quant_model_description.json

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

```
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
| `W8A16` | 权 8bit、激活 16bit |
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

- [aclnnMatmulCompressDequant](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/aolapi/context/ops-math/aclnnMatmulCompressDequant.md) — 稀疏/压缩权重解压缩后 MatMul 与反量化。

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

- [aclnnQuantMatmulV2](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/aclnnQuantMatmulV2.md) — W8A8 静态量化 MatMul；`deq_scale` 的 UINT64 格式要求见该算子 `deqScale` 入参说明。

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

- [aclnnDynamicQuantV2](https://www.hiascend.com/document/detail/zh/canncommercial/800/apiref/aolapi/context/aclnnDynamicQuantV2.md) — 激活 per-token 动态量化。
- [aclnnGroupedMatmulV4](https://www.hiascend.com/document/detail/zh/canncommercial/800/apiref/aolapi/context/aclnnGroupedMatmulV4.md) — 支持 per-token 激活 + per-channel 权重的动态量化 MatMul。

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

- [aclnnQuantMatmulV2](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/aclnnQuantMatmulV2.md) — 静态激活分支 MatMul。
- [aclnnDynamicQuantV2](https://www.hiascend.com/document/detail/zh/canncommercial/800/apiref/aolapi/context/aclnnDynamicQuantV2.md) — 动态激活分支量化。

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

- [aclnnGroupedMatmulV4](https://www.hiascend.com/document/detail/zh/canncommercial/800/apiref/aolapi/context/aclnnGroupedMatmulV4.md) — 权重量化 MatMul（激活保持浮点，通过 `antiquantScale` 等对 int8 权重反量化后计算）。

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

- [aclnnGroupedMatmulV4](https://www.hiascend.com/document/detail/zh/canncommercial/800/apiref/aolapi/context/aclnnGroupedMatmulV4.md) — 支持 INT4 权重 + per-token 动态激活的 MatMul。

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

- [aclnnGroupedMatmulV4](https://www.hiascend.com/document/detail/zh/canncommercial/800/apiref/aolapi/context/aclnnGroupedMatmulV4.md) — 支持 INT4 权重、`scale_bias` 等反量化参数的 MatMul。

---

### WFP8AFP8_DYNAMIC（FP8 动态量化）

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `weight` | float8_e4m3fn | FP8 权重 |
| `weight_scale` | float32 | 权重量化 scale |
| `weight_offset` | float32 | 权重量化 zero-point |
| `bias` | float32 | 原始浮点偏置（可选） |

#### NPU 算子实现

- [aclnnDynamicMxQuantV2](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta1/API/aolapi/context/ops-nn/aclnnDynamicMxQuantV2.md) — FP8 动态量化 MatMul 系列算子。

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

- [aclnnDynamicMxQuantV2](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta1/API/aolapi/context/ops-nn/aclnnDynamicMxQuantV2.md) — MXFP block-wise 量化 MatMul。

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

- [aclnnDequantRopeQuantKvcache](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/API/aolapi/context/aclnnDequantRopeQuantKvcache.md) — KV Cache 量化写入与 RoPE 融合算子。
---

### FAQuant（Flash Attention 量化）

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `scale` | float16/bfloat16 | 量化 scale |
| `offset` | float16/bfloat16 | 量化 zero-point |

#### 特性介绍
- [Attention 量化（MindIE LLM）](https://www.hiascend.com/document/detail/zh/mindie/20RC2/mindiellm/llmdev/mindie_llm0294.html) — `fa_quant_type` 与 scale/offset 字段说明。

#### NPU 算子实现

- [aclnnFusedInferAttentionScore](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/aclnnFusedInferAttentionScore.md) — 全量/增量 Flash Attention 融合算子，支持 `quantScale` / `quantOffset` 等 FA 量化参数。

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

- [aclnnFlatQuant](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta1/API/aolapi/context/ops-nn/aclnnFlatQuant.md) — FlatQuant 仿射变换与 LAC 裁剪。
- 内层 Linear 量化 MatMul 参见对应基础模式（如 W8A8 的 [aclnnQuantMatmulV2](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/aclnnQuantMatmulV2.md)）。

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

- `div.mul_scale` 在推理侧对激活做逐元素缩放（见上文公式），无独立融合算子；内层 Linear 按实际量化类型选用对应 MatMul 算子（如 W8A8 参见 [aclnnQuantMatmulV2](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/aclnnQuantMatmulV2.md)）。

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

## 相关文档

- 《[格式支持矩阵](README.md)》
- 《[一键量化生成结果](../feature_guide/quick_quantization_v1/quantization_result.md)》
- 《[权重在加速库/MindIE 中的使用](../case_studies/quantization_weight_use_cases_in_acceleration_and_mindie_torch.md)》
