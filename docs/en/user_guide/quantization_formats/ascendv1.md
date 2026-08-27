# AscendV1 Format Description

<!-- md-trans-meta sourceCommit=28d96890174a0f5178a7a9d402607bad1fc08375 translatedAt=2026-08-20T03:18:13.591Z pushedAt=2026-08-20T03:18:43.394Z -->

## Introduction

AscendV1 is the quantized weight format of msModelSlim for Ascend NPU inference, exported by `AscendV1Saver`. The inference framework (MindIE, vLLM Ascend, etc.) identifies the quantization type of each tensor through `quant_model_description.json` and loads the corresponding parameters from `quant_model_weights.safetensors`.

> For the output directory structure of one-click quantization, QuaRot, and debug information, see *[One-Click Quantization Results](../feature_guide/quick_quantization_v1/quantization_result.md)*.

## YAML Configuration

```yaml
spec:
  save:
    - type: "ascendv1_saver"
      part_file_size: 4
```

### Configuration Parameters

| Name | **Type** | Default Value | Description |
|------|------|--------|------|
| `type` | string | `"ascendv1_saver"` | Saver type identifier, a fixed value |
| `part_file_size` | int | `4` | Weight shard size (GB); `0` indicates no sharding |
| `ext` | object | `{}` | Optional extended configuration in key-value pair format. The key name is customizable, and the value is a JSON-compatible type. The current `AscendV1Saver` does not read this field, so it can be omitted in regular export.<br>Example: `ext: { custom_tag: "experiment-v1" }` |

## Export Artifacts

```text
save_directory/
├── quant_model_description.json         # Quantized weight description file
├── quant_model_weights.safetensors      # Quantized weights (may be sharded)
├── quant_model_weights.safetensors.index.json  # Shard index (optional)
├── config.json                          # Copied HF config (without quantization_config)
└── (other HF auxiliary files)
```

In `quant_model_description.json`, each tensor key corresponds to a quantization type identifier; all parameters of the same Linear layer (weight, scale, etc.) share the same type identifier.

## quant_model_description.json

### Global Metadata Fields

| Name | Type | Description |
|--------|------|------|
| `model_quant_type` | string | Overall quantization type of the model (the highest-priority one in mixed quantization) |
| `version` | string | Format version, currently `"1.0.0"` |
| `group_size` | int | Group size for grouped quantization |
| `kv_quant_type` / `kv_cache_type` | string | KV Cache quantization type |
| `fa_quant_type` | string | Flash Attention quantization type |
| `reduce_quant_type` | string | Communication quantization type |
| `metadata` | object | Extended metadata (such as QuaRot information) |
| `optional` | object | Optional exported artifacts (such as the QuaRot global rotation matrix path) |

### Quantization Type Priority

When a model contains multiple quantization types, `model_quant_type` is selected according to the following priority (the later an item appears in the list, the higher its priority):

```text
FLOAT → W16A16S → W8A16 → W8A8_DYNAMIC → W8A8_MIX → W8A8
→ WFP8AFP8_DYNAMIC → W8A8_MXFP8 → W4A8_MXFP
→ W4A4_DYNAMIC → W4A4_MXFP4 → W4A4_MXFP4_DUALSCALE
```

### Quantization Type Enumeration

| Enumeration Value | Description |
|--------|------|
| `FLOAT` | Floating-point (unquantized) |
| `W16A16S` | W16A16 sparse quantization |
| `W8A8` | W8A8 static quantization |
| `W8A8_DYNAMIC` | W8A8 dynamic quantization (per-token activation) |
| `W8A8_MIX` | W8A8 mixed quantization (PDMIX) |
| `W8A16` | 8-bit weight, 16-bit activation |
| `W4A4_DYNAMIC` | W4A4 dynamic quantization |
| `W4A8_DYNAMIC` | W4A8 dynamic quantization |
| `WFP8AFP8_DYNAMIC` | FP8 dynamic quantization |
| `W8A8_MXFP8` | MXFP8 quantization |
| `W4A8_MXFP` | W4A8 MXFP quantization |
| `W4A4_MXFP4` | W4A4 MXFP4 quantization |
| `W4A4_MXFP4_DUALSCALE` | W4A4 MXFP4 dual-scale quantization |
| `C8` | KV Cache 8-bit quantization |
| `FAQuant` | Flash Attention quantization |

The remaining key-value pairs are `{tensor name}: {quantization type}`, for example, `"model.layers.0.self_attn.q_proj.weight": "W8A8"`.

> Each quantization mode described below provides an **NPU operator implementation** section that links to the CANN operator documentation, making it easier to understand the inference-side usage by referring to the exported weight fields. Operator availability depends on the CANN version and chip model.

## Detailed Description of Each Quantization Mode

### FLOAT (unquantized)

| Name | Type | Description |
|--------|----------|------|
| `weight` | float16/bfloat16 | Original floating-point weight |
| `bias` | float16/bfloat16 | Bias (optional) |

#### NPU Operator Implementation

The weight retains floating-point precision, and the inference side uses a conventional floating-point MatMul without any dedicated quantization operator.

### W16A16S (Sparse Quantization)

| Name | Type | Description |
|--------|----------|------|
| `weight` | float16/bfloat16 | Weight after sparse processing |
| `scale` | float16/bfloat16 | Scale factor |

#### NPU Operator Implementation

- [aclnnMatmulCompressDequant](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/API/aolapi/context/ops-nn/aclnnMatmulCompressDequant.md) — MatMul and dequantization after decompressing sparse/compressed weights.

### W8A8 (Static Quantization)

W8A8 applies int8 static quantization to both weights and activations, and is one of the most commonly used formats for Ascend inference.

#### Quantization Parameters

| Name | Type | Description |
|--------|----------|------|
| `weight` | int8 | Quantized weight |
| `quant_bias` | int32 | Quantized bias |
| `input_scale` | float32 | Activation quantization scale |
| `input_offset` | float32 | Activation quantization zero-point |
| `deq_scale` | int64/float32 | Comprehensive dequantization scale |
| `bias` | float32 | Original floating-point bias (optional, marked as FLOAT) |

#### Quantization and Dequantization Formula

**Weight quantization** (per-channel symmetric):

$$quant\_weight = \text{round}\left(\frac{weight}{weight\_scale}\right)$$

**Activation quantization** (per-tensor asymmetric):

$$quant\_act = \text{round}\left(\frac{act}{input\_scale} + input\_offset\right)$$

**Derived parameters at export time** (`AscendV1Saver.on_w8a8_static`):

$$deq\_scale = input\_scale \times weight\_scale$$

$$correction = \left(\sum_{dim=1} quant\_weight\right) \times input\_offset$$

$$quant\_bias = \text{round}\left(\frac{bias}{deq\_scale} - correction\right)$$

**Inference dequantization** (conceptual formula, where $\cdot$ denotes matrix multiplication):

$$output = (quant\_act \cdot quant\_weight + quant\_bias) \times deq\_scale$$

#### deq_scale Storage Rules

- When the model-global dtype is **bfloat16**: `deq_scale` is stored as **float32**.

- Otherwise: the float32 bit pattern is reinterpreted and stored as **int64** to satisfy the **UINT64** input parameter requirement of the Ascend quantization matrix multiplication operator for `deqScale` (the inference side uses it directly according to the operator convention and does not cast it back to float32).

#### Feature Introduction

- [W8A8 Quantization Feature (MindIE LLM)](https://www.hiascend.com/document/detail/en/mindie/20RC2/mindiellm/llmdev/mindie_llm0288.html) — Describes the exported fields and inference integration.

#### NPU Operator Implementation

- [aclnnQuantMatmulV2](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/API/aolapi/context/ops-nn/aclnnQuantMatmulV2.md) — W8A8 static quantization MatMul; for the UINT64 format requirements of `deq_scale`, see the `deqScale` input parameter description of this operator.

### W8A8_DYNAMIC (Dynamic Quantization)

Weights use int8 per-channel static quantization, and activations use per-token dynamic quantization.

| Name | Type | Description |
|--------|----------|------|
| `weight` | int8 | Quantized weight |
| `weight_scale` | float32 | Weight quantization scale |
| `weight_offset` | float32 | Weight quantization zero-point (0 for symmetric quantization) |
| `bias` | float32 | Original floating-point bias (optional) |

**Dequantization Formula**:

```python
deq_weight = (weight - weight_offset) * weight_scale
```

Activation quantization parameters are computed dynamically during inference and are not written to the weight file.

#### NPU Operator Implementation

- [aclnnDynamicQuantV2](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/API/aolapi/context/ops-nn/aclnnDynamicQuantV2.md) — per-token dynamic quantization of activations.

- [aclnnGroupedMatmulV4](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/API/aolapi/context/ops-transformer/aclnnGroupedMatmulV4.md) — dynamic quantization MatMul supporting per-token activations and per-channel weights.

### W8A8_MIX (Mixed Quantization/PDMIX)

A mixed mode that combines W8A8 static activation quantization with W8A8 dynamic weight quantization. Its parameters are the union of those of W8A8 and W8A8_DYNAMIC:

| Name | Type | Description |
|--------|----------|------|
| `weight` | int8 | Quantized weight |
| `quant_bias` | int32 | Quantized bias |
| `input_scale` | float32 | Activation quantization scale |
| `input_offset` | float32 | Activation quantization zero-point |
| `deq_scale` | int64/float32 | Comprehensive dequantization scale |
| `weight_scale` | float32 | Weight quantization scale |
| `weight_offset` | float32 | Weight quantization zero-point |
| `bias` | float32 | Original floating-point bias (optional) |

The derivation formulas for `deq_scale` and `quant_bias` are the same as those for W8A8 static quantization.

#### NPU Operator Implementation

- [aclnnQuantMatmulV2](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/API/aolapi/context/ops-nn/aclnnQuantMatmulV2.md) — MatMul for the static activation branch.

- [aclnnDynamicQuantV2](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/API/aolapi/context/ops-nn/aclnnDynamicQuantV2.md) — Quantization for the dynamic activation branch.

### W8A16 (Weight Quantization)

Only weight quantization is performed, and activations retain floating-point precision.

| Name | Type | Description |
|--------|----------|------|
| `weight` | int8 | Quantized weight |
| `weight_scale` | float32 | Weight quantization scale |
| `weight_offset` | float32 | Weight quantization zero-point |
| `bias` | float32 | Original floating-point bias (optional) |

**Dequantization Formula**:

```python
deq_weight = (weight - weight_offset) * weight_scale
```

#### NPU Operator Implementation

- [aclnnGroupedMatmulV4](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/API/aolapi/context/ops-transformer/aclnnGroupedMatmulV4.md) — Weight quantization MatMul (the activation remains floating-point, and the int8 weight is dequantized through `antiquantScale` and other parameters before computation).

### W4A4_DYNAMIC (W4A4 dynamic quantization)

| Name | Type | Description |
|--------|----------|------|
| `weight` | int8 | int4 packed storage |
| `weight_scale` | float32 | Weight quantization scale |
| `weight_offset` | float32 | Weight quantization zero-point |
| `bias` | float32 | (Optional) Original floating-point bias |

Activation quantization parameters are computed dynamically during inference and are not saved.

#### NPU Operator Implementation

- [aclnnGroupedMatmulV4](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/API/aolapi/context/ops-transformer/aclnnGroupedMatmulV4.md) — MatMul with INT4 weights and per-token dynamic activation.

### W4A8_DYNAMIC (W4A8 dynamic quantization)

| Name | Type | Description |
|--------|----------|------|
| `weight` | int8 | int4 packed storage |
| `weight_scale` | float32 | Weight quantization scale |
| `weight_offset` | float32 | Weight quantization zero-point |
| `scale_bias` | float32 | Wdditional adjustment factor for dequantization |
| `bias` | float32 | (Optional) Original floating-point bias |

**Dequantization Formula**:

```python
deq_weight = (weight - weight_offset) * weight_scale + scale_bias
```

#### NPU Operator Implementation

- [aclnnGroupedMatmulV4](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/API/aolapi/context/ops-transformer/aclnnGroupedMatmulV4.md) — MatMul with INT4 weights and dequantization parameters such as `scale_bias`.

### WFP8AFP8_DYNAMIC (FP8 Dynamic Quantization)

| Name | Type | Description |
|--------|----------|------|
| `weight` | float8_e4m3fn | FP8 weight |
| `weight_scale` | float32 | Weight quantization scale |
| `weight_offset` | float32 | Weight quantization zero-point |
| `bias` | float32 | (Optional) Original floating-point bias |

#### NPU Operator Implementation

- [aclnnDynamicMxQuantV2](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/API/aolapi/context/ops-nn/aclnnDynamicMxQuantV2.md) — FP8 dynamic quantization MatMul series operator.

### MXFP Series (W8A8_MXFP8 / W4A8_MXFP / W4A4_MXFP4)

The MX (Microscaling) format uses FP8/FP4 weights with block-wise scale.

| Parameter Name | Data Type | Description |
|--------|----------|------|
| `weight` | float8_e4m3fn / uint8(packed fp4) | Quantized weight |
| `weight_scale` | uint8 | Scale (stored with a **+127 offset**, range 0 to 255) |
| `bias` | float32 | (Optional) Original floating-point bias |

**Scale offset**: During export, `weight_scale_stored = weight_scale + 127`, mapping the range -127 to 128 to the uint8 range.

#### NPU Operator Implementation

- [aclnnDynamicMxQuantV2](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/(optional)/API/aolapi/context/ops-nn/aclnnDynamicMxQuantV2.md) — MXFP block-wise quantized MatMul.

#### W4A4_MXFP4_DUALSCALE

In addition to W4A4_MXFP4, it further includes:

| Name | Type | Description |
|--------|----------|------|
| `weight_dual_scale` | float32 | Second-path scale |

Based on the NPU operator of W4A4_MXFP4, `weight_dual_scale` is used as the second-path scale input parameter.

### C8 (KV Cache Quantization)

| Name | Data Type | Description |
|--------|----------|------|
| `kv_cache_scale` | float32/float16 | KV Cache quantization scale |
| `kv_cache_offset` | float32/float16 | KV Cache quantization zero-point |

#### Feature Introduction

- [KV Cache int8 (MindIE LLM)](https://www.hiascend.com/document/detail/en/mindie/20RC2/mindiellm/llmdev/mindie_llm0292.html) — Description of the `kv_cache_scale` / `kv_cache_offset` fields.

#### NPU Operator Implementation

- [aclnnDequantRopeQuantKvcache](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-transformer/aclnnDequantRopeQuantKvcache.md) — KV Cache quantization write and RoPE fusion operator.

### FAQuant (Flash Attention Quantization)

| Name | Type | Description |
|--------|----------|------|
| `scale` | float16/bfloat16 | Quantization scale |
| `offset` | float16/bfloat16 | Quantization zero-point |

#### Feature Introduction

- [Attention Quantization (MindIE LLM)](https://www.hiascend.com/document/detail/en/mindie/20RC2/mindiellm/llmdev/mindie_llm0294.html) — Description of the `fa_quant_type` and scale/offset fields.

#### NPU Operator Implementation

- [aclnnFusedInferAttentionScore](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-transformer/aclnnFusedInferAttentionScore.md) — Full/incremental Flash Attention fused operator, supporting FA quantization parameters such as `quantScale` / `quantOffset`.

### FlatQuant (Dynamic/Static)

FlatQuant combines a linear transformation with quantization and additionally includes transformation matrices:

| Name | Type | Description |
|--------|----------|------|
| `weight` | int8/int32 | Quantized weight |
| `weight_scale` / `weight_offset` | float32 | Weight quantization parameter |
| `input_scale` / `input_offset` | float32 | Activation quantization parameter |
| `deq_scale` | float32 | Comprehensive dequantization scale |
| `quant_bias` | int32 | Quantized bias |
| `left_trans` / `right_trans` | float32 | Feature transformation matrix |
| `clip_ratio` | float32 | Clipping ratio |
| `bias` | float32 | (Optional) Original floating-point bias |

Taking the input activation of a Linear layer as an example, before quantization, a Kronecker affine transformation and learnable activation clipping (LAC) are applied to the activation $x$ in sequence:

$$x = x \cdot \mathrm{Kronecker}(left\_trans, right\_trans)$$

$$x = \mathrm{clamp}\big(x,\ x.\max() \cdot \mathrm{sigmoid}(clip\_ratio),\ x.\min() \cdot \mathrm{sigmoid}(clip\_ratio)\big)$$

Identifier: `W8A8_FLATQUANT_DYNAMIC` or `W4A8_FLATQUANT_DYNAMIC`.

#### NPU Operator Implementation

- [aclnnFlatQuant](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-nn/aclnnFlatQuant.md) — FlatQuant affine transformation and LAC clipping.

- For the inner layer Linear quantized MatMul, refer to the corresponding base mode (for example, [aclnnQuantMatmulV2](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/API/aolapi/context/ops-nn/aclnnQuantMatmulV2.md) for W8A8).

### NonFusionSmoothQuant (smooth quantization)

| Name | Type | Description |
|--------|----------|------|
| `div.mul_scale` | float32 | Smoothing scale factor |
| Inner Linear parameters | - | Determined by the inner quantization type |

Taking the input activation of a Linear layer as an example, smooth scaling is applied to the activation $x$ before quantization:

$$x = x \cdot div.mul\_scale$$

The inner weight is identified as `FLOAT` in the description.

#### NPU Operator Implementation

- `div.mul_scale` performs element-wise scaling on the activation on the inference side (see the formula above), with no independent fused operator; the inner layer Linear selects the corresponding MatMul operator based on the actual quantization type (for W8A8, see [aclnnQuantMatmulV2](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/API/aolapi/context/ops-nn/aclnnQuantMatmulV2.md)).

## Parameter Comparison Table

| Parameter | FLOAT | W16A16S | W8A8 | W8A8_DYN | W8A8_MIX | W8A16 | W4A4_DYN | W4A8_DYN | WFP8 | MXFP | C8 | FAQuant |
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

> W8A8_DYN = W8A8_DYNAMIC; W4A4_DYN = W4A4_DYNAMIC; W4A8_DYN = W4A8_DYNAMIC; WFP8 = WFP8AFP8_DYNAMIC; MXFP = W8A8_MXFP8 / W4A8_MXFP / W4A4_MXFP4 series.

## Related Documents

- *[Format Support Matrix](README.md)*

- *[One-Click Quantization Generation Results](../feature_guide/quick_quantization_v1/quantization_result.md)*

- *[Using Weights in Acceleration Libraries/MindIE](../../best_practices/quantization_weight_use_cases_in_acceleration_and_mindie_torch.md)*
