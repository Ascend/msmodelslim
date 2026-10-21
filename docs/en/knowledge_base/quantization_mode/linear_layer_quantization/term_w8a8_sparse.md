# W8A8 Sparse Quantization

> **Entry Category**: Quantization Data Format ([Linear Layer Quantization](README.md))<br>
> **English Name**: W8A8 Sparse<br>
> **Application Domain**: Large language model quantization and compression, inference acceleration<br>
> **Hosting IR Class**: `W8A8StaticFakeQuantLinear` ([`msmodelslim/ir/w8a8_static.py`](../../../../../msmodelslim/ir/w8a8_static.py))<br>
> **Export Type**: `W8A8S`; the secondary-compression deployment form is `W8A8SC`<br>

## 1. Overview

W8A8 sparse quantization (export type `W8A8S`) still applies [INT8](../../quantization_basic/term_int8.md) quantization to the weights and activations of linear layers, but the weight undergoes **numerical sparsity** rather than zeroing part of the elements. 8bit can usually express 256 integer levels; W8A8S does not use all 256 levels but only a subset of them, so the effective value range of the weight is smaller than full-range 8bit. It is then compressed (export type `W8A8SC`) to encode this narrower codebook, resulting in a higher compression ratio than ordinary [W8A8 Static Quantization](term_w8a8_static.md). The `S` here is not the suffix of the `WxAy` bit-width naming.

---

## 2. Term Description

### 2.1 Mode Specification

| Dimension | Value | Description |
|------|------|------|
| Quantization Target | Weight W, Activation A | The tensors on both sides of the linear layer matrix multiplication; the weight additionally undergoes numerical sparsity |
| Bit Width | W 8bit, A 8bit | The same as W8A8 static quantization; still stored and computed in 8bit |
| Data Type | INT8 | Integer format; numerical sparsity means that the integer levels actually used are a subset of the 256 levels, not zeroing |
| Parameter Acquisition | Static | The scale/offset is counted from the calibration set and then fixed |
| Quantization Granularity | Weight per-channel; activation per-tensor | The same as W8A8 static quantization |
| Symmetry | Weight symmetric; activation symmetric or asymmetric | The same as W8A8 static quantization |
| Sparse Form | `W8A8S` / `W8A8SC` | `W8A8S` is the INT8 export with a narrowed codebook; `W8A8SC` is its secondary-compression deployment form |

### 2.2 Quantization Formula

Quantization and dequantization reuse the linear mapping of [W8A8 Static Quantization](term_w8a8_static.md). The difference from ordinary W8A8 is not in the formula but in the codebook occupancy: W8A8 can use roughly all 256 integer levels, whereas W8A8S uses only a subset of them, so the weight values have a narrower distribution range and are easier to compress. `W8A8SC` does not change this quantization mapping; it merely re-encodes the already narrowed INT8 weights into a form loadable by a compression unit, giving a higher compression ratio than W8A8.

### 2.3 Relationship to Other Modes

- **Relationship to [W8A8 Static Quantization](term_w8a8_static.md) (both are INT8 static, differing in whether the codebook is narrowed)**:
  - This mode: after INT8 quantization the weight uses only a subset of the 256 levels and is then compressed a second time by `W8A8SC`, giving a higher compression ratio than W8A8.
  - W8A8 static: can use the full-range 8bit levels, and the compression comes mainly from the 8bit bit width itself.

- **Relationship to [Float Sparse Quantization](term_float_sparse.md) (different compression path)**:
  - This mode: INT8 quantization is performed first, and the compression ratio is then raised by narrowing the codebook together with compression, rather than zeroing the weights.
  - Float sparse: the weights stay 16bit floating point and part of the values are zeroed, used to evaluate the accuracy loss introduced by zeroing sparsity alone.

### 2.4 Applicable Scenarios and Limitations

#### 2.4.1 Applicable Scenarios

- **Deployments needing a higher compression ratio than W8A8**: on top of INT8 quantization, a narrowed codebook is used to compress further.
- **Secondary-compression deployment**: when the target hardware provides a compression unit, `W8A8SC` is exported from `W8A8S`.

#### 2.4.2 Usage Limitations

- **Codebook narrowing sacrifices expressiveness**: fewer than 256 levels are actually available, so the accuracy is usually slightly lower than full-range W8A8 and sensitive layers may need to fall back.
- **Depends on compression hardware**: without a compression unit, `W8A8S`/`W8A8SC` bring almost no extra deployment benefit over W8A8.
- **Same calibration dependence as static quantization**: the activation is still per-tensor statically quantized and is sensitive to outliers.

---

## 3. Related Workflows

- *[One-Click Quantization Usage Guide](../../../user_guide/usage_one_click_quantization.md)*: select this mode via `--quant_type` or a YAML configuration and run quantization.
- *[Quantization Accuracy Tuning Guide](../../../user_guide/process_quantization_precision_tuning.md)*: accuracy degradation caused by this mode can be rolled back layer by layer and tuned through that workflow.

---

## 4. Related Terms

- [Quantization Modes](../README.md): the parent concept; this entry belongs to the [Linear Layer Quantization](README.md) category and is one concrete mode under it.
- [W8A8 Static Quantization](term_w8a8_static.md): a peer mode; this mode narrows the codebook and compresses a second time on top of its INT8 static quantization.
- [Float Sparse Quantization](term_float_sparse.md): a comparison mode, zeroing sparsity on 16bit floating point, with a different compression path.
- [Linear Quantization Algorithm](../../quantization_algorithms/linear_quant/term_linear_quant.md): a companion entry; the code implementation of this mode.

---

## 5. Reference Documents

1. *[Linear Quantization Parameter Configuration Workflow Guide](../../quantization_algorithms/linear_quant/usage_linear_quant.md)*
2. *[Quantization Modes](../README.md)*
