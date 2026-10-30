# W8A8 Static Quantization

> **Entry Category**: Quantization Data Format ([Linear Layer Quantization](README.md))<br>
> **English Name**: W8A8 Static Quantization<br>
> **Application Domain**: Large language model quantization and compression, inference acceleration<br>
> **Hosting IR Class**: `W8A8StaticFakeQuantLinear` ([`msmodelslim/ir/w8a8_static.py`](../../../../../msmodelslim/ir/w8a8_static.py))<br>

## 1. Overview

W8A8 static quantization applies [INT8](../../dtype_ops/term_int8.md) static quantization to both the weights and the activations of linear layers. The mode name is decoded according to the naming rule of [Quantization Modes](../README.md): W8A8 means that both the weights (W) and the activations (A) are 8bit ([INT8](../../dtype_ops/term_int8.md)); "static" means that the quantization parameters (scale/offset) are determined and fixed during the offline calibration stage and used directly at inference with zero online overhead. It is the most basic and most widely used linear layer quantization mode in [Linear Layer Quantization](README.md).

---

## 2. Term Description

### 2.1 Mode Specification

| Dimension | Value | Description |
|------|------|------|
| Quantization Target | Weight W, Activation A | The tensors on both sides of the linear layer matrix multiplication; the weight is a static parameter read in full on every token's forward pass, and the activation is an intermediate tensor generated per token |
| Bit Width | W 8bit, A 8bit | Both are 8bit, halving the memory traffic and enabling INT8 integer GEMM |
| Data Type | INT8 | Integer format, with the weight using 1 byte per element; in the decode phase (the inference phase that generates tokens one by one) the weight memory traffic is halved |
| Parameter Acquisition | Static | The scale/offset is computed from the calibration set (such as minmax) and then fixed, with zero online statistics overhead at inference; the accuracy depends on how well the calibration set matches the real distribution |
| Quantization Granularity | Weight per-channel; activation per-tensor | The weight shares one scale per output channel, adapting to numerical differences between channels; the activation shares the scale/offset across the whole tensor, minimizing the parameter overhead |
| Symmetry | Weight symmetric; activation symmetric or asymmetric | The weight distribution is naturally symmetric and needs no offset; the activation distribution is often asymmetric and needs a scale plus a zero-point offset |

### 2.2 Quantization Formula

Quantization and dequantization use the standard linear mapping. Symmetric quantization (scale only, no zero point):
$$q = \mathrm{round}(x / s), \qquad \hat{x} = q \cdot s, \qquad s = \frac{\max(|x|)}{2^{b-1}-1}$$

Asymmetric quantization (with zero-point offset):
$$q = \mathrm{round}(x / s) + z, \qquad \hat{x} = (q - z) \cdot s, \qquad s = \frac{\max(x) - \min(x)}{2^b - 1}$$

Here $x$ is the tensor to quantize, $q$ is the quantized integer value, $\hat{x}$ is the dequantized reconstruction, $s$ is the scale, $z$ is the zero-point offset, and $b$ is the bit width ($b=8$ for INT8 and $b=4$ for INT4), with $z = \mathrm{round}(-\min(x)/s)$. For how the quantization parameters $s$ and $z$ are obtained (static calibration or online dynamic) and their granularity (per-channel, per-token, per-tensor, and so on), see the mode specification table above.

### 2.3 Relationship to Other Modes

- **Relationship to [W8A8 Dynamic Quantization](term_w8a8_dynamic.md) (same bit width, different parameter acquisition)**:
  - This mode (static): the advantage is that the activation scale is fixed offline with zero online overhead at inference; the disadvantage is that the activation shares parameters across the whole tensor, a coarse granularity that is insensitive to per-token distribution changes and has a lower accuracy ceiling than the dynamic scheme.
  - Dynamic: the advantage is that the scale is computed online per token with no calibration, giving higher activation accuracy; the disadvantage is one extra min/max reduction on every forward pass, which adds a small amount of latency overhead.

- **Relationship to [W8A16 Static Quantization](term_w8a16_static.md) (whether the activation is quantized differs)**:
  - This mode: the advantage is that the activation is also quantized to INT8, enabling INT8 integer GEMM for higher compute throughput and more thorough compression; the disadvantage is that the activation introduces quantization error and accuracy degrades for layers with outliers or that are quantization-sensitive.
  - W8A16: the advantage is that the activation stays FP16 with zero quantization error, making it one of the most accurate baselines; the disadvantage is that not quantizing the activation means the matrix multiplication still has to dequantize the INT8 weights back to floating point, so there is no integer GEMM speedup, the performance is relatively poor, and the overall compression is lower than W8A8.

- **Relationship to [W4A4 Dynamic Quantization](term_w4a4_dynamic.md) (different bit width)**:
  - This mode: the advantage is that 8bit has higher resolution and more stable accuracy, making it the general deployment baseline; the disadvantage is that the weight memory traffic is only halved (1/2 of FP16).
  - W4A4: the advantage is that the weight memory traffic drops to 1/4 of FP16, the most thorough compression; the disadvantage is that signed INT4 can represent only 16 integer levels from -8 to 7, so the accuracy risk is high and dynamic quantization + per-group granularity + outlier suppression are needed to compensate.

### 2.4 Applicable Scenarios and Limitations

#### 2.4.1 Applicable Scenarios

- **General deployment baseline**: the most balanced mature scheme in accuracy and performance, suitable as the default quantization for most large language models.
- **Bandwidth-limited decode phase (generating tokens one by one)**: the full weight set is read per token, so the benefit of halving the INT8 weight memory traffic is most evident.

#### 2.4.2 Usage Limitations

- **Depends on calibration data**: an overly large calibration-set distribution bias will distort the scale and cause accuracy drift.
- **Sensitive to activation outliers**: per-tensor static quantization degrades in accuracy when the activation has significant outliers, so outlier suppression (such as [SmoothQuant](../../quantization_algorithms/smooth_quant/term_smooth_quant.md)) must be applied first or [dynamic quantization](term_w8a8_dynamic.md) used instead.

---

## 3. Related Workflows

- *[One-Click Quantization Usage Guide](../../../user_guide/usage_one_click_quantization.md)*: select this mode via `--quant_type` or a YAML configuration and run quantization.
- *[Quantization Accuracy Tuning Guide](../../../user_guide/process_quantization_precision_tuning.md)*: accuracy degradation caused by this mode can be rolled back layer by layer and tuned through that workflow.

---

## 4. Related Terms

- [Quantization Modes](../README.md): the parent concept; this entry belongs to the [Linear Layer Quantization](README.md) category and is one concrete mode under it.
- [W8A8 Dynamic Quantization](term_w8a8_dynamic.md): a comparison mode, with the activation quantization parameters computed online.
- [W8A8 Sparse Quantization](term_w8a8_sparse.md): a peer mode that narrows the codebook and compresses a second time on top of this mode's INT8 static quantization.
- [W8A16 Static Quantization](term_w8a16_static.md): a peer mode, with the activation kept at 16bit.
- [W4A4 Dynamic Quantization](term_w4a4_dynamic.md): a peer mode; lower-bit weight + activation quantization.
- [Linear Quantization Algorithm](../../quantization_algorithms/linear_quant/term_linear_quant.md): a companion entry; the code implementation of this mode.
- [MinMax Quantization Algorithm](../../quantization_algorithms/minmax/term_minmax.md): a companion entry; the quantization parameter statistics method commonly used for static quantization.

---

## 5. Reference Documents

1. *[Linear Quantization Parameter Configuration Workflow Guide](../../quantization_algorithms/linear_quant/usage_linear_quant.md)*
2. *[MinMax Parameter Configuration Workflow Guide](../../quantization_algorithms/minmax/usage_minmax.md)*
3. *[Quantization Modes](../README.md)*
