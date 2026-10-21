# W4A4 Dynamic Quantization

> **Entry Category**: Quantization Data Format ([Linear Layer Quantization](README.md))<br>
> **English Name**: W4A4 Dynamic Quantization<br>
> **Application Domain**: Large language model quantization and compression, inference acceleration<br>
> **Hosting IR Class**: `W4A4DynamicPerChannelFakeQuantLinear` / `W4A4DynamicPerGroupFakeQuantLinear` ([`msmodelslim/ir/w4a4_dynamic.py`](../../../../../msmodelslim/ir/w4a4_dynamic.py))<br>

## 1. Overview

W4A4 dynamic quantization applies [INT4](../../quantization_basic/term_int4.md) quantization to both the weights and the activations of linear layers, with the activation quantization parameters computed online per token. It is the linear layer scheme with the highest compression ratio: the weight memory traffic drops to 1/4 of [FP16/BF16](../../quantization_basic/term_fp16_bf16.md), but signed INT4 can represent only 16 integer levels from -8 to 7, so per-token dynamic quantization together with per-channel/per-group fine granularity is indispensable to keep the accuracy loss within an acceptable range.

---

## 2. Term Description

### 2.1 Mode Specification

| Dimension | Value | Description |
|------|------|------|
| Quantization Target | Weight W, Activation A | The tensors on both sides of the linear layer matrix multiplication |
| Bit Width | W 4bit, A 4bit | Both 4bit; the weight memory traffic drops to 1/4 of [FP16/BF16](../../quantization_basic/term_fp16_bf16.md) and 1/2 of INT8 |
| Data Type | INT4 | Integer format (stored internally in an INT8 container) |
| Parameter Acquisition | Activation dynamic / weight static | The activation computes min/max online per token to derive the scale; the weight scale/offset is fixed during quantization |
| Quantization Granularity | Weight per-channel/per-group; activation per-token | The weight shares the scale/offset per output channel or per fixed group (such as 128 elements); the activation shares the scale per token |
| Symmetry | Activation symmetric; weight symmetric or asymmetric | The activation is symmetric with only a scale; the weight can add an offset to adapt to an asymmetric distribution |

### 2.2 Quantization Formula

Quantization and dequantization use the standard linear mapping. Symmetric quantization (scale only, no zero point):
$$q = \mathrm{round}(x / s), \qquad \hat{x} = q \cdot s, \qquad s = \frac{\max(|x|)}{2^{b-1}-1}$$

Asymmetric quantization (with zero-point offset):
$$q = \mathrm{round}(x / s) + z, \qquad \hat{x} = (q - z) \cdot s, \qquad s = \frac{\max(x) - \min(x)}{2^b - 1}$$

Here $x$ is the tensor to quantize, $q$ is the quantized integer value, $\hat{x}$ is the dequantized reconstruction, $s$ is the scale, $z$ is the zero-point offset, and $b$ is the bit width ($b=8$ for INT8 and $b=4$ for INT4), with $z = \mathrm{round}(-\min(x)/s)$. For how the quantization parameters $s$ and $z$ are obtained (static calibration or online dynamic) and their granularity (per-channel, per-token, per-tensor, and so on), see the mode specification table above.

### 2.3 Relationship to Other Modes

- **Relationship to [W8A8 Static Quantization](term_w8a8_static.md) (different bit width)**:
  - This mode: the advantage is that the weight memory traffic drops to 1/4 of [FP16/BF16](../../quantization_basic/term_fp16_bf16.md), maximizing the compression and bandwidth benefit; the disadvantage is that signed INT4 can represent only 16 integer levels from -8 to 7, so the accuracy risk is high and dynamic quantization + per-group granularity + outlier suppression are needed to compensate.
  - W8A8: the advantage is that 8bit has higher resolution and more stable accuracy, making it the general deployment baseline; the disadvantage is that the weight memory traffic is only halved.

- **Relationship to [W8A8 Dynamic Quantization](term_w8a8_dynamic.md) (both are in the dynamic family, with different bit widths)**:
  - This mode: the advantage is heavier compression and a lower compute bit width; the disadvantage is that a 4bit activation is extremely sensitive to the distribution, so the accuracy risk is markedly higher than at 8bit.
  - W8A8 dynamic: the advantage is more stable 8bit accuracy and lower online reduction and overall overhead; the disadvantage is the smaller compression margin.

- **Relationship to [W4A4 MX Dynamic Quantization](term_w4a4_mx_dynamic.md) (both are W4A4, with different data types)**:
  - This mode (INT4): the advantage is the integer format with a mature hardware operator ecosystem; the disadvantage is that uniform leveling is sensitive to outliers and the dynamic range is limited.
  - MXFP4: the advantage is that a block-level shared exponent preserves the floating-point dynamic range, making it more tolerant of outliers; the disadvantage is the dependence on hardware MX support and the need to align the block granularity.

### 2.4 Applicable Scenarios and Limitations

#### 2.4.1 Applicable Scenarios

- **High-compression-ratio deployment**: scenarios limited by memory or bandwidth and pursuing extreme compression, such as large-scale server-side multi-model deployment.
- **Accuracy-insensitive tasks**: scenarios where the task itself tolerates quantization noise well.

#### 2.4.2 Usage Limitations

- **High accuracy bar**: 4bit activation quantization carries a large accuracy risk and usually requires outlier suppression (such as [SmoothQuant](../../quantization_algorithms/smooth_quant/term_smooth_quant.md)) together with accuracy tuning before it can go live.
- **Hardware operator dependency**: INT4 integer GEMM requires support from the target hardware operator library; otherwise the compute benefit cannot be realized.

---

## 3. Related Workflows

- *[One-Click Quantization Usage Guide](../../../user_guide/usage_one_click_quantization.md)*: select this mode via `--quant_type` or a YAML configuration and run quantization.
- *[Quantization Accuracy Tuning Guide](../../../user_guide/process_quantization_precision_tuning.md)*: accuracy degradation caused by this mode can be rolled back layer by layer and tuned through that workflow.

---

## 4. Related Terms

- [Quantization Modes](../README.md): the parent concept; this entry belongs to the [Linear Layer Quantization](README.md) category and is one concrete mode under it.
- [W8A8 Dynamic Quantization](term_w8a8_dynamic.md): a comparison mode, INT8 bit width with more stable accuracy.
- [W8A8 Static Quantization](term_w8a8_static.md): a comparison mode, the INT8 static scheme and accuracy baseline.
- [W4A4 MX Dynamic Quantization](term_w4a4_mx_dynamic.md): a peer mode, switched to the MXFP4 floating-point format.
- [W8A16 Static Quantization](term_w8a16_static.md): a comparison mode, the high-accuracy scheme that keeps the activation at 16bit.
- [SVDQuant Quantization](term_svdquant.md): a comparison mode that absorbs outliers by low-rank decomposition and quantizes only the residual component, lowering the 4bit accuracy risk.
- [Linear Quantization Algorithm](../../quantization_algorithms/linear_quant/term_linear_quant.md): a companion entry; the code implementation of this mode.
- [LAOS Low-Bit Quantization Scheme](../../quantization_algorithms/laos/term_laos.md): a companion term, an accuracy optimization algorithm for W4A4.

---

## 5. Reference Documents

1. *[Linear Quantization Parameter Configuration Workflow Guide](../../quantization_algorithms/linear_quant/usage_linear_quant.md)*
2. *[LAOS Parameter Configuration Workflow Guide](../../quantization_algorithms/laos/usage_laos.md)*
3. *[Quantization Modes](../README.md)*
