# W8A8 PD-Mix Quantization

> **Entry Category**: Quantization Data Format ([Linear Layer Quantization](README.md))<br>
> **English Name**: W8A8 PD-Mix Quantization<br>
> **Application Domain**: Large language model quantization and compression, inference acceleration<br>
> **Hosting IR Class**: `W8A8PDMixFakeQuantLinear` ([`msmodelslim/ir/w8a8_pdmix.py`](../../../../../msmodelslim/ir/w8a8_pdmix.py))<br>

## 1. Overview

W8A8 PD-Mix quantization is a W8A8 scheme that mixes two activation quantization strategies split by the **Prefill / Decode inference phase**: in the Prefill phase the activation is dynamically quantized per token (preserving accuracy), and in the Decode phase the activation is statically quantized as a whole (zero online overhead). It is a phased fusion of [W8A8 dynamic](term_w8a8_dynamic.md) and [W8A8 static](term_w8a8_static.md), where PD-Mix means Prefill-Decode Mix.

---

## 2. Term Description

### 2.1 Mode Specification

| Dimension | Value | Description |
|------|------|------|
| Quantization Target | Weight W, Activation A | The tensors on both sides of the linear layer matrix multiplication |
| Bit Width | W 8bit, A 8bit | Both 8bit; memory traffic is halved |
| Data Type | INT8 | Integer format; can use INT8 integer GEMM |
| Parameter Acquisition | Mixed (the activation is dynamic/static switching by phase; the weight is static) | In the Prefill phase the activation computes min/max online per token to derive the scale; in the Decode phase the activation is statically quantized as a whole (reusing preset parameters; at that point a single token's activation is the whole tensor, so the granularity is equivalent to per-token); the weight scale is fixed |
| Quantization Granularity | Weight per-channel; activation Prefill per-token / Decode per-tensor | The weight shares a scale per output channel; the activation takes per-token or whole-tensor granularity depending on the phase |
| Symmetry | Weight symmetric; activation asymmetric | The asymmetric activation adds an offset to adapt to each token's distribution |

### 2.2 Quantization Formula

Quantization and dequantization use the standard linear mapping. Symmetric quantization (scale only, no zero point):
$$q = \mathrm{round}(x / s), \qquad \hat{x} = q \cdot s, \qquad s = \frac{\max(|x|)}{2^{b-1}-1}$$

Asymmetric quantization (with zero-point offset):
$$q = \mathrm{round}(x / s) + z, \qquad \hat{x} = (q - z) \cdot s, \qquad s = \frac{\max(x) - \min(x)}{2^b - 1}$$

Here $x$ is the tensor to quantize, $q$ is the quantized integer value, $\hat{x}$ is the dequantized reconstruction, $s$ is the scale, $z$ is the zero-point offset, and $b$ is the bit width ($b=8$ for INT8 and $b=4$ for INT4), with $z = \mathrm{round}(-\min(x)/s)$. For how the quantization parameters $s$ and $z$ are obtained (static calibration or online dynamic) and their granularity (per-channel, per-token, per-tensor, and so on), see the mode specification table above.

### 2.3 Relationship to Other Modes

- **Relationship to [W8A8 Dynamic Quantization](term_w8a8_dynamic.md) (both are W8A8, with different activation strategies)**:
  - This mode: the advantage is that the decode phase falls back to static, removing the pointless per-token reduction overhead at every step, while the prefill phase still preserves dynamic accuracy; the disadvantage is that the inference framework must recognize the phase and switch the quantization strategy, adding complexity and deployment constraints.
  - W8A8 dynamic: the advantage is per-token dynamic throughout with a uniform activation accuracy and a simple implementation; the disadvantage is that the single-token reduction in the decode phase is both statistically meaningless and adds latency.

- **Relationship to [W8A8 Static Quantization](term_w8a8_static.md) (both are W8A8, with different activation strategies)**:
  - This mode: the advantage is per-token dynamic in the prefill phase, giving higher activation accuracy than static per-tensor; the disadvantage is higher complexity and a phase dependency.
  - Static: the advantage is that the activation scale is fixed throughout with zero online reduction and the simplest implementation and deployment; the disadvantage is the coarse per-tensor granularity in the prefill phase and the low accuracy ceiling.

- **Relationship to [W4A8 Dynamic Quantization](term_w4a8_dynamic.md) (different bit width)**:
  - This mode: the advantage is more stable 8bit weight accuracy; the disadvantage is that the weight memory traffic is only halved.
  - W4A8: the advantage is that the weight is compressed to 1/4 of FP16; the disadvantage is that signed INT4 can represent only 16 integer levels from -8 to 7, so there is an accuracy risk.

### 2.4 Applicable Scenarios and Limitations

#### 2.4.1 Applicable Scenarios

- **Server-side inference valuing both Prefill and Decode**: scenarios where the prefill phase needs activation accuracy and the decode phase needs low latency and low overhead, such as incremental generation in long-context serving.
- **Models whose activation distribution varies markedly by phase**: models where the prefill activation range is large and discrete while the decode activation is relatively stable.

#### 2.4.2 Usage Limitations

- **Depends on phase awareness**: switching the quantization strategy requires the inference framework to recognize whether the current phase is Prefill or Decode, and both phases share the same INT8 data flow.
- **Currently limited to MindIE deployment**: phase awareness and switching currently take effect only under the MindIE inference framework.
- **INT8 only**: this mode does not provide INT4/FP8 data type variants.

---

## 3. Related Workflows

- *[One-Click Quantization Usage Guide](../../../user_guide/usage_one_click_quantization.md)*: select this mode via `--quant_type` or a YAML configuration and run quantization.
- *[Quantization Accuracy Tuning Guide](../../../user_guide/process_quantization_precision_tuning.md)*: accuracy degradation caused by this mode can be rolled back layer by layer and tuned through that workflow.

---

## 4. Related Terms

- [Quantization Modes](../README.md): the parent concept; this entry belongs to the [Linear Layer Quantization](README.md) category and is one concrete mode under it.
- [W8A8 Dynamic Quantization](term_w8a8_dynamic.md): a comparison mode, per-token dynamic throughout.
- [W8A8 Static Quantization](term_w8a8_static.md): a comparison mode, static throughout.
- [W4A8 Dynamic Quantization](term_w4a8_dynamic.md): a comparison mode, the mixed bit-width scheme with a 4bit weight.
- [PDMIX Inter-Phase Mixed Quantization Algorithm](../../quantization_algorithms/pdmix/term_pdmix.md): a companion term, the algorithm processor documentation corresponding to this mode.
- [Linear Quantization Algorithm](../../quantization_algorithms/linear_quant/term_linear_quant.md): a companion entry; the code implementation of linear layer quantization.

---

## 5. Reference Documents

1. *[Linear Quantization Parameter Configuration Workflow Guide](../../quantization_algorithms/linear_quant/usage_linear_quant.md)*
2. *[PDMIX Parameter Configuration Workflow Guide](../../quantization_algorithms/pdmix/usage_pdmix.md)*
3. *[Quantization Modes](../README.md)*
