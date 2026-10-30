# W8A16 Static Quantization

> **Entry Category**: Quantization Data Format ([Linear Layer Quantization](README.md))<br>
> **English Name**: W8A16 Static Quantization<br>
> **Application Domain**: Large language model quantization and compression, inference acceleration<br>
> **Hosting IR Class**: `W8A16StaticPerChannelFakeQuantLinear` / `W8A16StaticPerGroupFakeQuantLinear` ([`msmodelslim/ir/w8a16_static.py`](../../../../../msmodelslim/ir/w8a16_static.py))<br>

## 1. Overview

W8A16 static quantization compresses only the linear layer weight to [INT8](../../dtype_ops/term_int8.md) while keeping the activation at [FP16/BF16](../../dtype_ops/term_fp16_bf16.md) unquantized. It targets scenarios where the activation is accuracy-sensitive and should not be quantized. The weight is static data that must be read in full for every token, so compressing it to [INT8](../../dtype_ops/term_int8.md) halves the weight memory traffic; not quantizing the activation fully avoids activation quantization error, making it one of the most accurate linear layer schemes. The cost is that not quantizing the activation means the matrix multiplication must dequantize the [INT8](../../dtype_ops/term_int8.md) weights back to floating point, with no [integer GEMM](../term_gemm.md) speedup.

---

## 2. Term Description

### 2.1 Mode Specification

| Dimension | Value | Description |
|------|------|------|
| Quantization Target | Weight W only | The activation participates in the computation as `FP16`, introducing no quantization error at all |
| Bit Width | W 8bit, A 16bit | The weight uses 1 byte per element with halved memory traffic; the activation uses 2 bytes per element |
| Data Type | Weight INT8; activation `FP16` | Mixed bit width; the entire compression benefit comes from the weight side |
| Parameter Acquisition | Static | The weight scale/offset is fixed during quantization with zero online overhead at inference; the activation has no parameters |
| Quantization Granularity | Weight `per-channel`/`per-group` | The scale is shared per output channel or per fixed group (such as 128 elements), adapting to numerical differences between channels/groups |
| Symmetry | Weight symmetric or asymmetric | Chosen according to the weight distribution; asymmetric adds an offset |

### 2.2 Quantization Formula

Quantization and dequantization use the standard linear mapping. Symmetric quantization (scale only, no zero point):
$$q = \mathrm{round}(x / s), \qquad \hat{x} = q \cdot s, \qquad s = \frac{\max(|x|)}{2^{b-1}-1}$$

Asymmetric quantization (with zero-point offset):
$$q = \mathrm{round}(x / s) + z, \qquad \hat{x} = (q - z) \cdot s, \qquad s = \frac{\max(x) - \min(x)}{2^b - 1}$$

Here $x$ is the tensor to quantize, $q$ is the quantized integer value, $\hat{x}$ is the dequantized reconstruction, $s$ is the scale, $z$ is the zero-point offset, and $b$ is the bit width ($b=8$ for INT8, $b=4$ for INT4), with $z = \mathrm{round}(-\min(x)/s)$. For how the quantization parameters $s$ and $z$ are obtained (static calibration) and their granularity (`per-channel`, `per-group`), see the mode specification table above. In this mode only the weight follows the formula above ($b=8$); the activation stays `FP16` and is not quantized.

### 2.3 Relationship to Other Modes

- **Relationship to [W8A8 Static Quantization](term_w8a8_static.md) (whether the activation is quantized differs)**:
  - This mode: the advantage is that the activation stays `FP16` with zero quantization error, making it one of the most accurate baselines and suitable for layers whose activations have outliers or are quantization-sensitive; the disadvantage is that not quantizing the activation means the matrix multiplication must dequantize the [INT8](../../dtype_ops/term_int8.md) weights back to floating point, so there is no integer GEMM compute speedup, the performance is relatively poor, and the overall compression is lower than `W8A8`.
  - `W8A8`: the advantage is that the activation is also quantized to INT8, enabling integer GEMM for higher compute throughput and more thorough compression; the disadvantage is that the activation introduces quantization error.

- **Relationship to [W4A8 Dynamic Quantization](term_w4a8_dynamic.md) (different bit-width pairing)**:
  - This mode: the advantage is that the activation has no quantization error at all and there is no online reduction overhead; the disadvantage is that the weight is only compressed to 8bit, so the memory-traffic benefit is limited and there is no compute speedup.
  - `W4A8`: the advantage is that the weight is compressed to 4bit and the memory traffic drops to 1/4 of `FP16`; the disadvantage is that signed INT4 can represent only 16 integer levels from -8 to 7, so there is an accuracy risk, and a dynamic activation also introduces reduction overhead.

- **Relationship to [Float Sparse Quantization](term_float_sparse.md) (both are high-accuracy baselines, with different sources of benefit)**:
  - This mode: the benefit comes from weight bit-width compression (halved memory traffic).
  - Float sparse: neither the weight nor the activation is quantized, and the benefit comes from numerical sparsity (skipping zero-valued elements); it is the sparse comparison baseline once bit-width quantization has been removed.

### 2.4 Applicable Scenarios and Limitations

#### 2.4.1 Applicable Scenarios

- **Activation-accuracy-sensitive layers**: layers whose activation distribution has many outliers and whose accuracy degrades markedly after quantization can be rolled back to `W8A16` individually.
- **Weight-memory-dominated decode scenarios (the inference phase that generates tokens one by one)**: weight compression halves the memory traffic while preserving activation accuracy.
- **Accuracy-first deployment**: for scenarios needing deterministic accuracy, `W8A16` is a safe choice between no quantization and `W8A8`.

#### 2.4.2 Usage Limitations

- **No integer GEMM speedup**: the activation stays `FP16`, so the matrix multiplication remains floating point, the compute-throughput gain is limited, and only the weight memory traffic is saved.
- **Lower compression ratio than `W8A8`**: the activation is not compressed, so the overall bit-width benefit is lower than schemes that quantize both the weight and the activation.

---

## 3. Related Workflows

- *[One-Click Quantization Usage Guide](../../../user_guide/usage_one_click_quantization.md)*: select this mode via `--quant_type` or a `YAML` configuration and run quantization.
- *[Quantization Accuracy Tuning Guide](../../../user_guide/process_quantization_precision_tuning.md)*: accuracy degradation caused by this mode can be rolled back layer by layer and tuned through that workflow.

---

## 4. Related Terms

- [Quantization Modes](../README.md): the parent concept; this entry belongs to the [Linear Layer Quantization](README.md) category and is one concrete mode under it.
- [W8A8 Static Quantization](term_w8a8_static.md): a comparison mode, in which the activation is also quantized to INT8 in exchange for integer GEMM speedup.
- [W4A8 Dynamic Quantization](term_w4a8_dynamic.md): a comparison mode, the mixed bit-width scheme compressing the weight to 4bit.
- [Float Sparse Quantization](term_float_sparse.md): a peer mode, the sparse baseline with no quantization at all.
- [Linear Quantization Algorithm](../../quantization_algorithms/linear_quant/term_linear_quant.md): a companion entry; the code implementation of this mode.
- [MinMax Quantization Algorithm](../../quantization_algorithms/minmax/term_minmax.md): a companion entry; the quantization parameter statistics method commonly used for static quantization.

---

## 5. Reference Documents

1. *[Linear Quantization Parameter Configuration Workflow Guide](../../quantization_algorithms/linear_quant/usage_linear_quant.md)*
2. *[MinMax Parameter Configuration Workflow Guide](../../quantization_algorithms/minmax/usage_minmax.md)*
3. *[Quantization Modes](../README.md)*
