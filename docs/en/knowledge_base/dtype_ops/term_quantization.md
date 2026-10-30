# Quantization and Dequantization

> **Entry Category**: Quantization Data Type ([Quantization Data Type](README.md))<br>
> **English Name**: Quantization<br>
> **Application Domain**: Large language model quantization and compression, inference acceleration, KVCache compression<br>

## 1. Overview

**Quantization** maps a high-precision floating-point tensor onto a low-precision tensor, storing the same batch of data with fewer bits; **dequantization** is its inverse. This entry describes the quantization formula, scale / zero_point, static vs. dynamic, granularity, and symmetry. For how each data type maps onto these formulas, see [INT8](term_int8.md), [INT4](term_int4.md), [FP8](term_fp8.md), [MXFP8](term_mxfp8.md), and [MXFP4](term_mxfp4.md).

---

## 2. Term Description

### 2.1 Quantization Formula

For a floating-point value $x$, the quantization process is:

$$q = \mathrm{round}\left(\frac{x}{scale}\right) + zero\_point$$

- $x$: the floating-point value to be quantized.
- $scale$: the scaling factor, the floating-point step size corresponding to one integer level.
- $zero\_point$: the zero point, an optional offset introduced by asymmetric quantization.
- $q$: the integer after quantization.

**Example**: the values of a tensor are all in $[0, 12]$. Using asymmetric INT8 quantization (integer range $-128\sim127$) with $scale=12/255\approx0.047$ and $zero\_point=-128$:

- $x=12 \rightarrow q=\mathrm{round}(12/0.047)+(-128)=255-128=127$
- $x=6 \rightarrow q=\mathrm{round}(6/0.047)+(-128)\approx128-128=0$
- $x=0 \rightarrow q=\mathrm{round}(0)+(-128)=-128$

### 2.2 Dequantization Formula

$$x \approx (q - zero\_point) \times scale$$

- $q$: the quantized integer.
- $zero\_point$: the zero point.
- $scale$: the scaling factor.
- The restored value is an approximation of the original floating-point $x$.

In the example above, $q=0$ is restored to $(0-(-128))\times0.047\approx6.02$, close to the original value 6. Quantization inevitably causes information loss; the error comes from the rounding of `round` and the truncation of the value range by the limited bit width.

### 2.3 Why Quantize

- **Reduce memory and memory-access bandwidth**: weights drop from [FP16 / BF16](term_fp16_bf16.md) (2 bytes/element) to [INT8](term_int8.md) (1 byte/element), halving both the weight footprint and the read bandwidth; the same applies to KVCache.
- **Enable low-precision matrix computation**: once both weights and activations are quantized to integers, matrix multiplication can go through integer/low-precision operators (see [GEMM](../quantization_mode/term_gemm.md)), which brings computational gains on dedicated hardware.
- **Compress KVCache memory**: quantizing the cached historical K/V significantly reduces the memory footprint of long-context inference.

### 2.4 Quantization Parameters: scale and zero_point

The quantization parameters determine the mapping from floating-point values to integer levels:

- **scale (scaling factor)**: the floating-point step size represented by one integer level, $scale=\frac{\text{value range}}{\text{number of integer levels}}$. The value range is determined by the actual min/max of the tensor being quantized or of a local slice of it (such as a channel or a group of elements).
- **zero_point (zero point)**: the integer offset corresponding to floating-point 0 in asymmetric quantization; it is 0 in symmetric quantization and can be omitted.

### 2.5 How Parameters Are Obtained: Static vs. Dynamic

The **way of obtaining** quantization parameters falls into two categories:

- **Static quantization**: scale/zero_point are computed offline by statistics over a batch of calibration data and used directly during inference (the parameters are fixed). The online overhead is zero, but it depends on how well the calibration set characterizes the real distribution.
- **Dynamic quantization**: the actual range of each tensor (or each row) is computed online during inference and scale is calculated in real time. It needs no calibration and adapts well to distribution shifts, but costs one extra statistical reduction each time.

### 2.6 Quantization Granularity

Quantization granularity refers to **the range of elements covered by one set of quantization parameters**. The finer the granularity, the more accurate the characterization of the value distribution and the better the accuracy, but the larger the number of quantization parameters and the computational overhead. Common granularities and their characteristics are as follows:

- **per-tensor**: the whole tensor shares one scale.
- **per-channel**: one set of parameters per output channel (such as the out_dim column of the weight W), commonly used for weights.
- **per-group**: one set of parameters per fixed-size group (such as 128 elements), for example the group-wise quantization of [GPTQ](../quantization_algorithms/gptq/term_gptq.md)/[LAOS](../quantization_algorithms/laos/term_laos.md).
- **per-token**: one set of parameters per input row (each token), commonly used for dynamic quantization of activations.
- **per-head**: one set of parameters per attention head, used for attention-related tensors.
- **per-block**: a shared block-level exponent over a fixed-size block (such as the 32 elements of [MXFP8](term_mxfp8.md)/[MXFP4](term_mxfp4.md)); it is a dedicated granularity of MX formats. Unlike per-group, which records a scale per group, per-block shares an exponent.

### 2.7 Symmetry and Asymmetry

- **Symmetric quantization**: values are centered on 0; only scale is recorded, with no zero_point. It is simple to implement and suits weights whose distribution is approximately symmetric.
- **Asymmetric quantization**: both scale and zero_point are recorded, which can cover values with an overall offset and achieves higher utilization of the dynamic range.

---

## 3. Related Workflows

- [Quantization Data Type ops Usage Workflow Guide](usage_dtype_ops.md): maps the formulas in this entry onto function calls.
- [One-Click Quantization Usage Guide](../../user_guide/usage_one_click_quantization.md): the execution entry point for quantization tasks.
- [Quantization Accuracy Tuning Guide](../../user_guide/process_quantization_precision_tuning.md): accuracy problems caused by the data type and the quantization scheme can be diagnosed and tuned through this workflow.

---

## 4. Related Terms

- [Quantization Data Type](README.md): the higher-level concept; the directory this entry belongs to.
- [FP16 / BF16](term_fp16_bf16.md): a companion term; the 16-bit baseline before quantization.
- [GEMM](../quantization_mode/term_gemm.md): a companion term; quantization is the prerequisite of integer GEMM.
- [INT8](term_int8.md): a companion term; the most common target format of quantization.
- [Quantization Modes](../quantization_mode/README.md): the upper-layer concept; the concrete implementation of quantization/dequantization in each mode.

---

## 5. Reference Documents

1. IEEE. "IEEE Standard for Floating-Point Arithmetic." IEEE 754-2019, 2019. <https://ieeexplore.ieee.org/document/8766229>
2. Jacob B, et al. "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference." CVPR, 2018. <https://arxiv.org/abs/1712.05877>
