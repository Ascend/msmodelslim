# Quantization and Dequantization

> **Entry Category**: Quantization Basic Concept ([Quantization Basics](README.md))<br>
> **English Name**: Quantization / Dequantization<br>
> **Application Domain**: Large language model quantization and compression, inference acceleration, KVCache compression<br>

## 1. Overview

**Quantization** maps a high-precision floating-point tensor to a low-precision tensor, storing the same data with fewer bits; **dequantization** is its inverse process. This entry describes the quantization/dequantization formulas, the quantization parameters (scale and zero_point), how the parameters are obtained (static/dynamic), the quantization granularity, and symmetry.

---

## 2. Term Description

### 2.1 Quantization Formula

For a floating-point value $x$, the quantization process is:

$$q = \mathrm{round}\left(\frac{x}{scale}\right) + zero\_point$$

where $scale$ (the scaling factor) maps the floating-point range onto integer levels, $zero\_point$ (the zero point) is an optional offset introduced by asymmetric quantization, and $q$ is the quantized integer.

**Example**: a tensor whose values all lie in $[0, 12]$ is quantized with asymmetric INT8 (integer range $-128\sim127$), taking $scale=12/255\approx0.047$ and $zero\_point=-128$:

- $x=12 \rightarrow q=\mathrm{round}(12/0.047)+(-128)=255-128=127$
- $x=6 \rightarrow q=\mathrm{round}(6/0.047)+(-128)\approx128-128=0$
- $x=0 \rightarrow q=\mathrm{round}(0)+(-128)=-128$

### 2.2 Dequantization Formula

$$x \approx (q - zero\_point) \times scale$$

In the example above, $q=0$ is restored to $(0-(-128))\times0.047\approx6.02$, close to the original value 6. Quantization inevitably brings information loss: the error comes from the rounding of `round` and from the truncation of the value range by the limited bit width.

### 2.3 Why Quantize

- **Reduce memory and memory access bandwidth**: weights drop from [FP16/BF16](term_fp16_bf16.md) (2 bytes per element) to [INT8](term_int8.md) (1 byte per element), halving the weight footprint and the read bandwidth; the same applies to the KVCache.
- **Enable low-precision matrix computation**: once both weights and activations are quantized to integers, the matrix multiplication can use integer/low-precision operators (see [GEMM](../quantization_mode/term_gemm.md)), giving a computational benefit on dedicated hardware.
- **Compress the KVCache memory footprint**: quantizing the cached historical K/V significantly reduces the memory footprint of long-context inference.

### 2.4 Quantization Parameters: scale and zero_point

The quantization parameters determine the mapping from floating-point values to integer levels:

- **scale (the scaling factor)**: the floating-point step represented by one integer level, $scale=\frac{\text{value range}}{\text{number of integer levels}}$. The value range is determined by the actual min/max of the quantized tensor or of a local slice of it (such as a channel or a group of elements).
- **zero_point (the zero point)**: in asymmetric quantization, the integer offset corresponding to floating-point 0; in symmetric quantization it is 0 and can be omitted.

### 2.5 Parameter Acquisition: Static and Dynamic

The **acquisition method** of the quantization parameters falls into two categories:

- **Static quantization**: scale/zero_point are computed offline from a batch of calibration data and used directly at inference time (the parameters are fixed). The online overhead is zero, but it relies on how well the calibration set characterizes the real distribution.
- **Dynamic quantization**: the actual range of each tensor (or each row) is computed online at inference time, and scale is calculated in real time. It needs no calibration and adapts well to distribution shifts, but adds the overhead of an extra statistical reduction each time.

### 2.6 Quantization Granularity

Quantization granularity refers to **the range of elements covered by one set of quantization parameters**. The finer the granularity, the more accurately the value distribution is characterized and the better the accuracy, but the greater the number of quantization parameters and the computational overhead. The common granularities and their characteristics are as follows:

- **per-tensor**: the whole tensor shares one scale.
- **per-channel**: one set of parameters per output channel (such as the out_dim column of the weight W); commonly used for weights.
- **per-group**: one set of parameters per fixed-size group (such as 128 elements), as in the group quantization of [GPTQ](../quantization_algorithms/gptq/term_gptq.md)/[LAOS](../quantization_algorithms/laos/term_laos.md).
- **per-token**: one set of parameters per input row (each token); commonly used for the dynamic quantization of activations.
- **per-head**: one set of parameters per attention head, used for attention-related tensors.
- **per-block**: one block-level exponent shared per fixed-size block (such as the 32 elements of [MXFP8/MXFP4](term_mxfp.md)); it is the dedicated granularity of the MX format. Unlike per-group, which records a scale for each group, per-block shares the exponent.

### 2.7 Symmetry and Asymmetry

- **Symmetric quantization**: the values are centered on 0, and only scale is recorded, with no zero_point. It is simple to implement and suits weights whose distribution is approximately symmetric.
- **Asymmetric quantization**: both scale and zero_point are recorded, so values with an overall offset can be covered and the dynamic range is used more efficiently.

---

## 3. Related Workflows

- [One-Click Quantization Usage Guide](../../user_guide/usage_one_click_quantization.md): the execution entry point for quantization tasks.
- [Quantization Accuracy Tuning Guide](../../user_guide/process_quantization_precision_tuning.md): accuracy problems caused by the data type and the quantization scheme can be diagnosed and tuned through this workflow.

---

## 4. Related Terms

- [Quantization Basics](README.md): The parent concept and the directory this entry belongs to.
- [GEMM](../quantization_mode/term_gemm.md): A companion term; quantization is the prerequisite for an integer GEMM.
- [INT8](term_int8.md): A companion term; the most commonly used target format for quantization.
- [Quantization Modes](../quantization_mode/README.md): A higher-level concept; the concrete implementation of quantization/dequantization in each mode.
