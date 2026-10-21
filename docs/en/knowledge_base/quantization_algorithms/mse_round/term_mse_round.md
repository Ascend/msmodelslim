# MSE_Round Weight Quantization Algorithm Terminology Entry

> **Entry Category**: [Quantization Algorithms](../README.md#2-quantization-algorithms)<br>
> **English Name**: mse_round<br>
> **Application Domain**: MXFP8 weight quantization, low-bit quantization accuracy optimization<br>

---

## 1. Overview

MSE_Round is a shared exponent selection algorithm for MXFP8 per-block weight quantization. It tries adjacent exponent rounding candidates for each block separately and selects the better scale by the mean squared error of the actual quantization-dequantization, reducing the block-level reconstruction error caused by a fixed rounding rule; its core characteristics are per-block candidate comparison, MSE-based decisions, and scale optimization without extra training.

---

## 2. Term Description

In MXFP8 per-block quantization, the traditional minmax method fixedly uses $s = \lfloor \log_2(\max(|x|)) \rfloor - e_{\text{max}}$ to compute the shared exponent, after which the magnitude of the block maximum falls in $[8, 16)$. When the block maximum approaches the upper bound, the scaled result may exceed the MXFP8 representation ceiling ($448$), causing clipping and introducing a large quantization error. MSE_Round compares the actual quantization error of the ceil and floor candidates per block, achieving a better balance between avoiding clipping and preserving accuracy.

### 2.1 Core Idea

The core idea of MSE_Round is "per-block best-of-two-candidates by MSE": for each block it computes both the ceil and floor candidate shared exponents, performs quantization-dequantization for each and computes the in-block MSE, and selects the shared exponent with the smaller MSE as the final quantization parameter. The ceil candidate avoids clipping large values, while the floor candidate may achieve a better overall MSE when the distribution is uniform.

Per-block comparison adapts to different distributions: spiky blocks may need the range headroom of ceil, while blocks with a dense bulk and no extreme maximum may benefit from the finer step size of floor.

### 2.2 Working Mechanism

The shared exponent of block floating-point quantization can only take discrete integers. If a continuous ideal exponent is inferred from the block maximum, it usually lies between the two adjacent integers floor and ceil. Flooring corresponds to a smaller power-of-two scale with a finer quantization step but makes it easier for large values to exceed the representable range of the format; ceiling gives more dynamic-range headroom but coarsens the quantization interval of all values. MSE_Round does not presuppose which side is better; it constructs both candidates at the same time.

For each block, the algorithm fully performs quantization-dequantization using the floor exponent and the ceil exponent respectively, and computes the reconstruction MSE of that block; as long as the candidate exponent is within the range allowed by the format, the one with the smaller error is selected. Because the comparison happens after the real FP lattice and the real saturation rules, it directly incorporates the two opposing effects of "coarser step size" and "less clipping" into the decision. Compared with a method that uses a single global empirical boundary, this is a per-block local choice between two options.

### 2.3 Mathematical Description

Conventional floor scaling:

$$
s_{\text{floor}} = \lfloor \log_2(\max(|x|)) \rfloor - e_{\text{max}}
$$

MSE_Round computes both the ceil and floor candidates at the same time:

$$
s_{\text{ceil}} = \lceil \log_2(\max(|x|)) \rceil - e_{\text{max}}
$$

$$
s_{\text{floor}} = \lfloor \log_2(\max(|x|)) \rfloor - e_{\text{max}}
$$

Compute the in-block MSE for each:

$$
\text{MSE}_{\text{ceil}} = \frac{1}{N}\sum_{i=1}^{N}(x_i - \hat{x}_i(s_{\text{ceil}}))^2, \quad \text{MSE}_{\text{floor}} = \frac{1}{N}\sum_{i=1}^{N}(x_i - \hat{x}_i(s_{\text{floor}}))^2
$$

Final per-block selection:

$$
s^* = \begin{cases} s_{\text{ceil}} & \text{if } \text{MSE}_{\text{ceil}} < \text{MSE}_{\text{floor}} \\ s_{\text{floor}} & \text{otherwise} \end{cases}
$$

- $x$: the weight values within the block
- $\hat{x}(s)$: the value after quantization-dequantization using shared exponent $s$
- $N$: number of elements within a block
- $e_{\text{max}}$: the exponent bias; for the MXFP8 E4M3 format, $e_{\text{max}} = 2^{e_{\text{bits}}-1} = 8$

When a candidate's shared exponent falls outside the E8M0 representable range (marked as NaN), it automatically falls back to the other valid candidate.

If the continuous logarithmic scale is $r=\log_2 s_{ideal}$ and the integer shared exponent candidates are $e_f=\lfloor r\rfloor$ and $e_c=\lceil r\rceil$, then:

$$
\hat x^{(e)}=2^e\,\mathcal F(x/2^e),\qquad e\in\{e_f,e_c\},
$$

$$
e^*=\arg\min_{e\in\{e_f,e_c\}}\sum_i(x_i-\hat x_i^{(e)})^2.
$$

Here $\mathcal F$ is the in-block low-bit floating-point code table projection. Because the code table is non-uniform and may saturate, the MSE cannot be judged merely from whether $r$ is closer to floor or ceil; the real QDQ must be compared.

### 2.4 Key Properties

- **Two-candidate scaling**: per-block best-of-two by MSE between the ceil/floor options.
- **Clipping avoidance**: the ceil candidate compresses the scaled range to $(4, 8]$ to avoid clipping large values.
- **Finite candidate search**: only the adjacent floor/ceil exponents are compared; no continuous scale optimization is needed.
- **Controllable computation**: each block performs two quantization-dequantization evaluations, so the computational cost is about twice that of standard minmax.
- **Per-block discrete decision**: each block independently chooses the smaller-MSE option between two adjacent shared exponents, floor and ceil.

If the exponent values or FP format of the final backend are inconsistent with the simulation, the search conclusion also becomes invalid.

### 2.5 Use Cases

- Scenarios with higher accuracy requirements for MXFP8 weight quantization.
- Model layers where the block maximum distribution is uneven and floor scaling causes clipping of large values.

### 2.6 Usage Restrictions

- Currently registered only in the `mxfp8_per_block_sym` weight quantization scheme and supports weight quantization only.
- For activation quantization, continue to use existing methods such as `minmax`.

---

## 3. Related Terms

- [Linear Quantization](../linear_quant/term_linear_quant.md): the application target; MSE_Round is used as the weight quantization method of linear quantization.
- [Ceil_X](../ceil_x/term_ceil_x.md): a similar algorithm and another shared exponent optimization algorithm for the MXFP format (targeting MXFP4).
- [FouroverSix](../fouroversix/term_fouroversix.md): a similar algorithm and another per-block two-candidate selection quantization algorithm (targeting MXFP4).
- [MinMax](../minmax/term_minmax.md): a comparison algorithm; MSE_Round is the accuracy optimization of MinMax MXFP8 quantization.

---

## 4. Reference Documents

1. [MSE_Round Parameter Configuration Workflow Guide](./usage_mse_round.md)
