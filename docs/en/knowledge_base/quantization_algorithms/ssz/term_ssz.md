# SSZ Weight Quantization Algorithm Terminology Entry

> **Entry Category**: [Quantization Algorithms](../README.md#2-quantization-algorithms)<br>
> **English Name**: ssz<br>
> **Application Domain**: Large language model quantization and compression, low-bit weight quantization<br>

---

## 1. Overview

SSZ (Smooth Scale Zero) is a scale and zero-point optimization algorithm for low-bit weight quantization. It alternates between fixing the integer codes and updating the quantization parameters, seeking more suitable scale and zero-point values by minimizing the weight reconstruction error in order to improve on simple MinMax for unevenly distributed weights; its core characteristics are iterative optimization, reconstruction-error driving, and suitability for low-bit weight scenarios such as INT4.

---

## 2. Term Description

Traditional quantization methods (such as [MinMax](../minmax/term_minmax.md)) incur a larger quantization error when the weight distribution is uneven, affecting model accuracy. SSZ observes that by iteratively searching the scale and offset, better quantization parameters can be found for unevenly distributed weights, significantly reducing the quantization error.

### 2.1 Core Idea

The core idea of SSZ is to treat solving for the quantization parameters as a problem of "alternately optimizing the discrete integer codes and the continuous scale parameters". MinMax only provides an initial scale/zero-point based on the coverage range, but covering all extremes is not equivalent to minimizing the weight reconstruction error; SSZ first uses the current quantization parameters to determine the integer codes, then, with those integer codes temporarily fixed, solves a least-squares problem for a better scale (and, for asymmetric quantization, a better offset/zero-point), and then re-quantizes and continues iterating.

Because recomputing the scale changes which integer code each weight lands on, the overall objective is piecewise and contains a discrete variable, so a global optimum cannot be obtained in closed form in one step. Alternating optimization splits the hard problem into two easier subproblems and improves progressively by "accepting new parameters only when the error genuinely decreases", so it targets the squared reconstruction error of low-bit weights more directly than a single MinMax pass.

Alternating least squares is effective because MinMax's scale objective is only "coverage", whereas SSZ's objective is directly "fitting the weights with the existing finite integer codes".

### 2.2 Working Mechanism

Taking symmetric quantization as an example, once the current integer codes $q_i$ are given the dequantized weights are $\hat w_i=sq_i$. With $q_i$ fixed, minimizing $\sum_i(w_i-sq_i)^2$ over $s$ is a standard least-squares problem with the closed-form solution $s=\frac{\sum_iw_iq_i}{\sum_iq_i^2}$ (when the denominator is nonzero). Asymmetric quantization must also handle the zero point/offset, and after fixing the integer codes the corresponding affine fitting parameters can be solved.

After $s,z$ are updated, the weights are quantized again, the set of integer codes may change, and the next round begins. SSZ compares the old and new QDQ errors per channel or per group, accepts only parameters that genuinely improve, and uses absolute/relative error changes to decide convergence. This process can be seen as an alternating descent on $\min_{s,z,q}\|W-s(q-z)\|_2^2$: $q$ is a discrete variable constrained by the integer range, and $s,z$ are continuous/low-dimensional parameters.

### 2.3 Mathematical Description

For symmetric quantization the offset is fixed at 0 and only the scale is optimized; for asymmetric quantization both the scale and the offset are optimized. The objective is to minimize the quantization error:

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (w_i - \hat{w}_i)^2
$$

- $\text{MSE}$: the mean squared error of the weights before and after quantization
- $w_i$: the original weights
- $\hat{w}_i$: the weights after quantization-dequantization
- $n$: the number of weight elements

Convergence condition:

$$
\frac{|\text{best\_mse} - \text{current\_mse}|}{\text{best\_mse}} < \text{threshold} \quad \text{or} \quad |\text{best\_mse} - \text{current\_mse}| < \text{threshold}
$$

- $\text{best\_mse}$: the best mean squared error seen so far
- $\text{current\_mse}$: the current mean squared error
- $\text{threshold}$: the convergence threshold for judging whether the error improvement across consecutive iterations is already small enough.

It exits early when all channels satisfy the convergence condition.

For the asymmetric case, dequantization can be written as $\hat w_i=sq_i+b$ (where $b=-sz$). Once the integer codes $q_i$ are fixed, $s,b$ are simply ordinary univariate linear regression:

$$
(s^*,b^*)=\arg\min_{s,b}\sum_i(w_i-sq_i-b)^2.
$$

Its normal equations are:

$$
\begin{bmatrix}\sum q_i^2&\sum q_i\\\sum q_i&N\end{bmatrix}\begin{bmatrix}s\\b\end{bmatrix}=\begin{bmatrix}\sum q_iw_i\\\sum w_i\end{bmatrix}.
$$

After the continuous $b$ is obtained it must still satisfy the actual zero-point/integer format constraints, so it may need to be projected or rounded again. Symmetric quantization sets $b=0$, reducing to $s=\sum w_iq_i/\sum q_i^2$.

### 2.4 Key Properties

- **Iterative optimization**: progressively optimizes the quantization parameters over multiple iterations.
- **Least-squares solving**: uses least squares to compute the currently optimal scale and offset.
- **Greedy update**: keeps only parameters that improve the quantization error.
- **Convergence judgment**: decides convergence from relative and absolute error changes.
- **Low-bit friendly**: designed specifically for low-bit weight quantization such as INT4.
- **Alternating optimization**: the integer codes and the continuous scale/zero-point depend on each other, and the reconstruction error is progressively reduced by repeated recomputation.

Because the integer codes may jump each time the scale is recomputed, the objective is not convex; the algorithm can only guarantee that worse candidates are not accepted during iteration, not a global optimum. At low bit widths there are few code points and the piecewise boundaries are more pronounced, so SSZ usually has more room for optimization; but if the weights are close to uniform and MinMax is already good, the extra iteration gain may be limited.

### 2.5 Use Cases

- Linear layer quantization scenarios with high accuracy requirements and uneven weight distributions.
- Low-bit weight quantization scenarios such as INT4.

### 2.6 Usage Restrictions

- Currently supports per_channel symmetric quantization for the int8 and int4 scenarios.
- per_channel asymmetric quantization for the int4 scenario is not yet supported (planned).
- The per_tensor and per_group quantization granularities are not yet supported.
- The weights must be a 2D tensor.

---

## 3. Related Terms

- [Linear Quantization](../linear_quant/term_linear_quant.md): the application target; SSZ is used as the weight quantization method of linear quantization.
- [MinMax](../minmax/term_minmax.md): a prerequisite term; SSZ uses the MinMax observer to initialize the quantization parameters.
- [GPTQ](../gptq/term_gptq.md): a similar algorithm and another high-accuracy weight quantization optimization algorithm.
- [Flex AWQ SSZ](../flex_awq_ssz/term_flex_awq_ssz.md): a companion term; the weight configuration of Flex AWQ SSZ usually uses the SSZ method.

---

## 4. Reference Documents

1. [SSZ Parameter Configuration Workflow Guide](./usage_ssz.md)
