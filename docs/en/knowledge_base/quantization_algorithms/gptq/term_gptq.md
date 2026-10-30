# GPTQ Weight Quantization Algorithm Terminology Entry

> **Entry Category**: [Quantization Algorithms](../README.md#2-quantization-algorithms)<br>
> **English Name**: gptq<br>
> **Application Domain**: Large language model quantization and compression, high-accuracy weight quantization<br>
> **First Proposed**: Frantar et al., ICLR 2023

---

## 1. Overview

GPTQ is a post-training quantization algorithm that uses second-order information to optimize the low-bit weight quantization error. It constructs a Hessian approximation from the calibration activations and, while quantizing weights step by step, compensates the current rounding error into the not-yet-quantized weights, thereby reducing the layer output reconstruction error; its core characteristics are Hessian-weighted error, stepwise error compensation, and block processing, making it suitable for accuracy-first low-bit weight quantization.

---

## 2. Term Description

Traditional quantization methods (such as [MinMax](../minmax/term_minmax.md)) incur a larger quantization error when the weight distribution is uneven, affecting model accuracy. GPTQ observes that the error of quantizing the current column can be compensated by adjusting the weights of the subsequent unquantized columns, so that the accumulated error is continuously corrected during column-by-column quantization, minimizing the overall quantization error.

### 2.1 Core Idea

The core idea of GPTQ is to "use the input second-order statistics to guide stepwise rounding and compensate the quantization error already produced into the not-yet-quantized free weights". Unlike elementwise nearest rounding, which only cares about the weight's own distance, GPTQ focuses on the quadratic error that a weight perturbation causes in the layer output after passing through the calibration input; it can therefore distinguish the importance of different input directions and exploit the correlation between input channels to reduce the local quantization loss.

The value of second-order compensation can be understood in terms of "correlated input directions": if two input channels are highly correlated, the output deviation produced by quantizing the weight of one channel can be partly canceled by adjusting the weight of the other, not-yet-quantized channel.

### 2.2 Working Mechanism

GPTQ formulates the quantization of a layer's weights as a second-order optimization problem of "preserving that layer's output as much as possible given the calibration input". If the linear layer output is $Y=XW^T$, the squared output error caused by a weight perturbation $\Delta W$ can be written as a quadratic form involving $X^TX$ (or the activation second moment under an equivalent transposed convention). Whether a weight's error matters therefore depends not only on $|\Delta w|$ but also on the energy of the corresponding input direction and its correlation with other directions.

During stepwise quantization, once the current weight has been rounded to a fixed integer code, the error it produces can no longer be removed by that weight itself. GPTQ uses inverse-Hessian information to distribute this error to the not-yet-quantized weights in a second-order optimal way, so that the remaining free variables compensate for the current decision. In practice the quantization and the delayed update are usually performed in column/block batches to reuse matrix operations; adding diagonal damping to the Hessian mitigates ill-conditioned inverse matrices caused by insufficient calibration samples or strong collinearity.

### 2.3 Mathematical Description

Using the convention that samples are rows and input features are columns, let the calibration input be $X\in\mathbb R^{N\times d_{in}}$ and the original weight be $W\in\mathbb R^{d_{out}\times d_{in}}$, with the linear layer output $Y=XW^T$. GPTQ aims to make the layer output of the quantized weight $\hat W$ as close as possible to the floating-point output, subject to the constraints of the target quantization format:

$$
\min_{\hat W}\;\|XW^T-X\hat W^T\|_F^2.
$$

For the row of weights $w$ of a single output channel, let the input second moment be:

$$
H=X^TX.
$$

If the weight error of that row is $\delta w=w-\hat w$, the corresponding local quadratic loss is:

$$
\mathcal L(\delta w)=\delta w^T H\delta w.
$$

After the weight corresponding to the $q$-th input feature is fixed to the quantized value $\hat w_q$, the current error can no longer be removed by that coordinate itself. Let the set of still-unquantized coordinates be $R$; the second-order optimal compensation can be written as an update involving the inverse Hessian:

$$
w_R\leftarrow w_R-\frac{w_q-\hat w_q}{[H^{-1}]_{qq}}[H^{-1}]_{q,R}.
$$

Under this quadratic approximation, the minimum incremental loss caused by fixing the $q$-th coordinate to its quantized value is proportional to:

$$
\Delta\mathcal L_q\propto\frac{(w_q-\hat w_q)^2}{[H^{-1}]_{qq}}.
$$

This shows that a rounding error of the same size does not cost the same in different input directions: if a direction has a strong compensable correlation with others, its error can be shifted more to the remaining free weights; conversely, directions that are hard to compensate should be quantized more carefully.

To improve the numerical stability of the second-moment inversion or decomposition, diagonal damping is usually added:

$$
H_d=H+\lambda I.
$$

Damping can suppress noise amplification along small-eigenvalue directions, but too large a $\lambda$ weakens the second-order-related information, so it is essentially a trade-off between stability and second-order refinement.

### 2.4 Key Properties

- **Layer-wise local reconstruction**: splits a large model into tractable layer-level subproblems and controls each layer's output reconstruction error on the calibration input.
- **Second-order error correction**: uses the Hessian matrix to dynamically adjust the unquantized weights to compensate the quantization error.
- **Block processing**: can perform quantization and error updates in column-block batches to improve matrix-operation efficiency.
- **Grouped scales**: adjacent weights can share quantization parameters by group, trading off between scale metadata and local fitting capability.
- **Error compensation**: the error caused by already-quantized weights is compensated into the not-yet-quantized weights through inverse-Hessian information.

In terms of error and applicability boundaries, too few calibration samples make the second moment unreliable; an ill-conditioned Hessian amplifies inversion noise, so damping is needed; and fixing integer codes step by step is a greedy process that does not guarantee a global optimum. Grouped scales, column order, and block size change the local degrees of freedom and the computational overhead, and thereby affect the final reconstruction.

### 2.5 Use Cases

- Weight quantization scenarios with high accuracy requirements.
- Scenarios where the weight distribution is uneven and conventional MinMax quantization has a relatively large error.

### 2.6 Usage Restrictions

- Currently supports per_channel/per_group symmetric and asymmetric quantization for the int8 scenario; int4 is not yet supported.
- The per_tensor quantization granularity is not yet supported.
- Weights must be 2D tensors.
- Depends on model activations; quantizing MoE models requires the calibration set to cover all experts, so this scenario is not recommended.

---

## 3. Related Terms

- [MinMax](../minmax/term_minmax.md): a comparison algorithm; GPTQ is a high-precision optimization extending MinMax.
- [AutoRound](../autoround/term_autoround.md): a similar algorithm and another high-accuracy weight quantization optimization algorithm.
- [SSZ](../ssz/term_ssz.md): a similar algorithm and another low-bit weight quantization optimization algorithm.
- [Linear Quantization](../linear_quant/term_linear_quant.md): the application target; GPTQ can serve as the high-precision parameter estimation and error compensation method in linear weight quantization.

---

## 4. Reference Documents

1. Frantar E et al. GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. ICLR 2023. <https://arxiv.org/abs/2210.17323>
2. [GPTQ Parameter Configuration Workflow Guide](./usage_gptq.md)
