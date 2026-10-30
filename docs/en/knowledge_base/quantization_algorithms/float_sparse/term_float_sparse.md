# Float Sparse (ADMM) Algorithm Terminology Entry

> **Entry Category**: [Quantization Algorithms](../README.md#2-quantization-algorithms)<br>
> **English Name**: float_sparse<br>
> **Application Domain**: Model sparsification, high-compression-ratio deployment, inference acceleration<br>

---

## 1. Overview

Float Sparse is a model compression algorithm that finds a sparse weight structure using an optimization method. It zeroes out some floating-point weights while controlling the reconstruction error, reducing effective storage and computation; its core characteristics are error modeling based on calibration information, iterative sparse optimization, and a controllable sparsity ratio, making it suitable for further compressing models in deployment environments that support sparse weight execution.

---

## 2. Term Description

Existing W8A8S quantization methods do support sparse weight quantization, but the sparsity ratio is low, making it hard to achieve a desirable compression effect on the compression unit of Atlas inference products; and to meet accuracy requirements they usually need to roll back some network layers, significantly reducing inference performance. Float Sparse sparsifies the floating-point weights directly and, combined with the hardware compression unit, achieves a higher compression ratio, improving inference performance while maintaining accuracy.

Positioned within the model compression flow, Float Sparse reduces effective parameters and storage/computation overhead by driving some floating-point weights to zero, and its optimization objective differs from that of low-bit quantization. In practice a balance must be struck between "the compression benefit of a higher sparsity ratio" and "the accuracy loss caused by more zeroed weights", so the layer scope and the sparsity proportion are more direct control variables than the bit width.

### 2.1 Core Idea

The core idea of Float Sparse is to "measure, under a fixed sparsity budget, the impact of pruning a weight on the layer output using the input's second-order statistics, and to alternate between the reconstruction error and the hard sparsity constraint via ADMM". It does not equate the absolute magnitude of a weight directly with importance; instead it uses the energy and correlation of the calibration input to judge whether zeroing a given weight can easily be compensated by the other free weights.

To understand this idea, think of sparsification as weight optimization with a structural constraint: the algorithm simultaneously approximates the original model output and keeps projecting the weights onto the set that satisfies the target sparsity. ADMM separates these two objectives and handles them iteratively, so that the weight update and the sparsity constraint can be optimized alternately rather than deleting weights by absolute magnitude in one shot.

Hessian-weighted sparsification is closer to the linear layer output error than pure magnitude pruning: $\|X(W-W_s)^T\|_F^2$ can be expanded into a quadratic form in $W-W_s$ and $X^TX$, so weights along high-energy or highly correlated input directions may be worth keeping even when their values are small.

### 2.2 Working Mechanism

The goal of Float Sparse is not to map weights to fewer integer codes but to zero out some weights while keeping the rest in floating point. Pruning by the smallest $|W|$ alone ignores the correlation of the input activations: a weight with a modest value that corresponds to a high-energy or strongly correlated input direction may still cause a noticeable output error when removed. The algorithm therefore builds a second-order approximation matrix $H=X^TX$ from the calibration activations and uses it to measure the influence of different weight directions on the output reconstruction.

ADMM splits "preserving the output" and "satisfying a fixed sparsity ratio" into two alternating subproblems: one step updates the continuous variable under the quadratic reconstruction objective, another step retains the more important elements and zeroes the rest via a hard projection, and a dual variable gradually brings the two into agreement. This supports closed-form/linear-algebra updates for the second-order error while also handling the non-convex fixed-sparsity constraint. For scenarios that must additionally accommodate low-precision representation, the quantization residual and Hessian scale can be used to identify a small number of positions more worth keeping.

### 2.3 Mathematical Description

Using the convention that samples are rows and input features are columns, let the calibration input be $X\in\mathbb R^{N\times d_{in}}$ and the original weight be $W\in\mathbb R^{d_{out}\times d_{in}}$, and let the weight satisfying the target sparsity constraint be $Z$. A natural output-preserving objective is:

$$
\min_Z\;\|XW^T-XZ^T\|_F^2
\quad\text{s.t.}\quad
Z\in\mathcal S_k,
$$

where $\mathcal S_k$ denotes the sparse set satisfying a given nonzero budget $k$; for structured sparsity it can also represent a specific N:M or block sparsity constraint. Let the input second moment be:

$$
H=X^TX,
$$

Then the output reconstruction error can be written as:

$$
\|XW^T-XZ^T\|_F^2
=
\operatorname{tr}\left((W-Z)H(W-Z)^T\right).
$$

This shows that the cost of zeroing a weight depends not only on its magnitude but also on the energy of the corresponding input direction and the correlation between different input channels.

Since $\mathcal S_k$ is usually a non-convex set, ADMM can be used to separate the continuous reconstruction variable from the sparse variable. Introducing the continuous variable $V$, the sparse variable $Z$, and the scaled dual variable $U$, a typical alternating form can be written as:

$$
V^{t+1}
=
\arg\min_V\;
\operatorname{tr}\left((W-V)H(W-V)^T\right)
+
\frac{\rho}{2}\|V-Z^t+U^t\|_F^2,
$$

$$
Z^{t+1}
=
\Pi_{\mathcal S_k}(V^{t+1}+U^t),
$$

$$
U^{t+1}
=
U^t+V^{t+1}-Z^{t+1}.
$$

where $\Pi_{\mathcal S_k}$ denotes projection onto the target sparse set and $\rho$ controls the strength of agreement between "preserving the original layer output as much as possible" and "making the continuous variable satisfy the hard sparsity constraint". The continuous subproblem redistributes the error using the second-order information in $H$, while the projection subproblem strictly satisfies the sparsity budget; the two alternate until the reconstruction and the sparsity pattern stabilize.

### 2.4 Key Properties

- **ADMM optimization**: uses the alternating direction method of multipliers to solve the constrained sparse optimization problem.
- **Activation-statistics driven**: uses the second-order statistics of the calibration input to measure the influence of different weight perturbations on the layer output.
- **Importance protection**: the sparsity decision considers both the weight value and the input second-order statistics, avoiding the mistaken removal of critical directions by magnitude alone.
- **Iterative sparsification**: progressively optimizes the sparsity pattern over several iterations, balancing the sparsity ratio and accuracy.
- **Second-order importance**: uses the activation second moment to distinguish weights that are "numerically small but important to the output" from those that can safely be pruned.

In terms of error and applicability boundaries, as the sparse ratio increases the feasible set tightens rapidly and the error usually rises nonlinearly; an inadequately estimated calibration Hessian also biases the judgment of "important directions".

### 2.5 Use Cases

- Scenarios that require high-compression-ratio model deployment, especially scenarios involving the compression unit of Atlas inference products.
- Scenarios where the W8A8S sparsity ratio is insufficient and a higher compression ratio is needed.

### 2.6 Usage Restrictions

- The generated weights need to be further compressed on Atlas inference products using the hardware compression unit.
- Atlas inference products do not support the bfloat data type; the `torch_dtype` in the model's `config.json` must be changed to `float16`.
- Primarily aimed at layer-by-layer sparsification of linear layer weights; other parameter types require separately defined sparsity constraints and solution methods.
- The calibration data must contain ≥2048 token ids; the sparsity ratio is recommended to be adjusted gradually around 0.3.

These limitations mean that the sparsity ratio cannot be pursued independently of hardware capability and layer sensitivity. Too high a target sparsity ratio significantly increases the difficulty of recovering accuracy, and forcibly processing modules unsuited to sparsification may cause irreversible quality loss; reusing proven layer-scope and sparsity-ratio combinations should therefore be preferred.

---

## 3. Related Terms

- [Linear Quantization](../linear_quant/term_linear_quant.md): a companion term; after sparsification it can be combined with quantization for further compression.
- [SVDQuant](../svdquant/term_svdquant.md): a comparison algorithm and another model compression solution aimed at high compression ratios.
- [SSZ](../ssz/term_ssz.md): a companion term and another low-bit / low-resource model compression algorithm.

---

## 4. Reference Documents

1. [Float Sparse Parameter Configuration Workflow Guide](./usage_float_sparse.md)
