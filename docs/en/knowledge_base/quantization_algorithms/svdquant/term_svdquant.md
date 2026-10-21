# SVDQuant Low-Rank Residual Quantization Algorithm Terminology Entry

> **Entry Category**: [Quantization Algorithms](../README.md#2-quantization-algorithms)<br>
> **English Name**: svdquant<br>
> **Application Domain**: Diffusion model quantization, low-bit quantization accuracy optimization<br>
> **First Proposed**: Li et al., NeurIPS 2024

---

## 1. Overview

SVDQuant is a low-rank residual method for extremely low-bit quantization. It first migrates hard-to-quantize outlier information through smoothing, then uses singular value decomposition to extract the salient structure into a high-precision low-rank branch and quantizes the remaining main path at low bit width; its core characteristics are outlier migration, a low-rank high-precision bypass, and coordination with the low-bit residual, used to reduce the quantization error in configurations such as W4A4.

---

## 2. Term Description

Significant outliers exist in the activations of diffusion models; the extreme values of a few channels compress the quantization accuracy of the remaining channels, so direct quantization causes a severe accuracy drop. SVDQuant observes that after the outliers are first migrated to the weights, the outliers in the weights exhibit a low-rank structure, which is exactly suited to extraction by SVD low-rank decomposition. This "migrate → decompose" combination is the core innovation of SVDQuant: the outliers are first migrated to the weights and then absorbed by the low-rank branch, solving the quantization difficulty of the activations and the weights at once.

### 2.1 Core Idea

The core idea of SVDQuant is "three-stage cooperative quantization": first migrate the activation outliers to the weights through outlier suppression (such as Iterative Smooth); then perform SVD low-rank decomposition on the migrated weights, extracting the main structure into a low-rank component (kept at high precision) while the residual part is more amenable to quantization; finally quantize the residual weights at low bit width, with the low-rank branch running at high precision.

The low-rank bypass is effective because it isolates the hardest-to-compress high-energy directions from the low-bit residual.

### 2.2 Working Mechanism

SVDQuant first transfers the hard-to-quantize outlier directions from the activations to the weights through smoothing/scaling, and then, instead of compressing the whole new weight to low bit width directly, performs a decomposition $W=L+R$. The first $r$ singular directions of the SVD form the low-rank term $L=U_r\Sigma_rV_r^T$; by the Eckart–Young property, in the Frobenius norm sense this truncated SVD is the best low-rank approximation for the given rank, so it captures the highest-energy structural directions of the weight first.

In the remaining residual $R=W-L$, both the magnitude and the large structural components are usually weakened, making it better suited as the low-bit main branch; the low-rank term is computed through a high-precision bypass, protecting the hardest-to-quantize directions. At inference time the output is the sum of the low-bit residual path and the high-precision low-rank path. A larger rank lets the bypass absorb more quantization-sensitive energy but also increases computation/memory overhead; a smaller rank is lighter but puts more pressure on the residual quantizer.

### 2.3 Mathematical Description

Mathematical equivalent transform of stage one (outlier migration):

$$
Y = XW + b = (X \cdot \operatorname{diag}(s)^{-1}) \cdot (\operatorname{diag}(s) \cdot W) + b
$$

where the scaling factor $s$ is computed jointly from the activation and weight statistics:

$$
s = \frac{A_{\text{scale}}^{\alpha}}{W_{\text{scale}}^{1-\alpha}}, \quad s \geq s_{\min}
$$

- $X$: activation matrix
- $W$: weight matrix
- $s$: per-channel scale factor
- $A_{\text{scale}}$: the per-channel absolute maximum of the activation values
- $W_{\text{scale}}$: the per-column maximum of the weights
- $\alpha$: the balance parameter, controlling the migration strength

Stage two (low-rank decomposition): perform SVD decomposition on the migrated weight $W$:

$$
W \approx (U \cdot S) \cdot V^\top, \quad R = W - (U \cdot S) \cdot V^\top
$$

- $U$: the left singular vectors, with shape $[\text{out\_dim}, \text{rank}]$
- $S$: the singular values, with shape $[\text{rank}]$
- $V$: the right singular vectors, with shape $[\text{in\_dim}, \text{rank}]$
- $R$: the residual weight

Stage three (residual quantization): dual-path computation at inference time:

$$
\text{out} = Q(X \cdot \operatorname{diag}(s)^{-1}) \cdot Q(R) + (X \cdot \operatorname{diag}(s)^{-1} \cdot V) \cdot (U \cdot S)^\top \approx XW + b
$$

- $Q(\cdot)$: the quantization operation

If the singular values are sorted as $\sigma_1\ge\cdots\ge\sigma_p$, the truncated rank-$r$ SVD satisfies the Eckart–Young theorem:

$$
L_r=U_r\Sigma_rV_r^T=\arg\min_{\operatorname{rank}(L)\le r}\|W-L\|_F,
$$

and the residual energy is:

$$
\|W-L_r\|_F^2=\sum_{i>r}\sigma_i^2.
$$

Increasing the rank therefore monotonically reduces the Frobenius energy of the unquantized residual, but the final low-bit error also depends on the residual distribution rather than on the total energy alone. At inference time $XW^T=X(L_r+R)^T=XL_r^T+XR^T$, where the low-bit approximation is used only for $R$.

### 2.4 Key Properties

- **Three-stage cooperation**: outlier migration, low-rank decomposition, and residual quantization work together, keeping the output approximately equivalent in mathematical terms.
- **Low-rank bypass**: the low-rank component runs at high precision (such as FP16), while the residual main path is quantized at low bit width.
- **Mathematical equivalence**: the sum of the dual-path outputs is equivalent to a linear transform using the original weight $W$.
- **Quantizing the residual rather than discarding it**: the low-rank component is kept at high precision and the remaining matrix still participates in the computation, only in a low-bit representation.

In terms of error and applicability boundaries, the Frobenius optimality of the SVD is not equivalent to task-loss optimality, because it does not directly use the activation Hessian; calibration/end-to-end evaluation therefore remains important.

### 2.5 Use Cases

- Extremely low-bit quantization scenarios such as W4A4, especially well suited to diffusion models.
- Scenarios where the model activations contain significant outliers and those outliers exhibit a low-rank structure in the weights.

### 2.6 Usage Restrictions

- Primarily aimed at two-dimensional linear transform weights, so that the low-rank decomposition can act directly on the weight matrix.
- Outlier migration, low-rank decomposition, and residual quantization should act on the same set of target linear layers, to avoid a layer undergoing only part of the stages and thereby breaking the decomposition assumption.
- `rank` is limited by the operator implementation; not exceeding 128 is recommended.

---

## 3. Related Terms

- [Iterative Smooth](../iterative_smooth/term_iterative_smooth.md): a prerequisite term; SVDQuant uses it to complete outlier migration.
- [Linear Quantization](../linear_quant/term_linear_quant.md): a companion term; SVDQuant uses it to complete residual quantization.
- [SmoothQuant](../smooth_quant/term_smooth_quant.md): a similar algorithm in the same outlier suppression family.
- [Float Sparse](../float_sparse/term_float_sparse.md): a comparison algorithm and another model compression solution aimed at high compression ratios.

---

## 4. Reference Documents

1. Li M et al. SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models. NeurIPS 2024. <https://arxiv.org/abs/2411.05007>
2. [SVDQuant Parameter Configuration Workflow Guide](./usage_svdquant.md)
