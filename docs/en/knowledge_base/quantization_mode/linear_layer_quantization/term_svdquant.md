# SVDQuant Quantization

> **Entry Category**: Quantization Data Format ([Linear Layer Quantization](README.md))<br>
> **English Name**: SVDQuant Quantization<br>
> **Application Domain**: Large language model quantization and compression, low-bit quantization accuracy optimization<br>
> **First Proposed**: Li et al., NeurIPS 2024<br>
> **Hosting IR Class**: `SVDResidualWrapper` ([`msmodelslim/ir/svd_residual.py`](../../../../../msmodelslim/ir/svd_residual.py), a `WrapperIR` subclass, used with `linear_quant` to low-bit quantize the residual component)<br>

## 1. Overview

SVDQuant is a [linear layer quantization](README.md) mode for extremely low-bit scenarios: through the combined action of three stages—outlier migration, SVD low-rank decomposition, and residual quantization—it splits the weight into a high-accuracy low-rank bypass and a low-bit residual main path, and at inference the two paths are added to reconstruct the original linear transformation. It solves the severe accuracy loss caused by outliers in the activations and weights at extremely low bit widths such as 4bit, and differs from direct integer quantization as represented by [W4A4 Dynamic Quantization](term_w4a4_dynamic.md); its core characteristics are a high-accuracy low-rank bypass that absorbs the main structure and a low-bit quantized residual main path.

---

## 2. Term Description

SVDQuant does not directly apply low-bit quantization to the linear layer weights and activations; instead it first changes the numerical structure of the weight before quantization: outliers are first migrated into the weight, and then SVD decomposition extracts the main structure into the low-rank bypass, leaving the residual weight with a more uniform distribution that is better suited to low-bit quantization. The low-rank bypass runs at high accuracy in [FP16/BF16](../../dtype_ops/term_fp16_bf16.md), while the residual main path is quantized according to the `linear_quant` configuration; the two outputs are added, mathematically approximating the linear transformation before decomposition.

### 2.1 Mode Specification

| Dimension | Value | Description |
|------|------|------|
| Quantization Target | The weight W and activation A of the residual main path; the low-rank bypass is not quantized | After outlier migration the weight is split into a low-rank component (high accuracy) and a residual component (low-bit quantized) |
| Bit Width | Residual W/A low-bit (such as 4bit); low-rank bypass FP16 | The bit width depends on the residual linear quantization configuration |
| Data Type | Mixed | The residual uses low-bit (such as MXFP4 / INT4) and the low-rank bypass is FP16 |
| Parameter Acquisition | Follows the residual linear quantization configuration (static / dynamic) | Determined by [`linear_quant`](../../quantization_algorithms/linear_quant/term_linear_quant.md) |
| Quantization Granularity | Follows the residual linear quantization configuration (such as per-block) | No fixed granularity; it follows the configuration |
| Symmetry | Follows the residual linear quantization configuration | Follows the configuration |

### 2.2 Quantization Formula

Stage one (outlier migration): per-channel scaling factors transform the weights and activations so that activation outliers migrate into the weights.

Stage two (low-rank decomposition): SVD decomposition is applied to the migrated weight $W$:

$$
W \approx (U \cdot S) \cdot V^\top, \qquad R = W - (U \cdot S) \cdot V^\top
$$

- $W$: the weight matrix after outlier migration
- $U$: the left singular vectors, of shape $[\text{out\_dim}, \text{rank}]$
- $S$: the singular values, of shape $[\text{rank}]$
- $V$: the right singular vectors, of shape $[\text{in\_dim}, \text{rank}]$, with $V^\top$ its transpose
- $R$: the residual weight, entering low-bit quantization

Stage three (dual-path computation at inference):

$$
\text{out} = Q(X \cdot \operatorname{diag}(s)^{-1}) \cdot Q(R) + (X \cdot \operatorname{diag}(s)^{-1} \cdot V) \cdot (U \cdot S)^\top \approx XW + b
$$

- $X$: the input activation
- $s$: the per-channel scaling factor of outlier migration, computed jointly from activation and weight statistics
- $Q(\cdot)$: the low-bit quantization operation (such as 4bit quantization)
- $b$: the linear layer bias

### 2.3 Relationship to Other Modes

- **Relationship to [W4A4 Dynamic Quantization](term_w4a4_dynamic.md) (both are extremely low-bit schemes)**: W4A4 quantizes all weights and activations to integers directly; SVDQuant decomposes the weight first and quantizes only the residual component, using the low-rank bypass to absorb outliers and lowering the 4bit accuracy risk. The two can be understood as the "direct quantization" and "decompose-then-quantize" paths.
- **Relationship to [W8A8 Static Quantization](term_w8a8_static.md) (basic linear quantization scheme)**: the quantization of the residual main path is implemented through [`linear_quant`](../../quantization_algorithms/linear_quant/term_linear_quant.md), so linear quantization modes such as W8A8 can serve as the underlying quantization scheme of the residual main path.
- **Relationship to [Float Sparse Quantization](term_float_sparse.md) (both are structural-level modes)**: float sparse is characterized by keeping a high bit width and skipping sparse zero values, whereas SVDQuant is characterized by absorbing outliers with a low-rank component. Both reduce the accuracy risk of compression through the structural idea of "keeping part of the high-accuracy component + compressing the main body".

### 2.4 Applicable Scenarios and Limitations

#### 2.4.1 Applicable Scenarios

- Layers at extremely low bit widths (such as W4A4) whose activations/weights have significant outliers; especially applicable to diffusion models and models containing many Linear layers.
- Compression scenarios that need to preserve accuracy as much as possible: the low-rank bypass preserves the main structure at high accuracy while the residual component is low-bit quantized.

#### 2.4.2 Usage Limitations

- The target layers must be standard `torch.nn.Linear`, and their module names must be obtainable via `model.named_modules()`.
- The `include` / `exclude` of the three stages (outlier migration, SVD decomposition, residual quantization) should be kept consistent, ensuring that the same set of layers goes through the three stages in order.
- The decomposition rank `rank` is limited by the operator implementation; a value not exceeding 128 is recommended.

---

## 3. Related Workflows

- [One-Click Quantization Usage Guide](../../../user_guide/usage_one_click_quantization.md): configure SVDQuant's `iter_smooth` -> `svd_res` -> `linear_quant` three-stage pipeline via `--quant_type` or a YAML configuration.
- [Quantization Accuracy Tuning Guide](../../../user_guide/process_quantization_precision_tuning.md): when the accuracy falls short, the decomposition rank `rank` and the outlier migration strength `alpha` can be adjusted.

---

## 4. Related Terms

- [Quantization Modes](../README.md): the parent concept; this entry belongs to the [Linear Layer Quantization](README.md) category and is one structural-level mode under it.
- [Linear Layer Quantization](README.md): the parent concept, the category this entry belongs to; SVDQuant acts on the Linear layers within it.
- [W4A4 Dynamic Quantization](term_w4a4_dynamic.md): a comparison mode, also an extremely low-bit scheme; SVDQuant lowers the 4bit accuracy risk through decomposition.
- [W8A8 Static Quantization](term_w8a8_static.md): a companion mode that can serve as the underlying quantization scheme of the residual main path.
- [Float Sparse Quantization](term_float_sparse.md): a comparison mode, also a structural-level mode of "keeping a high-accuracy component + compressing the main body".
- [FP16/BF16](../../dtype_ops/term_fp16_bf16.md): a companion term, the high-accuracy data type used by the low-rank bypass.
- [SVDQuant Low-Rank Residual Quantization Algorithm](../../quantization_algorithms/svdquant/term_svdquant.md): a companion term, the algorithm-side explanation of the principle and implementation.

---

## 5. Reference Documents

1. Li M et al. SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models. NeurIPS 2024. <https://arxiv.org/abs/2411.05007>
2. *[SVDQuant Parameter Configuration Workflow Guide](../../quantization_algorithms/svdquant/usage_svdquant.md)*
3. *[Quantization Modes](../README.md)*
