# HQQ Weight Quantization Algorithm Terminology Entry

> **Entry Category**: [Quantization Algorithms](../README.md#2-quantization-algorithms)<br>
> **English Name**: hqq<br>
> **Application Domain**: Training-free and calibration-free weight quantization and compression for large language models<br>

---

## 1. Overview

HQQ is a training-free, calibration-free weight quantization algorithm. After [MinMax](../minmax/term_minmax.md) computes the initial scale/offset, it iteratively optimizes the quantization offset through half-quadratic splitting to minimize the quantization error. Its core characteristics are completing weight quantization without any calibration data, improving robustness to outliers through the Lp norm, and greedy updates at the specified granularity that ensure convergence quality; it is often used in high-precision low-bit weight quantization scenarios.

---

## 2. Term Description

Conventional calibration-free quantization (such as [MinMax](../minmax/term_minmax.md)) determines the quantization range directly from the weight extremes, which results in a relatively large quantization error when the weight distribution is uneven. HQQ observes that the quantization error can be significantly reduced by iteratively adjusting the offset: with the scale fixed, the optimal offset is obtained by applying an Lp-norm shrinkage to the quantization residual and then taking the mean, thereby approaching the optimal quantization parameters at the specified granularity without any data.

Positioned within the quantization flow, this algorithm addresses the problem of "how to map continuous floating-point values onto a restricted numerical set while preserving the model output as much as possible". Compared with basic methods that compute the scale directly from the extremes, HQQ takes minimizing the quantization error as its optimization objective and solves for the optimal offset iteratively, making it better suited to quantization scenarios with clear accuracy requirements.

### 2.1 Core Idea

The core idea of HQQ is to "iteratively optimize the offset through half-quadratic splitting": it decomposes the weight quantization error minimization problem into two subproblems that can be solved alternately, and with the scale fixed, iteratively alternates between "residual computation" and "offset update" at the specified granularity until convergence. Because the whole process depends only on the weights themselves, it is data-free.

The advantage of HQQ is that the quantization process is determined entirely by the weights and requires no activation or calibration data; when calibration data is hard to obtain and quantization runtime is not critical, it is a low-cost accuracy improvement over MinMax.

### 2.2 Working Mechanism

HQQ first uses the MinMax observer to gather the weight extremes and compute the initial scale and offset. It then enters half-quadratic splitting iteration: with the scale fixed, each iteration first computes the current pseudo-quantization residual $u=W-W_r$, applies a generalized soft threshold to the residual to obtain the shrunk residual, and updates the optimal offset at the specified granularity accordingly; for each granularity it compares the quantization error of the new and old parameters independently, keeps only the better one, and decides whether to stop early based on dual thresholds for relative decrease and absolute change.

Because the scale is fixed throughout, HQQ compresses the search dimension to the offset alone; combined with greedy updates at the specified granularity, this approaches better quantization parameters at a low computational cost under calibration-free conditions. Quantization and dequantization follow the framework convention $q=\operatorname{round}(w/\text{scale}+\text{offset})$ and $w=(q-\text{offset})\cdot\text{scale}$.

### 2.3 Mathematical Description

HQQ aims to minimize the Lp-norm error introduced by weight quantization:

$$
\min_{q,\, \text{offset}} \| W - \operatorname{dequant}(q) \|_p
$$

where $\operatorname{dequant}(q) = (q - \text{offset}) \cdot \text{scale}$. The iteration proceeds as follows:

- Compute the quantization residual $u = W - W_r$ (where $W_r$ is the current pseudo-quantized weight).
- Apply a generalized soft threshold to the residual:

$$
W_e = \operatorname{sign}(u) \cdot \operatorname{relu}\left(|u| - \frac{|u|^{p-1}}{\beta}\right)
$$

- Update the current optimal offset:

$$
\text{offset}_{\text{new}} = \mathbb{E}\left[q - \frac{W - W_e}{\text{scale}}\right]
$$

- At the specified granularity, greedily keep the better parameters and decide convergence from the relative/absolute decrease thresholds, exiting early.

- $W$: the original weight matrix
- $q$: the quantized weight
- $W_r$: the weight obtained by dequantizing the quantized weight
- $u$: the quantization residual, $u = W - W_r$
- $W_e$: the residual after shrinkage
- $\text{scale}$: the scaling factor, which is fixed and not updated in HQQ
- $\text{offset}$: the asymmetric quantization zero point
- $p$: the Lp-norm order, with value range (0,1]
- $\beta$: the regularization coefficient, controlling the penalty strength on the quantization error

Convergence condition (early exit when it holds for all granularities):

$$
\frac{|\text{best\_pnorm} - \text{current\_pnorm}|}{\text{best\_pnorm} \cdot \text{clamp}} < \text{min\_scale} \quad \text{or} \quad |\text{best\_pnorm} - \text{current\_pnorm}| < \text{converge\_threshold}
$$

### 2.4 Key Properties

- **Calibration-free**: the quantization process depends only on the weights and requires no calibration dataset.
- **Fixed scale, optimized offset**: HQQ does not update the scale and only iteratively optimizes the offset, reducing the search dimension.
- **Quantizes weights only, not activations**: HQQ solves for the offset iteratively, which takes relatively long, so it is not suitable for activation quantization.
- **Symmetric and asymmetric**: because HQQ reduces the quantization error by optimizing the offset, the offset is a variable, so only asymmetric quantization can be used.
- **Insensitive to outliers**: using the Lp norm of the residual as the optimization objective reduces the influence of outliers.

Because HQQ only optimizes the offset without changing the scale, its optimization space is relatively limited and it may not achieve the same gain as SSZ/GPTQ; but when the goal is "no activation quantization needed while higher weight quantization accuracy is required", HQQ is a relatively stable choice.

### 2.5 Use Cases

- Weight quantization scenarios with high accuracy requirements.
- Scenarios where the weight distribution is uneven and conventional MinMax quantization has a relatively large error.

More specifically, whether it applies depends mainly on the target bit width, the model structure, and the deployment backend. If the target deployment chain already clearly supports the quantization format corresponding to this algorithm and representative calibration data is unnecessary or difficult to prepare, it is usually reasonable to start from this algorithm's recommended configuration to establish a baseline, and then decide from the accuracy results whether to bring in a more complex, data-requiring optimization.

### 2.6 Usage Restrictions

- Currently supports int8 per_channel asymmetric quantization; int4 quantization is not yet supported.
- The per_tensor / per_group quantization granularities are not yet supported.
- Weights must be 2D tensors.

These limitations should be confirmed before tuning parameters rather than investigated after an accuracy anomaly. This is especially true for hard constraints such as the data type, tensor dimensions, group size, and backend operator support: once they are unmet, further adjustment of algorithm parameters usually cannot solve the problem, and one should first return to a supported configuration combination.

---

## 3. Related Terms

The relationship between this entry and other algorithms can be understood from three directions: similar methods, pre/post-processing relationships, and application targets. The related items below serve both to compare different technical routes horizontally and to help locate this algorithm within a complete quantization solution.

- [MinMax](../minmax/term_minmax.md): a prerequisite term; HQQ uses the MinMax observer to initialize scale/offset.
- [SSZ](../ssz/term_ssz.md): a comparison algorithm and another weight quantization method that iteratively optimizes quantization parameters, but SSZ depends on calibration data.
- [GPTQ](../gptq/term_gptq.md): a comparison algorithm; GPTQ depends on calibration data and second-order information, whereas HQQ is calibration-free.
- [Linear Quantization](../linear_quant/term_linear_quant.md): the application target; HQQ is used as a high-precision quantization method of linear weight quantization.

---

## 4. Reference Documents

The reference documents list the original paper or authoritative source first, supplemented by the corresponding usage guide in this repository. To understand parameter choices further, read the usage guide first and then return to the original paper to check the algorithm's assumptions and derivations.

1. Badri H, Shaji A. Half-Quadratic Quantization of Large Machine Learning Models. <https://dropbox.github.io/hqq_blog/>
2. [HQQ Parameter Configuration Workflow Guide](./usage_hqq.md)
