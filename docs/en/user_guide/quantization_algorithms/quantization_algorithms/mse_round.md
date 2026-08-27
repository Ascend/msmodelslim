# MSE_Round: Per-Block MSE Optimal MXFP8 Weight Quantization Algorithm

<!-- md-trans-meta sourceCommit=aa8b963711df0fed6becc2b09f134d3bf88d08f4 translatedAt=2026-08-19T08:16:23.789Z pushedAt=2026-08-19T08:17:52.366Z -->

## 1. Introduction

- **Problem**: In MXFP8 per-block quantization, the traditional minmax method fixes the shared exponent as $s = \lfloor \log_2(\max(|x|)) \rfloor - e_{\text{max}}$. After scaling, the magnitude of the maximum value within the block is $\max(|x|) / 2^s \in [2^{e_{\text{max}}}, 2^{e_{\text{max}}+1})$, that is, $\,[8, 16)$ ($e_{\text{max}} = 8$). When the maximum value within the block approaches the upper bound, the scaled result may exceed the representable upper limit of MXFP8 ($448$), causing clipping of large values and introducing significant quantization error.

- **Objective**: For each block, compare the actual quantization-dequantization MSE between the **ceil** and **floor** two-level shared exponents, and adaptively select the scaling scheme with the smaller error, thereby improving the accuracy of MXFP8 weight quantization.

## 2. Prerequisites

Install the msModelSlim tool. For details, see [msModelSlim Tool Installation Guide](../../../install_guide/install_guide.md).

## 3. Principle and Implementation

### 3.1 Principle

The MSE_Round algorithm addresses the clipping problem that may arise from floor scaling in traditional MXFP8 per-block quantization. Within each block, it performs an MSE comparison between the **ceil** and **floor** two-level shared exponents and selects the best one.

**Traditional floor scaling:**

The traditional minmax method computes the shared exponent as follows:

$$s_{\text{floor}} = \lfloor \log_2(\max(|x|)) \rfloor - e_{\text{max}}$$

The magnitude of the maximum value within the block after scaling is:

$$\frac{\max(|x|)}{2^{s_{\text{floor}}}} = \frac{\max(|x|) \cdot 2^{e_{\text{max}}}}{2^{\lfloor \log_2(\max(|x|)) \rfloor}} \in [2^{e_{\text{max}}}, 2^{e_{\text{max}}+1}) = [8, 16)$$

Since $\max(|x|) / 2^{\lfloor \log_2(\max(|x|)) \rfloor} \in [1, 2)$, the overall range falls within $\,[8, 16)$. The representable upper limit of MXFP8 (E4M3 format) is $2^{e_{\text{max}}} \times 1.75 = 448$. When the scaled maximum value approaches or exceeds this limit, clipping of large values occurs, resulting in significant error.

**MSE_Round:**

MSE_Round computes ceil and floor two-level candidate shared exponents for each block simultaneously:

$$s_{\text{ceil}} = \lceil \log_2(\max(|x|)) \rceil - e_{\text{max}}$$

$$s_{\text{floor}} = \lfloor \log_2(\max(|x|)) \rfloor - e_{\text{max}}$$

Complete quantization-dequantization using the two-level parameters respectively, and compute the MSE within the block:

$$\text{MSE}_{\text{ceil}} = \frac{1}{N}\sum_{i=1}^{N}(x_i - \hat{x}_i(s_{\text{ceil}}))^2, \quad \text{MSE}_{\text{floor}} = \frac{1}{N}\sum_{i=1}^{N}(x_i - \hat{x}_i(s_{\text{floor}}))^2$$

Finally, per-block selects the shared exponent with the smaller MSE:

$$s^* = \begin{cases} s_{\text{ceil}} & \text{if } \text{MSE}_{\text{ceil}} < \text{MSE}_{\text{floor}} \\ s_{\text{floor}} & \text{otherwise} \end{cases}$$

After scaling, the ceil level has a value range of $(2^{e_{\text{max}}-1}, 2^{e_{\text{max}}}] = (4, 8]$, which compresses the scaling magnitude compared with the floor level's $\,[8, 16)$, effectively avoiding clipping of large values when the maximum value within the block approaches the upper bound. The floor level, on the other hand, may achieve a better overall MSE when the block distribution is relatively uniform. MSE_Round combines the advantages of both scaling strategies through per-block adaptive selection.

**Core idea:**

1. **Block-based processing**: Divide the weight matrix into independent data blocks of size 32 along the specified axis.

2. **Dual-candidate computation**: For each block, compute the ceil and floor two-level shared exponents based on $\max(|x|)$, where $e_{\text{max}} = 2^{e_{\text{bits}}-1} = 8$ (MXFP8 E4M3 format).

3. **MSE-based selection of the best candidate**: For each candidate level, perform the complete quantization-dequantization process, compute the MSE within the block, and select the shared exponent with the smaller MSE as the final quantization parameter.

### 3.2 Implementation

The algorithm is implemented in [`msmodelslim/core/quantizer/impl/mse_round.py`](../../../../../msmodelslim/core/quantizer/impl/mse_round.py):

- Implementation class: `MXWeightPerBlockMseRound`

- Registered quantization type: `mxfp8_per_block_sym`

**Core code logic:**

```python
# Compute the per-block max
minmax_block_observer.update(weight_value, sync=False, shared_exp_axes=shared_exp_axes)
_, max_val = minmax_block_observer.get_min_max()

# Compute the ceil and floor two-level candidate shared exponents 
log2v = log2(max_val + FP32_MIN_NORMAL * (max_val == 0))
shared_exp_up = ceil(log2v) - emax      # ceil level
shared_exp_down = floor(log2v) - emax   # floor level

# Quantize and dequantize separately, and compute the MSE within the block
dequant_up = dequantize(quantize(weight, q_param_up), q_param_up)
dequant_down = dequantize(quantize(weight, q_param_down), q_param_down)
mse_up = (weight - dequant_up).pow(2).mean(dim=-1, keepdim=True)
mse_down = (weight - dequant_down).pow(2).mean(dim=-1, keepdim=True)

# Select the shared exponent with the smaller MSE
shared_exp = select_by_mse(mse_up, mse_down, shared_exp_up, shared_exp_down)
```

**MSE selection strategy:**

```python
# Select ceil when its MSE is valid and smaller; otherwise, select floor
select_up = valid_up & ((mse_up < mse_down) | ~valid_down)
shared_exp = where(select_up, shared_exp_up, shared_exp_down)
```

When the shared exponent of a candidate exceeds the E8M0 representable range (marked as NaN), it automatically falls back to the other valid candidate.

## 4. Applicable Requirements

- **Precision improvement**: Applicable to scenarios that require higher precision in MXFP8 weight quantization, especially model layers where the maximum value within a block is unevenly distributed and floor scaling causes clipping of large values.

- **Computing cost**: Each block requires two quantization-dequantization evaluations, with a computational load approximately twice that of standard minmax MXFP8 quantization. However, no additional hyperparameter search is required, so the overhead is manageable.

## 5. Feature Description

### 5.1 YAML Configuration Example

```yaml
spec:
  process:
    - type: "linear_quant"
      qconfig:
        act:
          scope: "per_block"
          dtype: "mxfp8"
          symmetric: true
          method: "minmax"
        weight:
          scope: "per_block"
          dtype: "mxfp8"
          symmetric: true
          method: "mse_round"
```

In the W8A8 MXFP mixed-precision scenario, you can set the weight to `mse_round` and keep the activation as `minmax`, as shown in the following example:

```yaml
spec:
  process:
    - type: "linear_quant"
      qconfig:
        act:
          scope: "per_block"
          dtype: "mxfp8"
          symmetric: true
          method: "minmax"
        weight:
          scope: "per_block"
          dtype: "mxfp8"
          symmetric: true
          method: "mse_round"
      include: ["*"]
```

## 6. Technical Advantages

### 6.1 Comparison with Traditional MXFP8 Quantization

| Feature | Traditional minmax MXFP8 Quantization | MSE_Round Quantization |
|------|----------------------|----------------|
| Scaling factor | Fixed $2^{\lfloor \log_2(\max) \rfloor - e_{\text{max}}}$ | Per-block MSE-based selection of the best between the ceil and floor two-level options |
| Value range after scaling | $[8, 16)$ | Ceil level $(4, 8]$ or floor level $[8, 16)$, adaptively selected |
| Clipping of large values | May clip when approaching the upper bound | Preferentially selects the ceil level that avoids clipping |
| Adaptive strategy | Not supported | Per-block MSE optimal selection |
| Additional hyperparameters | None | None (zero configuration, ready to use out of the box) |
| Computational complexity | O(N) | O(2N) |

### 6.2 Comparison with Ceil_X (MXFP4)

| Feature | Ceil_X (MXFP4) | MSE_Round (MXFP8) |
|------|----------------|-------------------|
| Target format | MXFP4 | MXFP8 |
| Improvement method | ceil + configurable divisor $c$ | per-block ceil/floor MSE to select the best |
| Search granularity | Optional global MSE search for $c$ | per-block two-level comparison |
| Typical scenario | W4A4 MXFP4 weight quantization | W8A8 MXFP8 weight quantization |

## 7. FAQs

### 7.1 Origin of the Name MSE_Round

MSE_Round derives from the core operation of the algorithm: by comparing **MSE** (mean squared error), it makes the optimal **Round** (selection) decision between the two-level shared exponents **ceil** and **floor**. Compared with the minmax method that always uses floor, MSE_Round adaptively selects the scaling scheme with smaller quantization error within each block.

### 7.2 Why Does MXFP8 Also Need an Improved Scaling Strategy?

Although the representable range of MXFP8 (upper limit 448) is much larger than that of MXFP4 (upper limit 6.0), the traditional floor scaling maps the maximum value within a block to the $[8, 16)$ magnitude. When the block maximum approaches the $2^k$ upper bound, the scaled result may still exceed the representable range of MXFP8 and cause clipping. MSE_Round compares the actual quantization errors of the ceil and floor two-level candidates per block, achieving a better balance between avoiding clipping and preserving quantization accuracy.

### 7.3 What Is the Difference Between MSE_Round and Ceil_X?

The two methods optimize the shared exponent computation strategy for the MXFP8 and MXFP4 formats, respectively. Ceil_X introduces a configurable divisor $c$ combined with the ceil operation to compress the scaling range, and supports a global MSE search for the optimal $c$. MSE_Round, in contrast, performs an MSE comparison between the ceil and floor levels within each block, requiring no additional hyperparameters and working out of the box. In W8A8 MXFP quantization practices for models such as Wan2.2, MSE_Round achieves good accuracy as a weight quantization method.

### 7.4 Is Activation Quantization Supported?

Currently, MSE_Round is registered only for the `mxfp8_per_block_sym` weight quantization scheme. For activation quantization, continue to use existing methods such as `minmax`.
