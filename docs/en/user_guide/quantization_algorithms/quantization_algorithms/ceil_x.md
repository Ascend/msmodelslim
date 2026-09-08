# Ceil_X: Adaptive Divisor MXFP4 Weight Quantization Algorithm

<!-- md-trans-meta sourceCommit=3909078a95b1f3d532dd72a3a08143f4d01c1435 translatedAt=2026-08-19T08:16:20.887Z pushedAt=2026-08-19T08:17:52.363Z -->

## Introduction

- **Problem**: In MXFP4 per-block quantization, the traditional method uses $s = \lfloor \log_2(\max(|x|)) \rfloor - e_{\text{max}}$ to compute the shared exponent, resulting in a scaled value range of $\max(|x|) / 2^s \in [4, 8)$, while the representable upper limit of MXFP4 is 6.0. This causes large values in the interval $\,[6, 8)$ to be truncated, introducing significant quantization error.

- **Objective**: By introducing a configurable divisor $c$ combined with the `ceil` operation, compress the scaled value range into the representable interval of MXFP4, thereby reducing truncation of large values and improving quantization accuracy.

## Prerequisites

Install the msModelSlim tool. For details, see [msModelSlim Tool Installation Guide](../../../install_guide/install_guide.md).

## Principles and Implementation

### Principle

The Ceil_X algorithm addresses the large-value truncation problem in traditional MXFP4 per-block quantization by redesigning the shared exponent computation using **ceil** + **configurable divisor**.

**Problems with traditional floor scaling:**

The traditional method computes the shared exponent as follows:

$$s_{\text{floor}} = \lfloor \log_2(\max(|x|)) \rfloor - e_{\text{max}}$$

The magnitude of the maximum value in the scaled block is:

$$\frac{\max(|x|)}{2^{s_{\text{floor}}}} = \frac{\max(|x|)}{2^{\lfloor \log_2(\max(|x|)) \rfloor - 2}} = 4 \cdot \frac{\max(|x|)}{2^{\lfloor \log_2(\max(|x|)) \rfloor}} \in [4, 8)$$

Since $\max(|x|) / 2^{\lfloor \log_2(\max(|x|)) \rfloor} \in [1, 2)$, the overall range falls within $\,[4, 8)$. However, the upper representable limit of MXFP4 is 6.0, so values in the $\,[6, 8)$ interval are truncated, resulting in significant error.

**Ceil_X:**

Ceil_X introduces a divisor $c$ and switches to the ceil operation. For each block, the scaling factor $2^s$ is computed as follows:

$$s = \text{ceil}\left(\log_2\left(\frac{\max(|x|)}{c}\right)\right)$$

The magnitude of the maximum value in the block after scaling is:

$$\frac{\max(|x|)}{2^s} = \frac{\max(|x|)}{2^{\text{ceil}(\log_2(\max(|x|)/c))}}$$

When $\max(|x|) \in (c/2, c]$, $\log_2(\max/c) \in (-1, 0]$, $\text{ceil}=0$, and the scaled value is $\max(|x|) \in (c/2, c]$; when $\max(|x|) \in (c, 2c)$, $\log_2(\max/c) \in (0, 1)$, $\text{ceil}=1$, and the scaled value is $\max(|x|)/2 \in (c/2, c)$. Therefore, the scaled value range is compressed to $(c/2, c]$.

With the default $c = 7.25$, the scaled value range is $(3.625, 7.25]$. Compared with the traditional method's $[4, 8)$:

- The lower bound decreases from $4$ to $3.625$, preserving a larger representation range for small values and improving the quantization precision of small values.
- Only $(6, 7.25]$ remains in the $[6, 8)$ interval with slight truncation, and the truncation ratio is significantly reduced.

**Core idea:**

1. **Block-based processing**: Divide the weight matrix into independent data blocks of size 32 along the specified axis.
2. **Ceil_X scaling**: Compute the shared exponent for each data block:

   $$s = \text{ceil}\left(\log_2\left(\frac{\max(|x|)}{c} + \epsilon\right)\right) - e_{\text{max}}$$
   where $c$ is the configurable divisor (ceil_x_value), $\epsilon = 9.6 \times 10^{-7}$ is the numerical stability term, and $e_{\text{max}} = 2^{e_{\text{bits}}-1} = 2$.
3. **Adaptive search (optional)**: Search for the divisor $c$ that minimizes the MSE within the range $\,[c_{\text{min}}, c_{\text{max}}]$ with a step size of $c_{\text{step}}$:

   $$c^* = \arg\min_{c \in [c_{\text{min}}, c_{\text{max}}]} \sum_{\text{blocks}} \|x - \hat{x}(c)\|^2$$

### Implementation

The algorithm is implemented in [`msmodelslim/core/quantizer/impl/ceil_x.py`](../../../../../msmodelslim/core/quantizer/impl/ceil_x.py):

- Implementation class: `MXWeightPerBlockCeilX`
- Registered quantization type: `mxfp4_per_block_sym`
- Configuration model: `CeilXExtConfig`

**Core code logic:**

```python
# Compute per-block min/max
self.minmax_block_observer.update(weight_value, sync=False, shared_exp_axes=shared_exp_axes)
min_val, max_val = self.minmax_block_observer.get_min_max()

# Compute the ceil_x shared exponent
shared_exp = ceil(log2(max_val / ceil_x_value + 9.6e-7))
shared_exp = clip(shared_exp, -scale_emax - emax, scale_emax - emax)

# Quantize
w_q_storage = quantize(QStorage(FLOAT, weight_value), q_param)
```

**enable_search search strategy:**

```python
# Search for the optimal ceil_x_value within [search_min, search_max] with a step size of search_step
candidates = [search_min + i * search_step for i in range(num_steps)]
for value in candidates:
    q_param = ceil_x_qparam(..., ceil_x_value=value)
    recon = dequantize(quantize(weight, q_param), q_param).value
    mse = ((weight - recon) ** 2).mean().item()
    if mse < best_mse:
        best_mse, best_value = mse, value
```

## Applicable Requirements

- **Accuracy improvement**: Applicable to scenarios that require higher mxFP4 quantization accuracy, especially model layers with a wide weight distribution range where floor scaling results in an overly coarse step size.
- **Computing cost**: In non-search mode, the computing cost is the same as that of standard MXFP4 quantization; when enable_search is enabled, several additional forward quantization evaluations are added.

## Feature Description

### YAML Configuration Example

```yaml
spec:
  process:
    - type: "linear_quant"
      qconfig:
        weight:
          scope: "per_block"
          dtype: "mxfp4"
          symmetric: true
          method: "ceil_x"
          ext:
            ceil_x_value: 7.25        # Divisor, with a value range of [6.0, 12.0]
            enable_search: false      # Whether to enable MSE search
            search_min: 6.0           # Lower bound of the search range
            search_max: 12.0          # Upper bound of the search range
            search_step: 0.25         # Search step size
```

### Detailed Description of YAML Configuration Fields

#### qconfig.weight.ext (Weight Quantization Extension Parameters)

| Parameter Name | Operation | Optional Values | Description | Default Value |
|--------|------|--------|------|--------|
| `ceil_x_value` | Divisor $c$ | [6.0, 12.0] | Controls the tightening degree of the shared exponent | 7.25 |
| `enable_search` | Whether to enable MSE search | `true`, `false` | Searches for the optimal divisor within the search range | `false` |
| `search_min` | Lower bound of the search range | [6.0, 12.0] | Starting value of the MSE search | 6.0 |
| `search_max` | Upper bound of the search range | [6.0, 12.0] | Ending value of the MSE search, must be greater than search_min | 12.0 |
| `search_step` | Search step size | > 0 | Step interval during the search | 0.25 |

## Technical Advantages

### Comparison with Traditional mxFP4 Quantization

| Feature | Traditional mxFP4 Quantization | Ceil_X Quantization |
|------|----------------|-------------|
| Scaling factor | $2^{\lfloor \log_2(\max) \rfloor - 2}$ | $2^{\text{ceil}(\log_2(\max / c))}$ |
| Scaled value range | $[4, 8)$ | $(c/2, c] = (3.625, 7.25]$ |
| Large-value truncation | Values in $[6, 8)$ are truncated | Only $(6, 7.25]$ is slightly truncated, with a significantly reduced truncation ratio |
| Small-value representation range | Lower bound $4$ | Lower bound $3.625$, with higher quantization precision for small values |
| Adaptive search | Not supported | Optional MSE-optimal search |
| Computational complexity | O(N) | O(N) (without search) / O(kN) (with search) |

## FAQs

### Origin of the name Ceil_X

Ceil_X derives from the two core operations used by the algorithm: `ceil` (rounding up) and the configurable divisor `x` (ceil_x_value). Compared with floor scaling, the ceil operation compresses the scaled value range from $[4, 8)$ to $(c/2, c]$, which falls entirely within the representable range $[0, 6]$ of MXFP4, thereby avoiding truncation of large values.

### How is the default value 7.25 determined

**7.25** is obtained through multiple empirical validations in W4A4 MXFP4 symmetric quantization. With this value, the ceil operation compresses the scaled value range from $[4, 8)$ to $(3.625, 7.25]$, which falls entirely within the representable range of MXFP4, eliminating the truncation error for large values.

### What does the enable_search mode search for

The search looks for a global divisor $c$ on the weights of the same layer that minimizes the overall MSE (rather than performing independent per-block searches). The search result is stored in a separate field and does not modify the user's original configuration values.
