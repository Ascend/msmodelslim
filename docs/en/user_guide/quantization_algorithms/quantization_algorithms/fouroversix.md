# Four Over Six: Adaptive Block Scaling Weight Quantization Algorithm

<!-- md-trans-meta sourceCommit=3909078a95b1f3d532dd72a3a08143f4d01c1435 translatedAt=2026-08-19T08:16:14.966Z pushedAt=2026-08-19T08:17:52.361Z -->

## Introduction

- **Problem**: Traditional mxFP4 quantization methods uniformly use a fixed scaling factor (typically scaling the maximum value to the upper limit of FP4, which is 6) when processing data blocks with different distributions, which may lead to large quantization errors in some blocks.

- **Objective**: By adaptively selecting the optimal scaling factor for each data block, minimize the overall quantization error while maintaining the mxFP4 quantization format, thereby improving model accuracy.

- **Paper link**: `https://arxiv.org/abs/2512.02010`

## Prerequisites

Install the msModelSlim tool. For details, see [*msModelSlim Tool Installation Guide*](../../../install_guide/install_guide.md).

## Principles and Implementation

### Principle

The core idea of the Four Over Six (4/6) algorithm is to **adaptively select** the optimal scaling scheme for each data block:

**Core idea:**

1. **Block-based processing**: Divide the weight matrix into multiple independent data blocks according to the specified block size.

2. **Dual-path evaluation**: Try two scaling schemes simultaneously for each data block:

   - **Scheme A (Scaling-to-6)**: Scale the maximum value in the block to 6, the maximum value of the FP4 format, to fully utilize the dynamic range.

   - **Scheme B (Scaling-to-4)**: Scale the maximum value in the block to 4, providing more headroom for the data distribution.

3. **Intelligent selection**: Compute the mean squared error (MSE) of both schemes and select the scheme with the smaller error as the final quantization scheme for the block.

4. **Exponent rounding**: The scaling factor is stored in the e8m0 format, and precision is ensured through nearest-neighbor rounding (including the banker's rounding rule).

### Implementation

The algorithm is implemented in [`msmodelslim/core/quantizer/impl/fouroversix.py`](../../../../../msmodelslim/core/quantizer/impl/fouroversix.py):

- Implementation class: `WeightFouroverSixQuantizer`

- Registered quantization type: `mxfp4_per_block_sym`

**Core code logic:**

```python
# Scheme A: Scaling to 6
scale_a = max_per_block / 6.0
scale_E_a = self.__nearest_neighbor_rounding_to_e8m0(scale_a)
# ... quantization, dequantization ...
mse_a = torch.mean((weight_tensor - dequantized_weights_a) ** 2, dim=-1)

# Scheme B: Scaling to 4
scale_b = max_per_block / 4.0
scale_E_b = self.__nearest_neighbor_rounding_to_e8m0(scale_b)
# ... quantization, dequantization ...
mse_b = torch.mean((weight_tensor - dequantized_weights_b) ** 2, dim=-1)

# Select the scheme with the smaller MSE
mask = mse_a <= mse_b
selected_scale = torch.where(mask, scale_E_a, scale_E_b)
reshape_quantized_weights = torch.where(mask, quantized_weights_a, quantized_weights_b)
```

**Exponent rounding strategy (e8m0 format):**

- **Mantissa > 0.5**: increment the exponent by 1.

- **Mantissa < 0.5**: keep the exponent unchanged.

- **Mantissa == 0.5**: apply the banker's rounding rule (round to even, increment on even, no increment on odd).

## Applicable Requirements

- **High-precision requirements**: Applicable to mxFP4 quantization scenarios that demand high precision, and particularly suitable for models with uneven data distribution.

- **Computing cost**: The 4/6 algorithm needs to perform two quantization/dequantization operations on each block and calculate the MSE, resulting in slightly higher computing overhead than traditional mxFP4 quantization.

- **Usage restrictions**:

  - Only per_block symmetric quantization in the mxFP4 format is supported.

  - The weight must be a 2D tensor.

## Feature Description

### YAML Configuration Example

When used as a Processor, the YAML configuration example is as follows:

```yaml
spec:
  process:
    - type: "linear_quant"
      qconfig:
        weight:
          scope: "per_block"    # Quantization scope
          dtype: "mxfp4"        # Quantization data type
          symmetric: true       # Whether symmetric quantization is used
          method: "fouroversix" # Quantization algorithm - FouroverSix
```

### Detailed Description of YAML Configuration Fields

#### qconfig.weight (Weight Quantization Configuration)

| Parameter Name | Function | Optional Value | Description | Default Value |
|-----------|--------|--------------------------|-----------------------------------------------|-----------------------------|
| scope     | Quantization scope | `"per_block"` | `per_block`: independent parameters for each block | `"per_block"` |
| dtype     | Quantization data type | `"mxfp4"` | mxFP4 format quantization | `"mxfp4"` |
| symmetric | Whether to use symmetric quantization | `true`, `false` | `true`: symmetric quantization, zero point is 0<br/>`false`: asymmetric quantization, zero point is adjustable | `true` |
| method    | Quantization method | `"fouroversix"` | `fouroversix`: adaptive block scaling weight quantization algorithm | `"fouroversix"` |

## Technical Advantages

### Comparison with Traditional mxFP4 Quantization

| Feature | Traditional mxFP4 Quantization | FouroverSix Quantization |
|------|----------------|-----------------|
| Scaling strategy | Fixed (maximum value → 6) | Adaptive (6 or 4) |
| Quantization error | Large error in some blocks | Optimal for each block |
| Computational complexity | O(N) | O(2N) |
| Accuracy performance | Baseline | Typically improved by 0.5–2.0 percentage points |

### Applicable Scenarios

- **Transformer models**: The weight distributions of attention layers and feed-forward network layers differ significantly, and FouroverSix can handle them adaptively.

- **Multimodal models**: The data distributions of different modalities vary significantly, and adaptive scaling can effectively improve accuracy.

- **Large model quantization**: The larger the model, the more diverse the weight distributions, and the more obvious the advantages of FouroverSix become.

## FAQs

### What is the origin of the FouroverSix algorithm name?

The name FouroverSix comes from the core idea of the algorithm: for the FP4 format (with a maximum value of 6), each data block can choose to scale its maximum value to 6 (fully utilizing the dynamic range) or to 4 (providing more headroom). "4-over-6" means intelligently choosing whether to use 4 as the actual scaling target within the upper limit of 6.

### Why choose 4 as the alternative scaling target?

- The dynamic range of the FP4 format is [-6, 6]. Choosing 4 as the alternative provides approximately 33% headroom for the data distribution.

- Experience shows that when outliers exist within a data block or the distribution is uneven, using a smaller scaling factor can significantly reduce quantization error.

- 4 is approximately 2/3 of 6, which is a mathematically reasonable balance point.
