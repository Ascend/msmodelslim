# DualScale: w4a4 Quantization Scheme Description

<!-- md-trans-meta sourceCommit=0dcb0df3ef37737beef6b9a2fa93197f635a1366 translatedAt=2026-08-19T08:16:26.626Z pushedAt=2026-08-19T08:17:52.368Z -->

## Introduction

- **Background**: Traditional grouped quantization typically adopts a single-scale strategy: each group of K-dimensional elements shares one scale factor. For example, in the MXFP4 format, every 32 elements use a shared exponent in FP8 format. However, structured outlier channels exist in activations. The values of certain channels are on average several orders of magnitude higher than those of other channels. Single-scale quantization struggles to accurately represent these vastly different value ranges simultaneously, resulting in accuracy loss.
- **Core idea**: Improve the precision of 4-bit quantization while maintaining hardware efficiency through two-level progressively refined scale factors.

## Prerequisites

Install the msModelSlim tool. For details, see [msModelSlim Tool Installation Guide](../../../install_guide/install_guide.md).

## Principles and Implementation

### Principle

The computing logic and core formulas are as follows:

#### 1. Dynamic Quantization and Dequantization of Weight Activations

**a) Dual Scale**

Divide the input X according to `x_dual_block_size`, compute the maximum absolute value of each large block, and obtain the dual scale $S_{dual\_x}$:
$$X_{dualscaled} = \frac{X}{S_{dual\_x}}, \quad S_{dual\_x} = \frac{\max(|X_{block}|)}{{MXFP4\_MAX\_NORMAL}}$$

**b) Inner Quant-Dequant**

Further divide $X_{dualscaled}$ according to `x_inner_block_size`, compute the inner scale, and convert it to the target low-bit format:
$$X_{q\_dq\_inner} = \text{mxfp4\_quantize\_dequantize}(X_{dualscaled}, S_{inner\_x})$$

**c) Dual Scale Dequantization**

$$X_{q\_dq} = X_{q\_dq\_inner} \times S_{dual\_x}$$

#### 2. Static Dequantization of Weights

Weights have already been quantized and stored during initialization. During forward propagation, only two-level dequantization is performed to restore them to high precision:

**a) Inner Dequantization**

Restore the base scale according to the inner parameter `inner_w_q_param`:
$$W_{dualscaled\_q\_dq} = \text{dequantize}(W_{quantized}, S_{inner\_w})$$

**b) Dual Scale Dequantization**

Multiply by the dual weight scale `weight_dual_scale` ($S_{dual\_w}$) fixed at model initialization:
$$W_{q\_dq} = W_{dualscaled\_q\_dq} \times S_{dual\_w}$$

#### 3. Linear Inverted

$$\text{Output} = X_{q\_dq} \cdot W_{q\_dq}^T + \text{bias}$$

### Implementation

The DualScale scheme implements dual-scale quantization through the collaboration of two quantizers, both of which are registered via `QABCRegistry.multi_register` with the dispatch_key `(qir.mxfp4_dual_scale_sym, "dualscale")`:

1. **Weight dual-scale quantizer** (`MXWeightDualScaleMinmax`, inheriting `AutoWeightQuantizer`): performs static quantization during the weight initialization phase.

   - Internally encapsulates an mxfp4 quantizer with `QScope.PER_BLOCK` to perform inner block quantization.

   - `init_weight` process:

     1. Reshape the weight into blocks according to `axes` and `dual_block_size`;

     2. Use `MsMinMaxBlockObserver` to collect the maximum value of each outer block and compute the dual scale $S_{dual\_w} = \frac{\max(|W_{block}|)}{\text{MXFP4\_MAX\_NORMAL}}$ (where `MXFP4_MAX_NORMAL = 6.0`);

     3. Divide the weight by $S_{dual\_w}$ and then pass it to the inner quantizer for mxfp4 quantization storage;

     4. Store the dual scale $S_{dual\_w}$ as a parameter in `q_param.ext['dual_scale']`.

   - `forward` (dequantization) flow:

     1. Invoke the inner quantizer to perform inner dequantization;

     2. Reshape the result into block form;

     3. Multiply by $S_{dual\_w}$ to restore the dual scale;

     4. Restore the original shape and return.

2. **Activation dual-scale quantizer** (`MXActDualScaleMinmax`, inheriting `AutoActQuantizer`): data-free quantization, with `is_data_free` returning `True`.

   - `forward` directly returns the input `x` (fake quantization; the actual quantization is performed by hardware during inference);

   - The configuration parameters are consistent with those of the weight quantizer, including `dual_block_size` and `axes`, ensuring that weights and activations adopt the same dual-scale block strategy.

## Applicable Requirements

- **Low-bit quantization**: suitable for 4-bit quantization in extremely low-bit quantization scenarios.

- **High-accuracy requirements**: maintains high model accuracy even under low-bit conditions.

- **Computing resources**: requires an additional optimization process, with higher computing cost than simple quantization methods.

- **Usage restrictions**:

    - Sufficient calibration data or training iterations are required to optimize the parameters. Because iterative optimization is involved, quantization takes longer than other methods.

    - Currently, this solution mainly targets low-bit quantization scenarios for the Qwen3 dense model series (such as Qwen3-8B/14B/32B), and generalization to other model series is not guaranteed.

## Feature Description

>[!NOTE]
>
>The algorithm implementation involves a training process and has certain requirements on NPU memory: >= 64 GB.

### YAML Configuration Example

When used as a Processor, the YAML configuration example is as follows:

```yaml
 process:
   - type: "linear_quant"
     qconfig:
       act:
         scope: "dual_scale"
         dtype: "mxfp4"
         symmetric: True
         method: "dualscale"
         ext: {
           dual_block_size: 512
         }
       weight:
         scope: "dual_scale"
         dtype: "mxfp4"
         symmetric: True
         method: "dualscale"
         ext: {
           dual_block_size: 512
         }
```

### YAML Configuration Fields

#### qconfig.act (Activation Quantization Configuration)

**Function**: Configures the quantization parameters for activations.

| Parameter Name | Function | Optional Value | Description | Default Value |
|-----------------|---------|-----------------|----------------------------------------|---------------|
| scope | Quantization scope | `"dual_scale"` | `dual_scale`: dual scale | `"per_block"` |
| dtype | Quantization data type | `"mxfp4"` | mxFP4 format quantization | `"mxfp4"` |
| symmetric | Whether to use symmetric quantization | `True`, `False` | `true`: symmetric quantization, zero point is 0<br/>`false`: asymmetric quantization, zero point is adjustable | `True` |
| method | Quantization method | `"dualscale"` | `dualscale`: two-level quantization algorithm | `"dualscale"` |
| ext | Extended configuration | `object` | Contains configuration parameters specific to DualScale | [See detailed configuration below](#ext-dualscale-extended-configuration) |

#### qconfig.weight (Weight Quantization Configuration)

**Function**: Configures the quantization parameters of the weights.

| Parameter Name             | Function      | Optional Value             | Description                                     | Default Value           |
|-----------------|---------|-----------------|----------------------------------------|---------------|
| scope           | Quantization scope    | `"dual_scale"`  | `dual_scale`: dual scale                        | `"per_block"` |
| dtype           | Quantization data type  | `"mxfp4"`       | mxFP4 format quantization                             | `"mxfp4"`     |
| symmetric       | Whether symmetric quantization  | `True`, `False` | `true`: symmetric quantization, zero point is 0<br/>`false`: asymmetric quantization, zero point is adjustable | `True`        |
| method          | Quantization method    | `"dualscale"`   | `dualscale`: two-level quantization algorithm                      | `"dualscale"` |
| ext             | Extended configuration    | `object`        | Contains configuration parameters specific to DualScale                  | [See detailed configuration below](#ext-dualscale-extended-configuration) |

#### ext (DualScale Extended Configuration)

**Function**: Configures parameters specific to the DualScale algorithm.

| Parameter Name             | Function      | Type     | Description                                | Example Value  |
|-----------------|---------|--------|-----------------------------------|-------|
| dual_block_size | block size | `int`  | `dual_block_size`: block size          | `512` |
