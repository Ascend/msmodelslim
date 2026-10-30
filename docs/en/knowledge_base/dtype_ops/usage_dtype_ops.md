# Quantization Data Type ops Usage Workflow Guide

## 1. Scope

This guide is intended for developers who need to call `msmodelslim.ir.qal.dtype_ops` directly. This package provides tensor-level quantization, dequantization, and parameter calculation functions by data type, and is not bound to the linear layer, KVCache, or FA structures.

Applicable scenarios:

- Quantizing weights or calibration activations offline to [INT8](term_int8.md), [INT4](term_int4.md), [FP8](term_fp8.md), [MXFP8](term_mxfp8.md), or [MXFP4](term_mxfp4.md);
- Dequantizing with existing parameters during inference, or computing parameters online dynamically;
- Training fake quantization: outputting a float32 grid and back-propagating.

Not applicable to:

- Selecting a whole-layer quantization mode or writing a YAML recipe; use [One-Click Quantization Usage Guide](../../user_guide/usage_one_click_quantization.md) and the corresponding mode entries instead;
- Storage packing (two 4-bit values packed into one byte); this layer does not perform pack / unpack.

For format definitions, see the entries under [Quantization Data Type](README.md); for formulas, see [Quantization and Dequantization](term_quantization.md).

## 2. Workflow Relationships and Prerequisites

**Parent workflow**: this workflow is entered when a single tensor needs format-level quantization during the implementation of a quantization algorithm or mode.

**Prerequisites**:

- msModelSlim matching the current repository is installed, and `import msmodelslim.ir.qal.dtype_ops` works;
- The target format, granularity, and symmetry have been determined, as well as whether the tensor comes from weights, activations, or KV;
- The input tensor can be converted to float32 for computation.

**Follow-up operations**: hand the returned `q` together with scale / offset / shared_exp over to the upper-layer mode or the persistence logic; if accuracy does not meet the target, go to [Quantization Accuracy Tuning Guide](../../user_guide/process_quantization_precision_tuning.md).

## 3. Inputs and Deliverables

| Type | Name | Source or Save Location | Format or Constraint | Acceptance Method |
| --- | --- | --- | --- | --- |
| Input | Floating-point tensor to be quantized | Weights, activations, or KV cache | Can be computed and reduced in float32 | Shape known, reducible along the selected axis |
| Input | Format and granularity selection | Determined by the caller based on the deployment target | INT8/INT4 support per_tensor, per_channel, per_group; FP8 supports per_tensor/per_channel only; MX uses block size 32 | Consistent with the target mode, see step 1 |
| Deliverable | Quantization grid and quantization parameters | Function return value | `q` is a float32 grid of the same shape by default; INT symmetric returns scale, asymmetric additionally returns offset; MX returns uint8 E8M0 | Can be restored by the corresponding `dequantize`; parameters are broadcastable |
| Deliverable | Dequantized floating-point tensor | Output of calling `dequantize` | float32, same shape as the input | Same shape as the original tensor, usable for subsequent computation or loss |

## 4. Workflow Overview

```mermaid
flowchart LR
  A[Select format and granularity] --> B[Calculate quantization parameters]
  B --> C[Quantize]
  C --> D[Dequantize]
```

- **Select format and granularity**: determine the encoding family first, then the granularity and `sym` / `asym` in the function name.
- **Calculate quantization parameters**: call `calc_qparam` first on the static path; on the dynamic path this is done inside `quantize_dynamic`.
- **Quantize**: obtain `q` and the parameters returned as-is; pass `dtype` only when low-precision storage is needed.
- **Dequantize**: restore `q` to float32 using the same set of granularity parameters.

## 5. Procedure

### Step 1: Select the Data Type, Granularity, and Symmetry

**Goal**: determine the function name to call, and avoid inconsistency between the format, granularity, symmetry, and tensor axis.

**Operation**: import the module-level functions from `msmodelslim.ir.qal.dtype_ops`, named `{format}_{operation}_{granularity}_{sym|asym}`. The operations are `calc_qparam`, `quantize`, `dequantize`, and `quantize_dynamic`. Class names are not exported.

| Granularity | Function Name Suffix | Additional Arguments | Applicable Formats |
| --- | --- | --- | --- |
| per-tensor | `_per_tensor_sym` / `_per_tensor_asym` | No `axis` | INT8, INT4, FP8; FP8 has no `_asym` |
| per-channel / per-token | `_per_channel_sym` / `_per_channel_asym` | `axis` | INT8, INT4, FP8; FP8 has no `_asym` |
| per-group | `_per_group_sym` / `_per_group_asym` | `axis`, `group_size` | INT8, INT4; the length of this dimension must be divisible by `group_size`, otherwise an error is raised |
| MX block | `_sym` (without `per_block`) | `axis=-1`, or `0`, or `[0, 1]` | MXFP8, MXFP4; if not a multiple of 32, padding is applied internally and cropped back on restore |

On the same static path, `calc_qparam`, `quantize`, and `dequantize` must use the same granularity, and `axis` / `group_size` must stay consistent.

**Output**: a determined set of function names and granularity parameters.

### Step 2: Calculate the Quantization Parameters

**Goal**: obtain scale / offset / shared_exp that are broadcastable and can be persisted.

**Operation**:

- INT symmetric: `int8_calc_qparam_per_channel_sym` and similar, returning a float32 scale. The interval is forced to include 0, and `scale = amax / 127` (`/7` for INT4).
- INT asymmetric: returns `(scale, offset)`, where `offset` is the integer zero-point.
- FP8: `fp8_calc_qparam_per_channel_sym(x, axis, fmt)`, with `scale = amax / max_norm`.
- MX: `mxfp8_calc_qparam_sym` / `mxfp4_calc_qparam_sym` directly return a persistable `uint8` E8M0.

`fmt` appears only in FP8 / MXFP8, and is used to select the element encoding, not the storage type. It defaults to `"e4m3"` (upper bound 448), and `"e5m2"` (upper bound 57344) is optional. Any other string raises `ValueError`. INT8, INT4, and MXFP4 do not take `fmt`. `calc_qparam` and the subsequent `quantize` must pass the same `fmt`.

**Output**: a float32 scale (and an optional offset), or a `uint8` E8M0.

### Step 3: Perform Quantization and Dequantization

**Goal**: obtain a quantization grid of the same shape as the input, and be able to restore it to float32.

**Operation**: when `dtype` is not passed, `q` is a float32 grid. Pass `dtype` only when low-precision storage is needed: `torch.int8` is commonly used for INT8 / INT4 / MXFP4, and the corresponding `float8` for FP8 / MXFP8. All are element-wise conversions without packing. A non-float32 `q` cannot be back-propagated.

```python
import torch
from msmodelslim.ir.qal.dtype_ops import (
    int8_calc_qparam_per_channel_sym, int8_quantize_per_channel_sym, int8_dequantize_per_channel_sym,
    int4_calc_qparam_per_group_sym, int4_quantize_per_group_sym, int4_dequantize_per_group_sym,
    fp8_quantize_dynamic_per_channel_sym, fp8_dequantize_per_channel_sym,
    mxfp8_quantize_dynamic_sym, mxfp8_dequantize_sym,
    mxfp4_calc_qparam_sym, mxfp4_quantize_sym, mxfp4_dequantize_sym,
)

scale = int8_calc_qparam_per_channel_sym(w, axis=0)
q, scale = int8_quantize_per_channel_sym(w, scale, axis=0, dtype=torch.int8)
w_hat = int8_dequantize_per_channel_sym(q, scale, axis=0)

scale4 = int4_calc_qparam_per_group_sym(w, axis=-1, group_size=128)
q4, scale4 = int4_quantize_per_group_sym(w, scale4, axis=-1, group_size=128, dtype=torch.int8)
w_hat4 = int4_dequantize_per_group_sym(q4, scale4, axis=-1, group_size=128)

q_fp8, scale_fp8 = fp8_quantize_dynamic_per_channel_sym(
    x, axis=-1, fmt="e4m3", dtype=torch.float8_e4m3fn
)
x_fp8 = fp8_dequantize_per_channel_sym(q_fp8, scale_fp8, axis=-1)

q_mx, se_mx = mxfp8_quantize_dynamic_sym(x, axis=-1, fmt="e4m3")
x_mx = mxfp8_dequantize_sym(q_mx, se_mx, axis=-1)

se = mxfp4_calc_qparam_sym(w, axis=-1)
q_mx4, se = mxfp4_quantize_sym(w, se, axis=-1)
w_mx = mxfp4_dequantize_sym(q_mx4, se, axis=-1)
```

**Output**: `(q, scale)`, or `(q, scale, offset)`, or `(q, shared_exp)`, together with the float32 restored tensor.

### Step 4: Choose Static or Dynamic by Scenario

**Goal**: static parameters are prepared by the caller; dynamic parameters are computed inside the function and detached from the computation graph.

**Operation**:

| Scenario | Call | Parameters |
| --- | --- | --- |
| Offline weight quantization | `calc_qparam` first, then `quantize` | The passed-in scale / offset / shared_exp are returned as-is |
| Inference dequantization | `dequantize` | Uses the parameters returned during quantization; `q` can be float32 or an already-converted low-precision type |
| Online / dynamic quantization | `quantize_dynamic` | Parameters are computed on the spot, `detach`ed, and returned together with `q` |

If static training needs to treat the parameters as constants, `detach` the parameters before calling `quantize`.

**Output**: the same return values as step 3; the parameters on the dynamic path are already `detach`ed.

### Step 5: Choose the Output Precision and Back-Propagation

**Goal**: distinguish the float32 grid (back-propagable) from low-precision storage (not back-propagable).

**Operation**: `dtype` determines the type of `q`. `ste_clamp` appears only in `quantize` / `quantize_dynamic` and defaults to `True`: saturation in the forward pass and identity in the backward pass, so the gradient at the saturation position is still 1; when `False`, the gradient at the saturation position is 0. `round_ste` is always enabled.

```python
import torch
from msmodelslim.ir.qal.dtype_ops import (
    int8_calc_qparam_per_channel_sym, int8_quantize_per_channel_sym,
    int8_dequantize_per_channel_sym, int8_quantize_dynamic_per_channel_sym,
)

scale = int8_calc_qparam_per_channel_sym(x.detach(), axis=-1).detach()
q, scale = int8_quantize_per_channel_sym(x, scale, axis=-1, dtype=torch.float32)
y = int8_dequantize_per_channel_sym(q, scale, axis=-1)

q, scale = int8_quantize_per_channel_sym(
    x, scale, axis=-1, dtype=torch.float32, ste_clamp=False
)

q, scale = int8_quantize_dynamic_per_channel_sym(x, axis=-1, dtype=torch.float32)
y = int8_dequantize_per_channel_sym(q, scale, axis=-1)
```

**Output**: the float32 grid can `backward` the loss; a low-precision `q` is only suitable for saving or inference.

## 6. Acceptance Criteria

- `calc_qparam`, `quantize`, and `dequantize` on the same path use the same granularity and the same `fmt`.
- The `q` returned by the default path has the same shape as the input; the shape still does not change after a low-precision `dtype` is passed in.
- The output of `dequantize` is float32 and has the same shape as the input.
- `per_group` fails when not divisible; MX is handled by internal padding when not divisible.

## 7. Terminology

| Term | Brief Description | Link |
| --- | --- | --- |
| Quantization and Dequantization | scale, zero_point, static vs. dynamic, and granularity | [Quantization and Dequantization](term_quantization.md) |
| INT8 | 8-bit signed integer affine | [INT8](term_int8.md) |
| INT4 | 4-bit signed integer affine | [INT4](term_int4.md) |
| FP8 | OCP OFP8 per-element floating point | [FP8](term_fp8.md) |
| MXFP8 | OCP MX 8-bit block-level format | [MXFP8](term_mxfp8.md) |
| MXFP4 | OCP MX 4-bit block-level format | [MXFP4](term_mxfp4.md) |
| Quantization Modes | Selecting the formats above by object | [Quantization Modes](../quantization_mode/README.md) |
