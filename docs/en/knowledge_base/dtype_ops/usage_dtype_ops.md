# Quantization Data Type Ops Usage Guide

## 1. Scope

This guide is intended for developers who need to call `msmodelslim.core.dtype_ops` directly. The package provides tensor-level quantization, dequantization, and parameter computation functions per data type, and it is not bound to linear layer, KVCache, or FA structures.

Applicable scenarios:

- Quantize weights or calibration activations offline to [INT8](term_int8.md), [INT4](term_int4.md), [FP8](term_fp8.md), [MXFP8](term_mxfp8.md), or [MXFP4](term_mxfp4.md);
- Dequantize at inference time with existing parameters, or compute the parameters dynamically online;
- Training-time fake quantization: output a float32 grid and backpropagate.

Not applicable to:

- Selecting a whole-layer quantization mode or writing a YAML recipe; use the [One-Click Quantization Usage Guide](../../user_guide/usage_one_click_quantization.md) and the corresponding mode entries instead;
- Storage packing (packing two 4-bit values into one byte); this layer does not perform pack / unpack.

For the type definitions, see the entries under [Quantization Data Types](README.md); for the formulas, see [Quantization and Dequantization](term_quantization.md). For the interface list, see the [Quantization Data Type Ops Python API](../../api_reference/python_api/dtype_ops.md).

## 2. Process Relationship and Prerequisites

**Upstream process**: enter this process when a quantization algorithm or mode implementation needs type-level quantization of a single tensor.

**Prerequisites**:

- msModelSlim matching the current repository is installed, and `import msmodelslim.core.dtype_ops` works;
- The target data type, granularity, and symmetry are determined, as is whether the tensor comes from weights, activations, or the KV cache;
- The input tensor can be converted to float32 for computation.

**Follow-up operations**: pass the returned `q` and scale / offset / shared_exp to the upper-layer mode or the persistence logic; if the accuracy does not meet the requirement, proceed to the [Quantization Accuracy Tuning Guide](../../user_guide/process_quantization_precision_tuning.md).

## 3. Inputs and Deliverables

| Type | Name | Source or Save Location | Format or Constraint | Acceptance Method |
| --- | --- | --- | --- | --- |
| Input | Floating-point tensor to be quantized | Weights, activations, or KV cache | Can be analyzed and computed in float32 | The shape is known and reduction along the selected axis is possible |
| Input | Data type and granularity selection | Determined by the caller according to the deployment target | INT8/INT4 support per_tensor, per_channel, and per_group; FP8 supports only per_tensor/per_channel; MX uses a block length of 32 | Consistent with the target mode, see step 1 |
| Deliverable | Quantization grid and quantization parameters | Function return value | `q` is a float32 grid of the same shape by default; symmetric INT returns scale, and asymmetric INT additionally returns offset; MX returns a uint8 E8M0 | Can be restored by the corresponding `dequantize`, and the parameters can be broadcast |
| Deliverable | Dequantized floating-point tensor | Output of calling `dequantize` | float32, with the same shape as the input | Has the same shape as the original tensor and can be used for subsequent computation or loss |

## 4. Process Overview

```mermaid
flowchart LR
  A[Select data type and granularity] --> B[Compute quantization parameters]
  B --> C[Quantize]
  C --> D[Dequantize]
```

- **Select data type and granularity**: first decide the encoding family, then decide the granularity and `sym` / `asym` in the function name.
- **Compute quantization parameters**: the static path calls `calc_qparam` first; the dynamic path is handled inside `quantize_dynamic`.
- **Quantize**: obtain `q` and the parameters returned as-is; pass `dtype` as well when low-precision storage is needed.
- **Dequantize**: restore `q` to float32 with the same set of granularity parameters.

## 5. Procedure

### Step 1: Select the Data Type, Granularity, and Symmetry

**Goal**: determine the function names to call and avoid a mismatch among the data type, the granularity, the symmetry, and the tensor axis.

**Operation**: import the module-level functions from `msmodelslim.core.dtype_ops`; they are named `{type}_{operation}_{granularity}_{sym|asym}`. The operations are `calc_qparam`, `quantize`, `dequantize`, and `quantize_dynamic`. Class names are not exported.

| Granularity | Function Name Suffix | Extra Input Parameter | Applicable Types |
| --- | --- | --- | --- |
| per-tensor | `_per_tensor_sym` / `_per_tensor_asym` | No `axis` | INT8, INT4, FP8; FP8 has no `_asym` |
| per-channel / per-token | `_per_channel_sym` / `_per_channel_asym` | `axis` | INT8, INT4, FP8; FP8 has no `_asym` |
| per-group | `_per_group_sym` / `_per_group_asym` | `axis`, `group_size` | INT8, INT4; the length of that dimension must be divisible by `group_size`, otherwise an error is reported |
| MX block | `_sym` (without `per_block`) | `axis=-1`, or `0`, or `[0, 1]` | MXFP8, MXFP4; padding is applied internally when the length is not a multiple of 32, and the result is trimmed back when restored |

On the same static path, `calc_qparam`, `quantize`, and `dequantize` must use the same granularity, and `axis` / `group_size` must stay consistent.

**Output**: a determined set of function names and granularity parameters.

### Step 2: Compute Quantization Parameters

**Goal**: obtain a scale / offset / shared_exp that can be broadcast and persisted.

**Operation**:

- Symmetric INT: `int8_calc_qparam_per_channel_sym` and the like, returning a float32 scale. The interval is forced to contain 0, and `scale = amax / 127` (`/7` for INT4).
- Asymmetric INT: returns `(scale, offset)`, where `offset` is the integer zero-point.
- FP8: `fp8_calc_qparam_per_channel_sym(x, axis, fmt)`, with `scale = amax / max_norm`.
- MX: `mxfp8_calc_qparam_sym` / `mxfp4_calc_qparam_sym` return the persistable `uint8` E8M0 directly.

`fmt` appears only in FP8 / MXFP8 and is used to select the element encoding; it is not a storage type. The default is `"e4m3"` (upper bound 448), and `"e5m2"` (upper bound 57344) is optional. Any other string raises `ValueError`. INT8, INT4, and MXFP4 do not take `fmt`. `calc_qparam` and the subsequent `quantize` must be passed the same `fmt`.

**Output**: a float32 scale (and an optional offset), or a `uint8` E8M0.

### Step 3: Perform Quantization and Dequantization

**Goal**: obtain a quantization grid with the same shape as the input, and restore it to float32.

**Operation**: when `dtype` is not passed, `q` is a float32 grid. Pass `dtype` when low-precision storage is needed: INT8 / INT4 / MXFP4 commonly use `torch.int8`, while FP8 / MXFP8 use the corresponding `float8`. All of them are element-wise conversions with no packing. A `q` that is not float32 cannot be backpropagated through.

```python
import torch
from msmodelslim.core.dtype_ops import (
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

**Output**: `(q, scale)`, `(q, scale, offset)`, or `(q, shared_exp)`, plus the float32 restored tensor.

### Step 4: Choose Static or Dynamic According to the Scenario

**Goal**: static parameters are prepared by the caller, while dynamic parameters are computed inside the function and detached from the computation graph.

**Operation**:

| Scenario | Call | Parameters |
| --- | --- | --- |
| Offline weight quantization | `calc_qparam` first, then `quantize` | The passed scale / offset / shared_exp are returned as-is |
| Inference dequantization | `dequantize` | Use the parameters returned at quantization time; `q` may be float32 or already converted to a lower precision |
| Online / dynamic quantization | `quantize_dynamic` | The parameters are computed on the spot, detached, and returned together with `q` |

For static training, if the parameters should be treated as constants, `detach` them before calling `quantize`.

**Output**: the same return values as in step 3; the parameters of the dynamic path are already detached.

### Step 5: Choose the Output Precision and Backpropagation

**Goal**: distinguish the float32 grid (which can be backpropagated through) from low-precision storage (which cannot).

**Operation**: `dtype` determines the type of `q`. `ste_clamp` appears only in `quantize` / `quantize_dynamic` and defaults to `True`: the forward pass saturates and the backward pass is the identity, so the gradient is still 1 at the saturation position; when it is `False`, the gradient is 0 at the saturation position. `round_ste` is always enabled.

```python
import torch
from msmodelslim.core.dtype_ops import (
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

**Output**: the float32 grid can be used to call `backward` on the loss; a low-precision `q` is suitable only for saving or inference.

## 6. Acceptance Criteria

- `calc_qparam`, `quantize`, and `dequantize` on the same path use the same granularity and the same `fmt`.
- The `q` returned on the default path has the same shape as the input; the shape still does not change after a low-precision `dtype` is passed.
- The output of `dequantize` is float32 and has the same shape as the input.
- `per_group` fails when the length is not divisible; MX handles non-divisible lengths with internal padding.

## 7. Terminology

| Term | Brief Description | Link |
| --- | --- | --- |
| Quantization and Dequantization | scale, zero_point, static and dynamic, granularity | [Quantization and Dequantization](term_quantization.md) |
| INT8 | 8-bit signed integer affine | [INT8](term_int8.md) |
| INT4 | 4-bit signed integer affine | [INT4](term_int4.md) |
| FP8 | OCP OFP8 per-element floating point | [FP8](term_fp8.md) |
| MXFP8 | OCP MX 8-bit block-level format | [MXFP8](term_mxfp8.md) |
| MXFP4 | OCP MX 4-bit block-level format | [MXFP4](term_mxfp4.md) |
| Quantization Modes | Select the types above according to the object | [Quantization Modes](../quantization_mode/README.md) |

## 8. Related Documents

| API or Document | Brief Description | Link |
| --- | --- | --- |
| Quantization Data Type Ops Python API | The public interface list of `msmodelslim.core.dtype_ops` | [Quantization Data Type Ops Python API](../../api_reference/python_api/dtype_ops.md) |
| Quantization Data Types | Standard definitions and entry index of each type | [Quantization Data Types](README.md) |
| One-Click Quantization Usage Guide | The execution entry point for quantization tasks | [One-Click Quantization Usage Guide](../../user_guide/usage_one_click_quantization.md) |
| Quantization Accuracy Tuning Guide | Troubleshooting accuracy problems caused by the data type | [Quantization Accuracy Tuning Guide](../../user_guide/process_quantization_precision_tuning.md) |
