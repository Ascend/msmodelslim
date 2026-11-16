# Quantization Data Type Ops Python API

## Function Description

`msmodelslim.core.dtype_ops` provides tensor-level quantization, dequantization, and parameter computation interfaces covering INT8, INT4, FP8, MXFP8, and MXFP4. The public entry points are module-level functions; classes are not exported. For naming, parameters, and calling steps, see [Quantization Data Type Ops Usage Guide](../../knowledge_base/dtype_ops/usage_dtype_ops.md); for the concept entries, see [Quantization Data Types](../../knowledge_base/dtype_ops/README.md).

## Function Prototype

```python
from msmodelslim.core.dtype_ops import (
    int8_calc_qparam_per_channel_sym,
    int8_quantize_per_channel_sym,
    int8_dequantize_per_channel_sym,
)
```

Function names are `{type}_{operation}_{granularity}_{sym|asym}`. The operations are `calc_qparam`, `quantize`, `dequantize`, and `quantize_dynamic`. MX does not add a granularity suffix and only takes `_sym`.

## Interface List

| Interface | Function |
| --- | --- |
| `int8_calc_qparam_per_tensor_sym` / `_per_channel_sym` / `_per_group_sym` | INT8 symmetric parameter computation |
| `int8_quantize_per_tensor_sym` / `_per_channel_sym` / `_per_group_sym` | INT8 symmetric static quantization |
| `int8_dequantize_per_tensor_sym` / `_per_channel_sym` / `_per_group_sym` | INT8 symmetric dequantization |
| `int8_quantize_dynamic_per_tensor_sym` / `_per_channel_sym` / `_per_group_sym` | INT8 symmetric dynamic quantization |
| `int8_calc_qparam_per_tensor_asym` / `_per_channel_asym` / `_per_group_asym` | INT8 asymmetric parameter computation |
| `int8_quantize_per_tensor_asym` / `_per_channel_asym` / `_per_group_asym` | INT8 asymmetric static quantization |
| `int8_dequantize_per_tensor_asym` / `_per_channel_asym` / `_per_group_asym` | INT8 asymmetric dequantization |
| `int8_quantize_dynamic_per_tensor_asym` / `_per_channel_asym` / `_per_group_asym` | INT8 asymmetric dynamic quantization |
| `int4_calc_qparam_per_tensor_sym` / `_per_channel_sym` / `_per_group_sym` | INT4 symmetric parameter computation |
| `int4_quantize_per_tensor_sym` / `_per_channel_sym` / `_per_group_sym` | INT4 symmetric static quantization |
| `int4_dequantize_per_tensor_sym` / `_per_channel_sym` / `_per_group_sym` | INT4 symmetric dequantization |
| `int4_quantize_dynamic_per_tensor_sym` / `_per_channel_sym` / `_per_group_sym` | INT4 symmetric dynamic quantization |
| `int4_calc_qparam_per_tensor_asym` / `_per_channel_asym` / `_per_group_asym` | INT4 asymmetric parameter computation |
| `int4_quantize_per_tensor_asym` / `_per_channel_asym` / `_per_group_asym` | INT4 asymmetric static quantization |
| `int4_dequantize_per_tensor_asym` / `_per_channel_asym` / `_per_group_asym` | INT4 asymmetric dequantization |
| `int4_quantize_dynamic_per_tensor_asym` / `_per_channel_asym` / `_per_group_asym` | INT4 asymmetric dynamic quantization |
| `fp8_calc_qparam_per_tensor_sym` / `_per_channel_sym` | FP8 symmetric parameter computation |
| `fp8_quantize_per_tensor_sym` / `_per_channel_sym` | FP8 symmetric static quantization |
| `fp8_dequantize_per_tensor_sym` / `_per_channel_sym` | FP8 symmetric dequantization |
| `fp8_quantize_dynamic_per_tensor_sym` / `_per_channel_sym` | FP8 symmetric dynamic quantization |
| `mxfp8_calc_qparam_sym` | MXFP8 parameter computation |
| `mxfp8_quantize_sym` | MXFP8 static quantization |
| `mxfp8_dequantize_sym` | MXFP8 dequantization |
| `mxfp8_quantize_dynamic_sym` | MXFP8 dynamic quantization |
| `mxfp4_calc_qparam_sym` | MXFP4 parameter computation |
| `mxfp4_quantize_sym` | MXFP4 static quantization |
| `mxfp4_dequantize_sym` | MXFP4 dequantization |
| `mxfp4_quantize_dynamic_sym` | MXFP4 dynamic quantization |

## Call Example

```python
import torch
from msmodelslim.core.dtype_ops import (
    int8_calc_qparam_per_channel_sym,
    int8_quantize_per_channel_sym,
    int8_dequantize_per_channel_sym,
)

w = torch.randn(4, 8)
scale = int8_calc_qparam_per_channel_sym(w, axis=0)
q, scale = int8_quantize_per_channel_sym(w, scale, axis=0, dtype=torch.int8)
w_hat = int8_dequantize_per_channel_sym(q, scale, axis=0)
```

For the remaining combinations and parameter descriptions, see [Quantization Data Type Ops Usage Guide](../../knowledge_base/dtype_ops/usage_dtype_ops.md).
