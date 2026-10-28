# 量化数据类型 ops Python API

## 功能说明

`msmodelslim.ir.qal.dtype_ops` 提供张量级量化、反量化与参数计算接口，覆盖 INT8、INT4、FP8、MXFP8、MXFP4。公开入口是模块级函数，类不对外导出。命名、参数与调用步骤见《[量化数据类型 ops 使用流程指南](../../knowledge_base/dtype_ops/usage_dtype_ops.md)》。

## 函数原型

```python
from msmodelslim.ir.qal.dtype_ops import (
    int8_calc_qparam_per_channel_sym,
    int8_quantize_per_channel_sym,
    int8_dequantize_per_channel_sym,
)
```

函数名为 `{类型}_{操作}_{粒度}_{sym|asym}`。操作为 `calc_qparam`、`quantize`、`dequantize`、`quantize_dynamic`。MX 不加粒度后缀，只接 `_sym`。

## 接口清单

| 接口 | 功能 |
| --- | --- |
| `int8_calc_qparam_per_tensor_sym` / `_per_channel_sym` / `_per_group_sym` | INT8 对称算参 |
| `int8_quantize_per_tensor_sym` / `_per_channel_sym` / `_per_group_sym` | INT8 对称静态量化 |
| `int8_dequantize_per_tensor_sym` / `_per_channel_sym` / `_per_group_sym` | INT8 对称反量化 |
| `int8_quantize_dynamic_per_tensor_sym` / `_per_channel_sym` / `_per_group_sym` | INT8 对称动态量化 |
| `int8_calc_qparam_per_tensor_asym` / `_per_channel_asym` / `_per_group_asym` | INT8 非对称算参 |
| `int8_quantize_per_tensor_asym` / `_per_channel_asym` / `_per_group_asym` | INT8 非对称静态量化 |
| `int8_dequantize_per_tensor_asym` / `_per_channel_asym` / `_per_group_asym` | INT8 非对称反量化 |
| `int8_quantize_dynamic_per_tensor_asym` / `_per_channel_asym` / `_per_group_asym` | INT8 非对称动态量化 |
| `int4_calc_qparam_per_tensor_sym` / `_per_channel_sym` / `_per_group_sym` | INT4 对称算参 |
| `int4_quantize_per_tensor_sym` / `_per_channel_sym` / `_per_group_sym` | INT4 对称静态量化 |
| `int4_dequantize_per_tensor_sym` / `_per_channel_sym` / `_per_group_sym` | INT4 对称反量化 |
| `int4_quantize_dynamic_per_tensor_sym` / `_per_channel_sym` / `_per_group_sym` | INT4 对称动态量化 |
| `int4_calc_qparam_per_tensor_asym` / `_per_channel_asym` / `_per_group_asym` | INT4 非对称算参 |
| `int4_quantize_per_tensor_asym` / `_per_channel_asym` / `_per_group_asym` | INT4 非对称静态量化 |
| `int4_dequantize_per_tensor_asym` / `_per_channel_asym` / `_per_group_asym` | INT4 非对称反量化 |
| `int4_quantize_dynamic_per_tensor_asym` / `_per_channel_asym` / `_per_group_asym` | INT4 非对称动态量化 |
| `fp8_calc_qparam_per_tensor_sym` / `_per_channel_sym` | FP8 对称算参 |
| `fp8_quantize_per_tensor_sym` / `_per_channel_sym` | FP8 对称静态量化 |
| `fp8_dequantize_per_tensor_sym` / `_per_channel_sym` | FP8 对称反量化 |
| `fp8_quantize_dynamic_per_tensor_sym` / `_per_channel_sym` | FP8 对称动态量化 |
| `mxfp8_calc_qparam_sym` | MXFP8 算参 |
| `mxfp8_quantize_sym` | MXFP8 静态量化 |
| `mxfp8_dequantize_sym` | MXFP8 反量化 |
| `mxfp8_quantize_dynamic_sym` | MXFP8 动态量化 |
| `mxfp4_calc_qparam_sym` | MXFP4 算参 |
| `mxfp4_quantize_sym` | MXFP4 静态量化 |
| `mxfp4_dequantize_sym` | MXFP4 反量化 |
| `mxfp4_quantize_dynamic_sym` | MXFP4 动态量化 |

## 调用示例

```python
import torch
from msmodelslim.ir.qal.dtype_ops import (
    int8_calc_qparam_per_channel_sym,
    int8_quantize_per_channel_sym,
    int8_dequantize_per_channel_sym,
)

w = torch.randn(4, 8)
scale = int8_calc_qparam_per_channel_sym(w, axis=0)
q, scale = int8_quantize_per_channel_sym(w, scale, axis=0, dtype=torch.int8)
w_hat = int8_dequantize_per_channel_sym(q, scale, axis=0)
```

其余组合与参数说明见《[量化数据类型 ops 使用流程指南](../../knowledge_base/dtype_ops/usage_dtype_ops.md)》。
