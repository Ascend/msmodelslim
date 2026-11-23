# mxfp8

## 功能说明

MXFP8对称量化、反量化与参数计算接口。块长固定 32。四类操作入参相近：共用分块轴 `axis`，量化参数为块级共享指数。

## 函数原型

```python
mxfp8_calc_qparam_sym(x, axis=-1, fmt="e4m3")
mxfp8_quantize_sym(x, shared_exp_t, axis=-1, fmt="e4m3", dtype=None, ste_clamp=True)
mxfp8_dequantize_sym(q, shared_exp_t, axis=-1)
mxfp8_quantize_dynamic_sym(x, axis=-1, fmt="e4m3", dtype=None, ste_clamp=True)
```

## 参数说明

| 参数名 | 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| x | 输入 | 待量化浮点张量。 | `calc_qparam` / `quantize` / `quantize_dynamic` 必选。<br>数据类型：torch.Tensor。<br>可转为 float32。 |
| q | 输入 | MXFP8 量化网格。 | `dequantize` 必选。<br>数据类型：torch.Tensor。 |
| shared_exp_t | 输入 | 块级共享指数。 | `quantize` / `dequantize` 必选。<br>数据类型：torch.Tensor。<br>uint8 E8M0 或 float32，须能广播到切块后的张量。 |
| axis | 输入 | 分块轴，块长固定 32。 | 可选。<br>数据类型：int 或 list。<br>默认 -1。 |
| fmt | 输入 | 元素编码。 | 可选。<br>数据类型：string。<br>默认 e4m3（上界 448）。<br>可选 e5m2（上界 57344）。<br>仅 `calc_qparam` / `quantize` / `quantize_dynamic` 使用，须与算参时一致。<br>`dequantize` 不传。 |
| dtype | 输入 | 量化网格存储类型。 | 可选。<br>数据类型：torch.dtype。<br>默认 None，表示 float32。<br>低精度按 fmt 传 torch.float8_e4m3fn 或 torch.float8_e5m2。非 float32 不能反传。 |
| ste_clamp | 输入 | 饱和是否使用 STE。 | 可选。<br>仅 `quantize` / `quantize_dynamic`。<br>数据类型：bool。<br>默认 True：前向饱和、反向恒等。<br>False 时饱和位置梯度为 0。 |
| shared_exp | 返回值 | 块级共享指数。 | `calc_qparam` 返回 uint8 E8M0。<br>`quantize` 原样返回。<br>`quantize_dynamic` 已 detach。 |
| q | 返回值 | 与 x 同形状的量化网格。 | `quantize` / `quantize_dynamic` 返回。<br>数据类型：由 dtype 决定，默认 float32。 |
| x_hat | 返回值 | 反量化浮点张量。 | `dequantize` 返回。<br>数据类型：torch.Tensor，float32。 |

## 调用示例

```python
import torch
from msmodelslim.core.dtype_ops import (
    mxfp8_calc_qparam_sym,
    mxfp8_quantize_sym,
    mxfp8_dequantize_sym,
)

x = torch.randn(2, 32)
se = mxfp8_calc_qparam_sym(x, axis=-1, fmt="e4m3")
q, se = mxfp8_quantize_sym(x, se, axis=-1, fmt="e4m3")
x_hat = mxfp8_dequantize_sym(q, se, axis=-1)
```
