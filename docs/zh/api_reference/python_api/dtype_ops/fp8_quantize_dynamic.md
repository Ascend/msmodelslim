# fp8_quantize_dynamic

## 功能说明

FP8 对称动态量化。在函数内计算 scale 并 `detach`，再完成量化。粒度由函数名后缀区分：`per_tensor`、`per_channel`。无非对称接口。

## 函数原型

```python
fp8_quantize_dynamic_per_tensor_sym(x, fmt="e4m3", dtype=None, ste_clamp=True)
fp8_quantize_dynamic_per_channel_sym(x, axis, fmt="e4m3", dtype=None, ste_clamp=True)
```

## 参数说明

| 参数名 | 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| x | 输入 | 待量化浮点张量。 | 必选。<br>数据类型：torch.Tensor。<br>可转为 float32。 |
| axis | 输入 | 量化轴。 | `_per_channel_*` 必选。<br>`_per_tensor_*` 不传。<br>数据类型：int。 |
| fmt | 输入 | 元素编码。 | 可选。<br>数据类型：string。<br>默认 e4m3（上界 448）。可选 e5m2（上界 57344）。 |
| dtype | 输入 | 量化网格存储类型。 | 可选。<br>数据类型：torch.dtype。<br>默认 None，表示 float32。<br>低精度按 fmt 传 torch.float8_e4m3fn 或 torch.float8_e5m2。非 float32 不能反传。 |
| ste_clamp | 输入 | 饱和是否使用 STE。 | 可选。<br>数据类型：bool。<br>默认 True：前向饱和、反向恒等。<br>False 时饱和位置梯度为 0。 |
| q | 返回值 | 与 x 同形状的量化网格。 | 数据类型：由 dtype 决定，默认 float32。 |
| scale | 返回值 | 量化 scale，已 detach。 | 数据类型：torch.Tensor，float32。 |

## 调用示例

```python
import torch
from msmodelslim.core.dtype_ops import fp8_quantize_dynamic_per_channel_sym

x = torch.randn(2, 8)
q, scale = fp8_quantize_dynamic_per_channel_sym(x, axis=-1, fmt="e4m3")
```
