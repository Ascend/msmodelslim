# int8_quantize

## 功能说明

INT8 静态量化。输入已有量化参数，返回与 x 同形状的量化网格，并原样带回量化参数。粒度由函数名后缀区分：`per_tensor`、`per_channel`、`per_group`。

## 函数原型

```python
int8_quantize_per_tensor_sym(x, scale, dtype=None, ste_clamp=True)
int8_quantize_per_channel_sym(x, scale, axis, dtype=None, ste_clamp=True)
int8_quantize_per_group_sym(x, scale, axis, group_size, dtype=None, ste_clamp=True)
int8_quantize_per_tensor_asym(x, scale, offset, dtype=None, ste_clamp=True)
int8_quantize_per_channel_asym(x, scale, offset, axis, dtype=None, ste_clamp=True)
int8_quantize_per_group_asym(x, scale, offset, axis, group_size, dtype=None, ste_clamp=True)
```

## 参数说明

| 参数名 | 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| x | 输入 | 待量化浮点张量。 | 必选。<br>数据类型：torch.Tensor。<br>可转为 float32。 |
| scale | 输入 | 量化 scale。 | 必选。<br>数据类型：torch.Tensor，float32。<br>须能广播到 x。 |
| offset | 输入 | 整数 zero-point。 | `_asym` 必选。<br>数据类型：torch.Tensor。<br>`_sym` 不传。 |
| axis | 输入 | 量化轴。 | `_per_channel_*`、`_per_group_*` 必选。<br>`_per_tensor_*` 不传。<br>数据类型：int。 |
| group_size | 输入 | 分组大小。 | `_per_group_*` 必选。<br>数据类型：int。<br>axis 维长度须整除 group_size。 |
| dtype | 输入 | 量化网格存储类型。 | 可选。<br>数据类型：torch.dtype。<br>默认 None，表示 float32。<br>低精度存储常用 torch.int8。非 float32 不能反传。 |
| ste_clamp | 输入 | 饱和是否使用 STE。 | 可选。<br>数据类型：bool。<br>默认 True：前向饱和、反向恒等。<br>False 时饱和位置梯度为 0。 |
| q | 返回值 | 与 x 同形状的量化网格。 | 数据类型：由 dtype 决定，默认 float32。 |
| scale | 返回值 | 量化 scale，原样返回。 | 数据类型：torch.Tensor，float32。 |
| offset | 返回值 | 量化 offset，原样返回。 | 仅 `_asym` 返回。<br>数据类型：torch.Tensor。 |

## 调用示例

```python
import torch
from msmodelslim.core.dtype_ops import int8_quantize_per_channel_sym

x = torch.randn(4, 8)
scale = torch.tensor(0.1)
q, scale = int8_quantize_per_channel_sym(x, scale, axis=0, dtype=torch.int8)
```
