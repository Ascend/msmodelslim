# int8_quantize_dynamic

## 功能说明

INT8 动态量化。在函数内计算量化参数并 `detach`，再完成量化。粒度由函数名后缀区分：`per_tensor`、`per_channel`、`per_group`。

## 函数原型

```python
int8_quantize_dynamic_per_tensor_sym(x, dtype=None, ste_clamp=True)
int8_quantize_dynamic_per_channel_sym(x, axis, dtype=None, ste_clamp=True)
int8_quantize_dynamic_per_group_sym(x, axis, group_size, dtype=None, ste_clamp=True)
int8_quantize_dynamic_per_tensor_asym(x, dtype=None, ste_clamp=True)
int8_quantize_dynamic_per_channel_asym(x, axis, dtype=None, ste_clamp=True)
int8_quantize_dynamic_per_group_asym(x, axis, group_size, dtype=None, ste_clamp=True)
```

## 参数说明

| 参数名 | 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| x | 输入 | 待量化浮点张量。 | 必选。<br>数据类型：torch.Tensor。<br>可转为 float32。 |
| axis | 输入 | 量化轴。 | `_per_channel_*`、`_per_group_*` 必选。<br>数据类型：int。<br>`_per_tensor_*` 不传。 |
| group_size | 输入 | 分组大小。 | `_per_group_*` 必选。<br>数据类型：int。<br>axis 维长度须整除 group_size。 |
| dtype | 输入 | 量化网格存储类型。 | 可选。<br>数据类型：torch.dtype。<br>默认 None，表示 float32。<br>低精度存储常用 torch.int8。非 float32 不能反传。 |
| ste_clamp | 输入 | 饱和是否使用 STE。 | 可选。<br>数据类型：bool。<br>默认 True：前向饱和、反向恒等。<br>False 时饱和位置梯度为 0。 |
| q | 返回值 | 与 x 同形状的量化网格。 | 数据类型：由 dtype 决定，默认 float32。 |
| scale | 返回值 | 量化 scale，已 detach。 | 数据类型：torch.Tensor，float32。 |
| offset | 返回值 | 量化 offset，已 detach。 | 仅 `_asym` 返回。<br>数据类型：torch.Tensor。 |

## 调用示例

```python
import torch
from msmodelslim.core.dtype_ops import int8_quantize_dynamic_per_channel_sym

x = torch.randn(4, 8)
q, scale = int8_quantize_dynamic_per_channel_sym(x, axis=0)
```
