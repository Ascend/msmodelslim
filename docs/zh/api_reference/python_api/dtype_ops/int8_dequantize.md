# int8_dequantize

## 功能说明

INT8 反量化。用量化时的参数把网格还原为 float32。粒度由函数名后缀区分：`per_tensor`、`per_channel`、`per_group`。

## 函数原型

```python
int8_dequantize_per_tensor_sym(q, scale)
int8_dequantize_per_channel_sym(q, scale, axis)
int8_dequantize_per_group_sym(q, scale, axis, group_size)
int8_dequantize_per_tensor_asym(q, scale, offset)
int8_dequantize_per_channel_asym(q, scale, offset, axis)
int8_dequantize_per_group_asym(q, scale, offset, axis, group_size)
```

## 参数说明

| 参数名 | 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| q | 输入 | INT8 量化网格。 | 必选。<br>数据类型：torch.Tensor。<br>可为 float32 或 torch.int8。 |
| scale | 输入 | 量化时使用的 scale。 | 必选。<br>数据类型：torch.Tensor，float32。 |
| offset | 输入 | 量化时使用的 zero-point。 | `_asym` 必选。<br>数据类型：torch.Tensor。<br>`_sym` 不传。 |
| axis | 输入 | 量化轴。 | `_per_channel_*`、`_per_group_*` 必选。<br>数据类型：int。<br>`_per_tensor_*` 不传。 |
| group_size | 输入 | 分组大小。 | `_per_group_*` 必选。<br>数据类型：int。<br>须与量化时一致。 |
| x_hat | 返回值 | 反量化浮点张量。 | 数据类型：torch.Tensor，float32。<br>形状与 q 一致。 |

## 调用示例

```python
import torch
from msmodelslim.core.dtype_ops import int8_dequantize_per_channel_sym

q = torch.randint(-128, 128, (4, 8)).float()
scale = torch.tensor(0.1)
x_hat = int8_dequantize_per_channel_sym(q, scale, axis=0)
```
