# fp8_dequantize

## 功能说明

FP8 对称反量化。用量化时的 scale 把网格还原为 float32。粒度由函数名后缀区分：`per_tensor`、`per_channel`。

## 函数原型

```python
fp8_dequantize_per_tensor_sym(q, scale)
fp8_dequantize_per_channel_sym(q, scale, axis)
```

## 参数说明

| 参数名 | 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| q | 输入 | FP8 量化网格。 | 必选。<br>数据类型：torch.Tensor。<br>可为 float32 或 float8。 |
| scale | 输入 | 量化时使用的 scale。 | 必选。<br>数据类型：torch.Tensor，float32。 |
| axis | 输入 | 量化轴。 | `_per_channel_*` 必选。<br>数据类型：int。<br>`_per_tensor_*` 不传。 |
| x_hat | 返回值 | 反量化浮点张量。 | 数据类型：torch.Tensor，float32。<br>形状与 q 一致。 |

## 调用示例

```python
import torch
from msmodelslim.core.dtype_ops import fp8_dequantize_per_channel_sym

q = torch.randn(2, 8)
scale = torch.tensor(0.1)
x_hat = fp8_dequantize_per_channel_sym(q, scale, axis=-1)
```
