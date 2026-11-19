# int8_calc_qparam

## 功能说明

计算 INT8 量化参数。对称接口返回 scale，非对称接口返回 (scale, offset)。粒度由函数名后缀区分：`per_tensor`、`per_channel`、`per_group`。

## 函数原型

```python
int8_calc_qparam_per_tensor_sym(x)
int8_calc_qparam_per_channel_sym(x, axis)
int8_calc_qparam_per_group_sym(x, axis, group_size)
int8_calc_qparam_per_tensor_asym(x)
int8_calc_qparam_per_channel_asym(x, axis)
int8_calc_qparam_per_group_asym(x, axis, group_size)
```

## 参数说明

| 参数名 | 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| x | 输入 | 待量化浮点张量。 | 必选。<br>数据类型：torch.Tensor。<br>可转为 float32。 |
| axis | 输入 | 量化轴。 | `_per_channel_*`、`_per_group_*` 必选。<br>`_per_tensor_*` 不传。<br>数据类型：int。 |
| group_size | 输入 | 分组大小。 | `_per_group_*` 必选。<br>数据类型：int。<br>axis 维长度须整除 group_size。 |
| scale | 返回值 | 量化 scale。 | 数据类型：torch.Tensor，float32。<br>对称：scale = amax / 127。<br>非对称：scale = (xmax - xmin) / 255。<br>区间强制包含 0。 |
| offset | 返回值 | 整数 zero-point。 | 仅 `_asym` 返回。<br>数据类型：torch.Tensor。<br>公式：offset = round(-xmin / scale) + (-128)。 |

## 调用示例

```python
import torch
from msmodelslim.core.dtype_ops import (
    int8_calc_qparam_per_channel_sym,
    int8_calc_qparam_per_group_asym,
)

x = torch.randn(4, 8)
scale = int8_calc_qparam_per_channel_sym(x, axis=0)
scale, offset = int8_calc_qparam_per_group_asym(x, axis=-1, group_size=4)
```
