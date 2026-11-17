# fp8_calc_qparam

## 功能说明

计算 FP8 对称量化参数，返回 scale = amax / max_norm。粒度由函数名后缀区分：`per_tensor`、`per_channel`。无非对称接口。

## 函数原型

```python
fp8_calc_qparam_per_tensor_sym(x, fmt="e4m3")
fp8_calc_qparam_per_channel_sym(x, axis, fmt="e4m3")
```

## 参数说明

| 参数名 | 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| x | 输入 | 待量化浮点张量。 | 必选。<br>数据类型：torch.Tensor。<br>可转为 float32。 |
| axis | 输入 | 量化轴，该维每个通道一份 scale。 | `_per_channel_*` 必选。<br>数据类型：int。<br>`_per_tensor_*` 不传。 |
| fmt | 输入 | 元素编码。 | 可选。<br>数据类型：string。<br>默认 e4m3（上界 448）。可选 e5m2（上界 57344）。<br>仅 FP8 / MXFP8 接口有此参数。须与后续量化一致。 |
| scale | 返回值 | 对称量化 scale。 | 数据类型：torch.Tensor，float32。<br>公式：scale = amax / max_norm。 |

## 调用示例

```python
import torch
from msmodelslim.core.dtype_ops import fp8_calc_qparam_per_channel_sym

x = torch.randn(2, 8)
scale = fp8_calc_qparam_per_channel_sym(x, axis=-1, fmt="e4m3")
```
