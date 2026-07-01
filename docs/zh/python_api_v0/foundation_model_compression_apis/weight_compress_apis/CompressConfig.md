# CompressConfig

## 功能说明

权重压缩的参数配置类，保存权重压缩过程中配置的参数。

## 函数原型

```python
CompressConfig(do_pseudo_sparse=False, sparse_ratio=1, is_debug=False, compress_disable_layers=None, record_detail_root='./', multiprocess_num=1)
```

## 参数说明

| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| do_pseudo_sparse | 输入 | 是否进行模拟权重稀疏。| 可选。<br>数据类型：bool。<br>默认为False。 |
| sparse_ratio | 输入 | 模拟权重稀疏时的稀疏率。| 可选。<br>数据类型：float。<br>默认为1，可选范围为[0, 1]。 |
| is_debug | 输入 | 是否输出详细日志信息。| 可选。<br>数据类型：bool。<br>默认为False。 |
| compress_disable_layers | 输入 | 无需进行压缩指定层的权重。| 可选。<br>数据类型：list或tuple。<br>默认为None。 |
| record_detail_root | 输入 | 压缩过程的中间结果存放路径。| 可选。<br>数据类型：str。<br>默认为"./"。 |
| multiprocess_num | 输入 | 多进程压缩模式下开启进程数量。| 可选。<br>数据类型：int。<br>默认值为1，使用单进程压缩模式。<br>说明：用户需要根据设备环境设置multiprocess_num参数，以达到加速压缩权重的效果。设置过大的multiprocess_num值，可能会导致申请的系统资源不足，从而造成压缩程序运行失败。建议用户先设置当前系统可以打开的最大进程数，例如 ulimit -n 65536；然后再检查当前设备的NPU使用状况，确保无较大显存占用。multiprocess_num的建议值为8。 |

## 调用示例

```python
from msmodelslim.pytorch.weight_compression import CompressConfig
compress_config = CompressConfig(do_pseudo_sparse=False, sparse_ratio=1, is_debug=True, compress_disable_layers=None, record_detail_root='./record_root', multiprocess_num=8)
```
