# export()

## 功能说明

权重压缩参数配置类，通过Compressor类封装压缩算法来保存压缩后的权重及相关参数。

说明：若在Compressor中使用'weight_path'参数，需通过'compressor.export'函数将压缩后的权重导出。

## 函数原型

```python
compressor.export(arr, path, dtype=numpy.int8)
```

## 参数说明

| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
|arr|输入|compressor.run函数的返回结果。|必选。<br>数据类型：dict。|
|path|输入|压缩结果的保存路径。|必选。<br>数据类型：str。|
|dtype|输入|压缩结果的保存格式。|可选。<br>数据类型：numpy.dtype。<br>默认值：numpy.int8。|

## 调用示例

- 使用weight_path参数指定待压缩的权重文件路径。

```python
import numpy
from msmodelslim.pytorch.weight_compression import CompressConfig, Compressor
compress_config = CompressConfig(do_pseudo_sparse=False, sparse_ratio=1, is_debug=True, compress_disable_layers=None, record_detail_root='./record_root')
weight_save_path = './quant_weight.npy'  # 根据实际情况修改待压缩的权重文件路径
compressor = Compressor(compress_config, weight_save_path)
compress_result_weight, compress_result_index, compress_result_info = compressor.run()
compressor.export(compress_result_weight, './compress_weight')
compressor.export(compress_result_index, './compress_index')
compressor.export(compress_result_info, './compress_info', dtype=numpy.int64)
```
