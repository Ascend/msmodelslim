# Compressor

## 功能说明

权重压缩参数配置类，通过Compressor类封装压缩算法。

## 函数原型

```python
Compressor(config: CompressConfig, weight_path=None, weight=None, quant_model_description=None)
```

## 参数说明

| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| config | 输入 | 已配置的CompressConfig类。| 必选。<br>数据类型：CompressConfig。 |
| weight_path | 输入 | 需要压缩的模型权重文件路径。| 必选。与 (`weight` + `quant_model_description`) 二选一。<br>数据类型：String。<br>说明：当save()中输出save_type参数为numpy格式时，使用该方式传入权重文件。导出压缩后权重文件需要使用export()。 |
| weight | 输入 | 量化工具生成的稀疏量化权重。| 必选。需与`quant_model_description`同时传入；与`weight_path`二选一。<br>数据类型：dict。<br>说明：当save()中输出save_type参数为safe_tensor格式时，使用该方式传入权重文件。导出压缩后权重文件需要使用export_safetensors()。|
| quant_model_description | 输入 | 量化权重描述文件。| 必选。需与`weight`同时传入；与`weight_path`二选一。<br>数据类型：dict。<br>说明：当save()中输出save_type参数为safe_tensor格式时，使用该方式传入权重文件。导出压缩后权重文件需要使用export_safetensors()。|

## 调用示例

- 使用weight_path参数创建Compressor对象。

```python
from msmodelslim.pytorch.weight_compression import CompressConfig, Compressor
compress_config = CompressConfig(do_pseudo_sparse=False, sparse_ratio=1, is_debug=True, compress_disable_layers=None, record_detail_root="./", multiprocess_num=1)
weight_save_path = './quant_weight.npy'  # 根据实际情况修改待压缩的权重文件路径
compressor = Compressor(config=compress_config, weight_path=weight_save_path)
```

- 使用weight、quant_model_description参数创建Compressor对象。

```python
from safetensors.torch import load_file
import json
# 导入权重压缩接口
from msmodelslim.pytorch.weight_compression import CompressConfig, Compressor
# 准备待压缩权重文件和相关压缩配置，请根据实际情况进行修改
weight_path = "./quant_model_weight_w8a8s.safetensors"       # 待压缩权重文件的路径
json_path = "./quant_model_description_w8a8s.json"          # 待压缩权重文件的描述文件的路径
# 使用CompressConfig接口，配置压缩参数，并返回配置实例
compress_config = CompressConfig(do_pseudo_sparse=False, sparse_ratio=1, is_debug=True, compress_disable_layers=None, record_detail_root="./", multiprocess_num=1)
sparse_weight = load_file(weight_path)
with open(json_path, 'r') as f:
    quant_model_description = json.load(f)
#使用Compressor接口，输入加载的压缩配置和待压缩权重文件
compressor = Compressor(config=compress_config, weight=sparse_weight, quant_model_description=quant_model_description)
```
