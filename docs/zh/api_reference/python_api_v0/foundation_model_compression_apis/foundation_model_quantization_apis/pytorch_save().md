# save()

## 功能说明

量化参数配置类，通过calibrator类封装量化算法来保存量化后的权重及相关参数。

说明：因为在存储量化参数过程中存在反序列化风险，所以已通过在存储过程中，将保存的量化结果文件夹权限设置为750，量化权重文件权限设置为400，量化权重描述文件设为600来消减风险。

## 函数原型

```python
calibrator.save(output_path, safetensors_name=None, json_name=None, save_type=None, part_file_size=None)
```

## 参数说明

| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| output_path | 输入 | 量化后的权重及相关参数保存路径。| 必选。<br>数据类型：string。|
| safetensors_name | 输入 | safetensors格式量化权重文件的名称。| 可选。<br>数据类型：string。|
| json_name | 输入 | safetensors格式量化权重json描述文件的名称。| 可选。<br>数据类型：string。|
| part_file_size | 输入 | 保存成safetensors权重文件时，每个部分的大小，单位为GB。| 可选。<br>数据类型：int。<br>参数默认为None，不启用分片保存的功能。|
| save_type | 输入 | 量化后权重的保存格式。| 可选。<br>数据类型：list，元素类型：string。<br>参数默认为["safe_tensor"], 量化权重保存格式为safetensors。|

## 参数补充说明

- part_file_size
<br>该参数设置为大于0的整数时，开启分片保存功能。将会按照用户设置的值（GB）进行分片，实际保存的权重可能会略大于设置的值。
- save_type
<br>该参数支持设置为"numpy"，"safe_tensor"或"ascendV1"三种格式。用户需按照模型权重保存的实际情况进行选择：
<br>（1）设置为"numpy"时，仅导出npy格式的量化权重文件。`注意：量化类型为W4A8_DYNAMIC时，设置"numpy"格式保存会报错。`
<br>（2）设置为"safe_tensor"时，仅导出safetensors的量化权重文件和json描述文件。safetensors权重文件包含浮点、量化权重，量化层使用的量化权重和未量化层使用的原始浮点权重。json描述文件包含模型的所有module，并标明该module的量化或浮点类型，例如FLOAT、W8A8、W8A16。
<br>（3）设置为"ascendV1"时，模型权重与使用safe_tensor导出时的保持一致，但config.json和quant_model_description_{quant_type}.json会有略微改动。`注意：量化类型为 W4A16 时，默认会对权重进行pack。`配置文件的改动说明如下：config.json文件内容跟浮点权重保持一致；同时quant_model_description_{quant_type}.json文件名变为quant_model_description.json，并新增了一个version说明。

## 调用示例

根据实际需求，在QuantConfig初始化中完成所有参数的配置。

```python
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
quant_config = QuantConfig(dev_type='cpu', pr=0.5, mm_tensor=False)
model = AutoModel.from_pretrained('/chatglm2-6b', local_files_only=True, torch_dtype=torch.float32).cpu()   #根据模型实际路径配置
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
calibrator.run(int_infer=False)
quant_weight_save_path = "/path/to/save_path" # 根据实际路径配置
calibrator.save(quant_weight_save_path)
```
