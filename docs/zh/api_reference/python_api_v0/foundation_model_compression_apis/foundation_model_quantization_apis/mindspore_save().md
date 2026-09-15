# save()

## 功能说明

量化参数配置类，通过calibrator类封装量化算法来保存量化后的权重及相关参数。

说明：因为在存储量化参数过程中存在反序列化风险，所以需要在存储过程中，将保存的量化结果文件夹权限设置为750，量化权重文件权限设置为400，量化权重描述文件权限设为600来减少风险。

## 函数原型

```python
calibrator.save(output_path="")
```

## 参数说明

| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| output_path | 输入 | 量化后的权重及相关参数保存路径。| 必选。<br>数据类型：string。|

## 调用示例

```python
from msmodelslim.mindspore.llm_ptq import Calibrator, QuantConfig
quant_config = QuantConfig(disable_names=["lm_head"], fraction=0.01)
model = Model()    #根据模型实际情况进行加载
calibrator = Calibrator(cfg=quant_config, model=model, model_ckpt="./model.ckpt", calib_data=dataset_calib)
calibrator.run()
calibrator.save("./quant_model.ckpt")
```
