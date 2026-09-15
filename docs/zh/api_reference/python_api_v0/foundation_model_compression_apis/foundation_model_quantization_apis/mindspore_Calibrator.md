# Calibrator

## 功能说明

量化参数配置类，通过Calibrator类封装量化算法。

## 函数原型

```python
Calibrator(cfg: QuantConfig, model, model_ckpt, calib_data=None)
```

## 参数说明

| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| cfg | 输入 | 已配置的QuantConfig类。| 必选。<br>数据类型：QuantConfig。|
| model | 输入 | 模型。|  必选。<br>数据类型：MindFormer 模型。 |
| model_ckpt | 输入 | 模型权重的ckpt文件。|  必选。<br>数据类型：str。 |
| calib_data | 输入 | LLM大模型量化校准的数据，输入真实数据用于量化。|  可选。<br>数据类型：object。<br>默认值为None。<br>输入模板：\[[input1],[input2],[input3]]。 |

## 调用示例

```python
from msmodelslim.mindspore.llm_ptq import Calibrator, QuantConfig
quant_config = QuantConfig(disable_names=["lm_head"], fraction=0.01)
model = Model()  #根据模型实际情况进行加载
calibrator = Calibrator(cfg=quant_config, model=model, model_ckpt="./model.ckpt", calib_data=dataset_calib)
```
