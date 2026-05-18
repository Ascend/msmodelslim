# QatConfig

## 功能说明

量化参数配置类，保存量化过程中配置的参数。

## 函数原型

```python
QatConfig(w_bit=8, a_bit=8, a_sym=False, amp_num=0, steps=1, ema=0.99, is_forward=False, ignore_head_tail_node=False, disable_names=None, has_init_quant=False, quant_mode=True, grad_scale=0.0, compressed_model_checkpoint=None, opset_version=11, save_params=False, input_names=None, output_names=None, save_onnx_name=None)
```

## 参数说明

|参数名|可选/必选|参数说明|
|---------| ------ |----------------------------------------------------------|
|w_bit|可选|权重量化bit。<br>数据类型：int。默认为8，不支持修改。<br>输入值/返回值：输入值。|
|a_bit|可选|激活层量化bit。<br>数据类型：int。默认为8，不支持修改。<br>输入值/返回值：输入值。|
|a_sym|可选| 激活值是否对称量化。<br>数据类型：bool。默认为False。<br>输入值/返回值：输入值。|
|amp_num|可选|自动回退层数。<br>精度降低过多时，可增加回退层数，推荐优先回退1~3层，如果精度恢复不明显，再增加回退层数。<br>数据类型：int。取值范围为[0,10]，默认为0，可输入1、2、3等。<br>输入值/返回值：输入值。|
|steps|可选|自动回退的步数。<br>数据类型：int。默认为1，取值范围大于等于1。<br>输入值/返回值：输入值。|
|ema|可选|Adam优化器中参数，指数移动平均数指标。<br>数据类型：float。取值范围为[0.1,1.0]，默认为0.99。<br>输入值/返回值：输入值。|
|is_forward|可选|是否参考mmdetection对前向进行处理。<br>数据类型：bool。默认为False。<br>输入值/返回值：输入值。|
|ignore_head_tail_node|可选|是否将首尾层忽略，不进行量化。<br>数据类型：bool。默认为False。<br>输入值/返回值：输入值。|
|disable_names|可选|需排除量化的节点名称，即手动回退的量化层名称。<br> 如精度太差，可以选择回退的量化层。<br>数据类型：list[str]。默认为None。<br>输入值/返回值：输入值。|
|has_init_quant|可选|模型是否做过量化初始化。<br>数据类型：bool。默认为False。<br>输入值/返回值：输入值。|
|quant_mode|可选|是否开启量化模式。<br>数据类型：bool。默认值为True。<br>输入值/返回值：输入值。|
|grad_scale|可选|梯度补偿力度。<br>数据类型：float。默认值为0.0，建议配置为0.001。<br>输入值/返回值：输入值。|
|compressed_model_checkpoint|可选|导出ONNX模型时，保存的伪量化模型权重文件及所在路径。<br>数据类型：string。默认为None。<br>输入值/返回值：输入值。|
|opset_version|可选|导出ONNX模型时版本号,需提前安装对应的ONNX版本。<br>数据类型：int。可选值为11和13，默认为11。<br>输入值/返回值：输入值。|
|save_params|可选|导出时是否将量化相关参数保存为npy文件。<br>数据类型：bool。默认为False。<br>输入值/返回值：输入值。|
|input_names|可选|ONNX的输入名称。<br>数据类型：list[str]。默认为None。<br>输入值/返回值：输入值。|
|output_names|可选|ONNX的输出名称。<br>数据类型：list[str]。默认为None。<br>输入值/返回值：输入值。|
|save_onnx_name|可选|伪量化模型权重。<br>数据类型：str。默认为None。<br>输入值/返回值：输入值。|

## 调用示例

```python
from msmodelslim.pytorch.quant.qat_tools import QatConfig
quant_config = QatConfig(grad_scale=0.001)
```
