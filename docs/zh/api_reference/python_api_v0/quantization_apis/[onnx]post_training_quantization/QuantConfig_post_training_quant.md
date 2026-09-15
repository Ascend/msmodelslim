# QuantConfig

## 功能说明

量化参数配置类，保存量化过程中配置的参数。

## 函数原型

```python
QuantConfig(quant_mode=1, is_signed_quant=True, is_per_channel=True, calib_data=None, calib_method=0, quantize_nodes=None, exclude_nodes=None, amp_num=0, is_optimize_graph=True, is_quant_depthwise_conv=True, input_shape=None, is_dynamic_shape=False)
```

## 参数说明

| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| quant_mode | 输入 | 量化模式。| 可选。<br>数据类型：int。<br>可选值[0, 1]，默认值为1。<br>1表示Label-Free量化。0表示Data-Free量化。|
| is_signed_quant | 输入 | 激活是否符号量化。| 可选。<br>数据类型：bool。<br>默认值为True，False表示uint8量化，True表示int8量化。<br>CNN类模型建议配置True，Transformer类模型建议配置False。|
| is_per_channel | 输入 | 权重是否per_channel量化。| 可选。<br>数据类型：bool。<br>默认值为True。|
| calib_data | 输入 | 校正数据。| 可选。<br>数据类型：list，默认值为[]。<br>对于单输入模型，配置\[[input1]]，多输入模型，配置\[[input1,input2,input3]]。<br>配置为空时，将随机生成校正数据。|
| calib_method | 输入 | 激活校正的方法。| 可选。<br>数据类型：int。<br>可选值[0,1,2]，默认值为0。<br>(1)0表示min-max校正。(2)1表示Percentile。(3)2表示Entropy。|
| quantize_nodes | 输入 | 需要量化的节点。| 可选。<br>数据类型：list。<br>默认值为[]。仅当列表为非空时，该字段生效。|
| exclude_nodes | 输入 | 排除量化的节点名称。| 可选。<br>数据类型：list。<br>默认值为[]。|
| amp_num | 输入 | 混合精度回退层数。| 可选。<br>数据类型：int。<br>默认为0。精度降低过多时，可以增大此值，以减少量化的层数。|
| is_optimize_graph | 输入 | 是否进行图优化。| 数据类型：bool，默认为True。|
| is_quant_depthwise_conv | 输入 | 是否量化DepthwiseConv算子。|可选。<br>数据类型：bool。<br>默认为True。当模型中有DepthwiseConv算子，量化精度损失较大时，可以配置为False。|
| input_shape | 输入 | 当输入模型支持动态shape时，用户需指定input_shape参数，用以生成量化时的校准数据。|可选，当模型支持动态shape时必须指定。<br>数据类型：list [list [int]]。<br>默认值：None。<br>当模型有多个输入时，按照顺序指定input_shape，例如：\[[1, 3,224, 224], [1, 3, 640, 640]]。|
| is_dynamic_shape | 输入 | 指定输入的模型是否支持动态shape。|可选。输入模型支持动态shape时，另一配置参数input_shape也必须指定。<br>数据类型：bool。<br>默认为False。<br>True：输入的模型支持动态shape。False：输入的模型为静态shape。|

## 调用示例

```python
from msmodelslim.onnx.post_training_quant import QuantConfig
def custom_read_data():
    calib_data = []
    # TODO 读取数据集，进行数据预处理，将数据存入calib_data
    return calib_data
calib_data = custom_read_data()
quant_config = QuantConfig(calib_data=calib_data, amp_num=5)
```
