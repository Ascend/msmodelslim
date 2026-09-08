# QuantConfig

## Function

A quantization parameter configuration class that stores the parameters configured during the quantization process.

## Prototype

```python
QuantConfig(quant_mode=1, is_signed_quant=True, is_per_channel=True, calib_data=None, calib_method=0, quantize_nodes=None, exclude_nodes=None, amp_num=0, is_optimize_graph=True, is_quant_depthwise_conv=True, input_shape=None, is_dynamic_shape=False)
```

## Parameters

| Parameter| Input/Return Value | Description | Constraints |
| ------ | ------ | ------ | ------ |
| quant_mode | Input | Quantization mode. | Optional.<br>Data Type: int.<br>Value Range [0, 1], Default Value: 1.<br>1 indicates Label-Free quantization. 0 indicates Data-Free quantization. |
| is_signed_quant | Input | Whether activation uses signed quantization. | Optional.<br>Data Type: bool.<br>Default Value: True. False indicates uint8 quantization, True indicates int8 quantization.<br>It is recommended to set True for CNN models and False for Transformer models. |
| is_per_channel | Input | Whether weights use per_channel quantization. | Optional.<br>Data Type: bool.<br>Default Value: True. |
| calib_data | Input | Calibration data. | Optional.<br>Data Type: list, Default Value: [].<br>For single-input models, configure \[[input1]]; for multi-input models, configure \[[input1, input2, input3]].<br>When configured as empty, calibration data will be randomly generated. |
| calib_method | Input | Method for activation calibration. | Optional.<br>Data Type: int.<br>Value Range [0, 1, 2], Default Value: 0.<br>(1) 0 indicates min-max calibration. (2) 1 indicates Percentile. (3) 2 indicates Entropy. |
| quantize_nodes | Input | Nodes to be quantized. | Optional.<br>Data Type: list.<br>Default Value: []. This field takes effect only when the list is non-empty. |
| exclude_nodes | Input | Names of nodes excluded from quantization. | Optional.<br>Data Type: list.<br>Default Value: []. |
| amp_num | Input | Number of mixed-precision fallback layers. | Optional.<br>Data Type: int.<br>Default: 0. When accuracy drops significantly, this value can be increased to reduce the number of quantized layers. |
| is_optimize_graph | Input | Whether to perform graph optimization. | Data Type: bool. Default: True. |
| is_quant_depthwise_conv | Input | Whether to quantize the DepthwiseConv operator. | Optional.<br>Data Type: bool.<br>Default: True. When the model contains DepthwiseConv operators and quantization accuracy loss is significant, it can be set to False. |
| input_shape | Input | When the input model supports dynamic shapes, the user must specify the input_shape parameter to generate calibration data for quantization. | Optional, but must be specified when the model supports dynamic shapes.<br>Data Type: list [list [int]].<br>Default Value: `None`.<br>When the model has multiple inputs, specify input_shape in order, for example: \[[1, 3, 224, 224], [1, 3, 640, 640]]. |
| is_dynamic_shape | Input | Specifies whether the input model supports dynamic shapes. | Optional. When the input model supports dynamic shapes, the other configuration parameter input_shape must also be specified.<br>Data Type: bool.<br>Default: False.<br>True: The input model supports dynamic shapes. False: The input model has static shapes. |

## Sample

```python
from msmodelslim.onnx.post_training_quant import QuantConfig
def custom_read_data():
    calib_data = []
    # TODO: Load the dataset, perform data preprocessing, and store the data into calib_data.
    return calib_data
calib_data = custom_read_data() 
quant_config = QuantConfig(calib_data=calib_data, amp_num=5)
```
