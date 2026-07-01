# QatConfig

## Function

Quantization parameter configuration class, which saves the parameters configured during quantization.

## Prototype

```python
QatConfig(w_bit=8, a_bit=8, a_sym=False, amp_num=0, steps=1, ema=0.99, is_forward=False, ignore_head_tail_node=False, disable_names=None, has_init_quant=False, quant_mode=True, grad_scale=0.0, compressed_model_checkpoint=None, opset_version=11, save_params=False, input_names=None, output_names=None, save_onnx_name=None)
```

## Parameters

|Parameter|Optional/Required|Description|
|---------| ------ |----------------------------------------------------------|
|w_bit|Optional|Weight quantization bit.<br>Data type: int. The default value is `8` and cannot be modified.<br>Input/Return: input value.|
|a_bit|Optional|Quantization bit of the activation layer.<br>Data type: int. The default value is `8` and cannot be modified.<br>Input/Return: input value.|
|a_sym|Optional| Specifies whether to use symmetric quantization for activation values.<br>Data type: bool. Default value: `False`.<br>Input/Return: input value.|
|amp_num|Optional|Number of automatic fallback layers.<br>When accuracy decreases significantly, increase the number of fallback layers. You are advised to roll back one to three layers first. If accuracy recovery is not obvious, increase the number of fallback layers further.<br>Data type: int. Value range: [0, 10]. Default value: `0`. Inputs such as 1, 2, or 3 are supported.<br>Input/Return: input value.|
|steps|Optional|Number of automatic fallback steps.<br>Data type: int. Default value: `1`. The value must be greater than or equal to 1.<br>Input/Return: input value.|
|ema|Optional|Exponential moving average parameter in the Adam optimizer.<br>Data type: float. Value range: [0.1,1.0]. Default value: `0.99`.<br>Input/Return: input value.|
|is_forward|Optional|Specifies whether to handle the forward pass based on `mmdetection`.<br>Data type: bool. Default value: `False`.<br>Input/Return: input value.|
|ignore_head_tail_node|Optional|Specifies whether to ignore the head and tail layers to exclude them from quantization.<br>Data type: bool. Default value: `False`.<br>Input/Return: input value.|
|disable_names|Optional|Node names excluded from quantization, representing manually rolled back quantization layers.<br> If accuracy drops significantly, select these quantization layers to roll back.<br>Data type: list[str]. Default value: `None`.<br>Input/Return: input value.|
|has_init_quant|Optional|Indicates whether the model has undergone quantization initialization.<br>Data type: bool. Default value: `False`.<br>Input/Return: input value.|
|quant_mode|Optional|Specifies whether to enable quantization mode.<br>Data type: bool. Default value: `True`.<br>Input/Return: input value.|
|grad_scale|Optional|Gradient compensation strength.<br>Data type: float. Default value: `0.0`. Recommended value: `0.001`.<br>Input/Return: input value.|
|compressed_model_checkpoint|Optional|Weight file path of the fake-quantized model saved during ONNX model export.<br>Data type: string. Default value: `None`.<br>Input/Return: input value.|
|opset_version|Optional|Version number used during ONNX model export. The corresponding ONNX version must be installed in advance.<br>Data type: int. Valid values: `11` or `13`. Default value: `11`.<br>Input/Return: input value.|
|save_params|Optional|Specifies whether to save quantization parameters as an .npy file during export.<br>Data type: bool. Default value: `False`.<br>Input/Return: input value.|
|input_names|Optional|Input names of the ONNX model.<br>Data type: list[str]. Default value: `None`.<br>Input/Return: input value.|
|output_names|Optional|Output names of the ONNX model.<br>Data type: list[str]. Default value: `None`.<br>Input/Return: input value.|
|save_onnx_name|Optional|Weight of the fake quantized model.<br>Data type: str. Default value: `None`.<br>Input/Return: input value.|

## Sample

```python
from msmodelslim.pytorch.quant.qat_tools import QatConfig
quant_config = QatConfig(grad_scale=0.001)
```
