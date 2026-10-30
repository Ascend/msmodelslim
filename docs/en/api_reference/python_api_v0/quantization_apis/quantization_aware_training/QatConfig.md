# QatConfig

## Function

Quantization parameter configuration class, used to save the parameters configured during the quantization process.

## Prototype

```python
QatConfig(w_bit=8, a_bit=8, a_sym=False, amp_num=0, steps=1, ema=0.99, is_forward=False, ignore_head_tail_node=False, disable_names=None, has_init_quant=False, quant_mode=True, grad_scale=0.0, compressed_model_checkpoint=None, opset_version=11, save_params=False, input_names=None, output_names=None, save_onnx_name=None)
```

## Parameters

| Parameter Name | Optional/Mandatory | Description |
|---------| ------ |----------------------------------------------------------|
| w_bit | Optional | Weight quantization bit.<br>Data type: int. Default is 8, and modification is not supported.<br>Input/Return: Input. |
| a_bit | Optional | Activation layer quantization bit.<br>Data type: int. Default is 8, and modification is not supported.<br>Input/Return: Input. |
| a_sym | Optional | Whether to use symmetric quantization for activation values.<br>Data type: bool. Default is False.<br>Input/Return: Input. |
| amp_num | Optional | Number of automatic fallback layers.<br>When accuracy drops significantly, you can increase the number of fallback layers. It is recommended to prioritize fallback of 1 to 3 layers. If accuracy recovery is not significant, then increase the number of fallback layers.<br>Data type: int. Value range: [0,10]. Default is 0. Inputs such as 1, 2, and 3 are allowed.<br>Input/Return: Input. |
| steps | Optional | Number of steps for automatic fallback.<br>Data type: int. Default is 1, value range is greater than or equal to 1.<br>Input/Return: Input. |
| ema | Optional | Parameter in the Adam optimizer, the exponential moving average.<br>Data type: float. Value range: [0.1, 1.0]. Defaults to 0.99.<br>Input/Return: Input. |
| is_forward | Optional | Whether to process the forward pass with reference to mmdetection.<br>Data type: bool. Default is False.<br>Input/Return: Input. |
| ignore_head_tail_node | Optional | Whether to ignore the first and last layers and not quantize them.<br>Data type: bool. Default is False.<br>Input/Return: Input. |
| disable_names | Optional | Names of nodes to be excluded from quantization, i.e., the names of quantization layers for manual fallback.<br>If the accuracy is poor, you can select quantization layers for fallback.<br>Data type: list[str]. Default is None.<br>Input/Return: Input. |
| has_init_quant | Optional | Whether the model has undergone quantization initialization.<br>Data type: bool. Default is False.<br>Input/Return: Input. |
| quant_mode | Optional | Whether to enable quantization mode.<br>Data type: bool. Default is True.<br>Input/Return: Input. |
| grad_scale | Optional | Gradient compensation strength.<br>Data type: float. Default is 0.0, and the recommended configuration is 0.001.<br>Input/Return: Input. |
| compressed_model_checkpoint | Optional | The weight file and its path for the pseudo-quantized model saved when exporting the ONNX model.<br>Data type: string. Default is None.<br>Input/Return: Input. |
| opset_version | Optional | Version number when exporting the ONNX model. The corresponding ONNX version must be installed in advance.<br>Data type: int. Optional values are 11 and 13, and the default is 11.<br>Input/Return: Input. |
| save_params | Optional | Whether to save quantization-related parameters as .npy files during export.<br>Data type: bool. Default is False.<br>Input/Return: Input. |
| input_names | Optional | Input names for ONNX.<br>Data type: list[str]. Default is None.<br>Input/Return: Input. |
| output_names | Optional | Output names for ONNX.<br>Data type: list[str]. Default is None.<br>Input/Return: Input. |
| save_onnx_name | Optional | Pseudo-quantized model weights.<br>Data type: str. Default is None.<br>Input/Return: Input. |

## Sample

```python
from msmodelslim.pytorch.quant.qat_tools import QatConfig
quant_config = QatConfig(grad_scale=0.001)
```
