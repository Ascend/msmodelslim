# export_quant_onnx

## Function

A quantization parameter configuration class that saves the quantized ONNX model by encapsulating the quantization algorithm through a calibrator class.

## Prototype

```python
export_quant_onnx(save_path, fuse_add=True, use_external=False)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| save_path | Input | The storage path for the quantized model. | Required.<br>Data Type: String. |
| fuse_add | Input | Whether the exported quantized model fuses the quantization bias. | Optional.<br>Data Type: bool.<br>Default: True. |
| use_external | Input | Whether to use external data to store the model. If the model size is too large (> 2 GB), this parameter must be enabled to store the model using external data. | Optional.<br>Data Type: bool.<br>Default: False. |

## Sample

```python
from msmodelslim.onnx.squant_ptq import OnnxCalibrator, QuantConfig 
quant_config = QuantConfig(disable_names=[],
                     quant_mode=0,
                     amp_num=0)
input_model_path="/path/to/Resnet50/resnet50.onnx"         # Configure according to the actual path of the model. 
output_model_path="/path/to/Resnet50/resnet50_quant.onnx"  # Configure according to the actual path of the model. 
calibrator = OnnxCalibrator(input_model_path, quant_config)
calibrator.run() 
calibrator.export_quant_onnx(output_model_path)
```
