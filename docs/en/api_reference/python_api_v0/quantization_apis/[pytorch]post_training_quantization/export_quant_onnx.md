# export_quant_onnx

## Function

Quantization parameter configuration class that encapsulates the quantization algorithm through a calibrator class to save the quantized ONNX model.

## Prototype

```python
export_quant_onnx(model_arch, save_path, input_names=None, fuse_add=True, save_fp=False)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
|model_arch|Input|Name of the model architecture.|Required.<br>Data Type: String. |
|save_path|Input|Storage path for the quantized model.|Required.<br>Data Type: String. |
|input_names|Input|Input names for ONNX. If there are N inputs, N names must be provided.|Optional.<br>Data Type: list[str].<br>By default, ONNX names are assigned sequentially as input.1, input.2, ..., input.N, with the numeric suffix incrementing based on the number of model inputs.|
|fuse_add|Input|Whether to fuse quantization bias in the exported quantized model.|Optional.<br>Data Type: bool.<br>Default is True. |
|save_fp|Input|Whether to retain the pre-quantization ONNX model.|Optional.<br>Data Type: bool.<br>Default is False. |

## Sample

```python
from msmodelslim.pytorch.quant.ptq_tools import QuantConfig, Calibrator
disable_names = []
input_shape = [1, 3, 224, 224]
quant_config = QuantConfig(disable_names=disable_names, amp_num=0, input_shape=input_shape)
calibrator = Calibrator(model, quant_config)
calibrator.run()
calibrator.export_quant_onnx("model", "./output", ["input.1"])
```
