# OnnxCalibrator

## Function

A quantization parameter configuration class that encapsulates quantization algorithms through the Calibrator class.

## Prototype

```python
OnnxCalibrator(input_model, cfg: QuantConfig, calib_data=None)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| input_model | Input | The path and filename where the model to be quantized is stored.| Required.<br>Data Type: String.|
| cfg | Input | The configured QuantConfig class.| Required.<br>Data Type: QuantConfig.|
| calib_data | Input | Model training data. Real data can be input for Label-Free quantization, or virtual data can be input to achieve Label-Free quantization.| Optional.<br>Data Type: list, default value is [].<br>For a single-input model, configure \[[input1]]; for a multi-input model, configure \[[input1,input2,input3]].<br>(1) In a single-input scenario, data input may not be necessary. If the model supports a single float format input and input_shape is specified, the Label-Free quantization process will be invoked automatically.<br>(2) For models with multiple inputs or those requiring a custom input format, the user must manually input data to achieve Label-Free quantization. Template example: calib_data = \[[np.random.random(size=(1, 3, 127, 127)).astype(np.float32), np.random.random(size=(1, 3, 255, 255)).astype(np.float32)]].|

## Sample

```python
from msmodelslim.onnx.squant_ptq import OnnxCalibrator, QuantConfig 
quant_config = QuantConfig(disable_names=[],
                     quant_mode=0,
                     amp_num=0)
input_model_path="/home/xxx/Resnet50/resnet50_pytorch.onnx"   # Configure according to the actual path of the model.
calib = OnnxCalibrator(input_model_path, quant_config)
```
